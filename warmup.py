"""
Generate expert training data from heuristic agents.

Bootstraps the neural network by pre-training on games from agents
that already know basic Quoridor strategy (advance pawn, block opponent).
This is what the original AlphaGo did — learn from experts first,
then improve via self-play.

Usage:
    python warmup.py                     # generate data + pre-train
    python warmup.py --games 2000        # more games
    python warmup.py --generate-only     # just generate data, don't train
"""

import argparse
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from game import QuoridorGame, MoveEncoder
from network import QuoridorNet, count_parameters
from config import Config


# =====================================================================
# Heuristic agents that play reasonable Quoridor
# =====================================================================

class ExpertAgent:
    """
    A strong heuristic agent that plays like a competent human:
    - Usually advances along shortest path
    - Sometimes places walls to block opponent's path
    - Adapts wall placement to game state
    - Makes the wall/move tradeoff based on relative distance

    This isn't optimal play, but it's *reasonable* play — exactly
    what the network needs to see to bootstrap past random noise.
    """

    def choose_move(self, game: QuoridorGame) -> tuple:
        player = game.current_player
        opponent = 1 - player

        my_dist = game.shortest_path_length(player)
        opp_dist = game.shortest_path_length(opponent)
        my_walls = game.walls_remaining[player]

        # If I can win in 1 move, always take it
        if my_dist == 1:
            path = game.shortest_path(player)
            if path and len(path) > 1:
                move = ("move", path[1])
                if move in game.get_legal_pawn_moves():
                    return move

        # Decide: advance or place wall?
        # Place wall if opponent is ahead or equal AND we have walls
        should_wall = False
        if my_walls > 0:
            if opp_dist is not None and my_dist is not None:
                # Wall more likely when opponent is ahead
                if opp_dist <= my_dist:
                    should_wall = random.random() < 0.5
                elif opp_dist <= my_dist + 2:
                    should_wall = random.random() < 0.25
                else:
                    should_wall = random.random() < 0.1

                # Always try to wall if opponent is about to win
                if opp_dist <= 2:
                    should_wall = random.random() < 0.7

        if should_wall:
            wall = self._find_best_wall(game, player, opponent)
            if wall:
                return wall

        # Default: advance along shortest path
        return self._advance(game, player)

    def _find_best_wall(self, game, player, opponent):
        """Find a wall that maximally increases opponent's distance."""
        opp_dist_before = game.shortest_path_length(opponent)
        if opp_dist_before is None:
            return None

        opp_path = game.shortest_path(opponent)
        if not opp_path or len(opp_path) < 2:
            return None

        best_wall = None
        best_increase = 0

        # Check walls near opponent's path
        for cell in opp_path[1:5]:
            tr, tc = cell
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    wr, wc = tr + dr, tc + dc
                    for o in ("H", "V"):
                        if game.is_valid_wall(wr, wc, o):
                            g2 = game.clone()
                            g2.make_move(("wall", (wr, wc, o)))
                            new_dist = g2.shortest_path_length(opponent)
                            if new_dist is not None:
                                increase = new_dist - opp_dist_before
                                # Also check we don't hurt ourselves too much
                                my_new_dist = g2.shortest_path_length(player)
                                my_old_dist = game.shortest_path_length(player)
                                if my_new_dist is not None and my_old_dist is not None:
                                    self_cost = my_new_dist - my_old_dist
                                    net_benefit = increase - self_cost
                                    if net_benefit > best_increase:
                                        best_increase = net_benefit
                                        best_wall = ("wall", (wr, wc, o))

        if best_wall and best_increase >= 1:
            return best_wall
        return None

    def _advance(self, game, player):
        """Move pawn along shortest path, with some randomness."""
        path = game.shortest_path(player)
        if path and len(path) > 1:
            move = ("move", path[1])
            if move in game.get_legal_pawn_moves():
                # Usually follow shortest path, sometimes try alternatives
                if random.random() < 0.85:
                    return move

        # Random pawn move as fallback
        pawn_moves = game.get_legal_pawn_moves()
        if pawn_moves:
            return random.choice(pawn_moves)
        return random.choice(game.get_legal_moves())


# =====================================================================
# Data generation
# =====================================================================

def generate_expert_games(num_games: int = 1000,
                          max_moves: int = 200,
                          verbose: bool = True) -> List[dict]:
    """
    Play games between expert agents and record training data.
    Returns list of (state_tensor, policy, value) samples.
    """
    agent = ExpertAgent()
    all_samples = []
    total_moves = 0
    wins = {0: 0, 1: 0, -1: 0}

    t0 = time.time()

    for game_idx in range(num_games):
        game = QuoridorGame(2)
        trajectory = []  # (state, move_made, player)
        move_count = 0

        while not game.is_over and move_count < max_moves:
            state = game.to_tensor()
            cp = game.current_player
            move = agent.choose_move(game)

            trajectory.append((state, move, cp))
            game.make_move(move)
            move_count += 1

        # Determine outcome
        winner = game.winner if game.winner is not None else -1
        wins[winner] = wins.get(winner, 0) + 1
        total_moves += move_count

        # Convert to training samples
        pawn_pos_history = []
        replay = QuoridorGame(2)
        for state, move, cp in trajectory:
            pawn_pos = replay.pawns[cp]

            # Create policy: one-hot on the move that was played
            # (softened slightly to avoid overconfident targets)
            policy = np.zeros(MoveEncoder.TOTAL_ACTIONS, dtype=np.float32)
            idx = MoveEncoder.encode(move, pawn_pos)
            policy[idx] = 0.85

            # Spread remaining 15% across other legal moves
            legal_mask = MoveEncoder.legal_mask(replay)
            legal_mask[idx] = 0  # exclude the played move
            n_other = legal_mask.sum()
            if n_other > 0:
                policy += (0.15 / n_other) * legal_mask

            # Value
            if winner == -1:
                d0 = game.shortest_path_length(0) or 9
                d1 = game.shortest_path_length(1) or 9
                p0_adv = (d1 - d0) / (d0 + d1) if (d0 + d1) > 0 else 0
                value = p0_adv if cp == 0 else -p0_adv
            elif winner == cp:
                value = 1.0
            else:
                value = -1.0

            all_samples.append((state, policy, value))
            replay.make_move(move)

        if verbose and (game_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            avg_len = total_moves / (game_idx + 1)
            print(f"  {game_idx+1}/{num_games} games, "
                  f"avg {avg_len:.0f} moves, "
                  f"{elapsed:.1f}s, "
                  f"P0:{wins[0]} P1:{wins[1]} draw:{wins[-1]}")

    elapsed = time.time() - t0
    print(f"\nGenerated {len(all_samples)} samples from {num_games} games "
          f"in {elapsed:.1f}s")
    print(f"  Avg game length: {total_moves/num_games:.0f} moves")
    print(f"  P0 wins: {wins[0]}, P1 wins: {wins[1]}, draws: {wins[-1]}")

    return all_samples


# =====================================================================
# Pre-training
# =====================================================================

def pretrain(samples: list, config: Config = None, num_epochs: int = 20,
             device: str = "auto"):
    """
    Pre-train the network on expert data.
    """
    if config is None:
        config = Config()
    if device == "auto":
        device = config.resolve_device()

    print(f"\nPre-training on {len(samples)} samples")
    print(f"Device: {device}")

    network = QuoridorNet(
        input_planes=config.input_planes,
        num_blocks=config.num_res_blocks,
        num_channels=config.num_channels,
    ).to(device)
    print(f"Network parameters: {count_parameters(network):,}")

    # Prepare data
    states = np.array([s[0] for s in samples])
    policies = np.array([s[1] for s in samples])
    values = np.array([s[2] for s in samples], dtype=np.float32)

    dataset = TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(policies),
        torch.from_numpy(values).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    optimizer = optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)
    network.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_ploss = 0
        total_vloss = 0
        n = 0

        for batch_s, batch_p, batch_v in loader:
            batch_s = batch_s.to(device)
            batch_p = batch_p.to(device)
            batch_v = batch_v.to(device)

            pred_policy, pred_value = network(batch_s)
            log_policy = torch.log(pred_policy.clamp(min=1e-8))
            policy_loss = -torch.sum(batch_p * log_policy, dim=1).mean()
            value_loss = nn.MSELoss()(pred_value, batch_v)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ploss += policy_loss.item()
            total_vloss += value_loss.item()
            n += 1

        avg_loss = total_loss / n
        avg_ploss = total_ploss / n
        avg_vloss = total_vloss / n
        print(f"  Epoch {epoch+1:>2}/{num_epochs}: "
              f"loss={avg_loss:.4f} "
              f"(policy={avg_ploss:.4f}, value={avg_vloss:.4f})")

    # Save as iteration 0 checkpoint (self-play will continue from here)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, "model_iter_0000.pt")
    torch.save({
        "iteration": 0,
        "model_state_dict": network.state_dict(),
        "config": asdict(config),
    }, path)
    print(f"\nSaved warm-up checkpoint: {path}")
    print("Now run:  python train.py --resume --iterations 30 --games 30 "
          "--simulations 100 --workers 4")

    return network


# =====================================================================
# Diagnostic: test the pre-trained network
# =====================================================================

def quick_test(network, device="cpu"):
    """Quick sanity check of the pre-trained network."""
    network.eval()

    positions = [
        ("Start", QuoridorGame(2)),
    ]

    g2 = QuoridorGame(2)
    g2.pawns = [(1, 4), (7, 4)]
    positions.append(("P0 one step from goal", g2))

    for name, game in positions:
        state = game.to_tensor()
        mask = MoveEncoder.legal_mask(game)
        policy, value = network.predict(state, mask, device)

        pawn_total = 0
        pos = game.pawns[game.current_player]
        for m in game.get_legal_pawn_moves():
            idx = MoveEncoder.encode(m, pos)
            pawn_total += policy[idx]

        print(f"\n  {name}:")
        print(f"    Value: {value:.3f}")
        print(f"    Pawn move %: {pawn_total*100:.1f}%")

        # Top 3
        moves = []
        for m in game.get_legal_pawn_moves() + game.get_legal_walls()[:20]:
            idx = MoveEncoder.encode(m, pos)
            moves.append((policy[idx], m))
        moves.sort(reverse=True)
        for p, m in moves[:3]:
            print(f"    {p*100:5.1f}%  {m}")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train on expert games")
    parser.add_argument("--games", type=int, default=1000,
                        help="Number of expert games to generate")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Pre-training epochs")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate data, don't train")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 1: Generating expert games")
    print("=" * 60)
    samples = generate_expert_games(num_games=args.games)

    if not args.generate_only:
        print("\n" + "=" * 60)
        print("  Phase 2: Pre-training network")
        print("=" * 60)
        cfg = Config()
        device = args.device if args.device != "auto" else cfg.resolve_device()
        network = pretrain(samples, config=cfg, num_epochs=args.epochs,
                          device=device)

        print("\n" + "=" * 60)
        print("  Phase 3: Quick test")
        print("=" * 60)
        quick_test(network, device)

        print("\n" + "=" * 60)
        print("  Done! Next step:")
        print("  caffeinate -i python train.py --resume --iterations 30 "
              "--games 30 --simulations 100 --workers 4")
        print("=" * 60)
