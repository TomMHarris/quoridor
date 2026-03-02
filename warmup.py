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

class MCTSExpert:
    """
    Strong expert using pure MCTS with heuristic rollouts.
    This is essentially gorisanson's approach: no neural network,
    just search + smart rollout policy. With 500-2000 rollouts
    per move, this plays genuinely strong Quoridor.

    The rollout policy follows shortest path 70% of the time
    (exactly gorisanson's design) and places random probable
    walls otherwise.
    """

    def __init__(self, num_rollouts: int = 500):
        self.num_rollouts = num_rollouts

    def choose_move(self, game: QuoridorGame) -> tuple:
        player = game.current_player

        # If we can win immediately, do it
        if game.shortest_path_length(player) == 1:
            path = game.shortest_path(player)
            if path and len(path) > 1:
                move = ("move", path[1])
                if move in game.get_legal_pawn_moves():
                    return move

        # Get candidate moves: pawn moves + probable walls
        pawn_moves = game.get_legal_pawn_moves()
        try:
            walls = game.get_probable_walls()
        except AttributeError:
            walls = game.get_legal_walls()

        candidates = pawn_moves + walls
        if not candidates:
            return random.choice(game.get_legal_moves())

        # Allocate rollouts across candidates
        rollouts_per = max(1, self.num_rollouts // len(candidates))
        best_move = None
        best_score = -1

        for move in candidates:
            wins = 0
            for _ in range(rollouts_per):
                g = game.clone()
                g.make_move(move)
                result = self._heuristic_rollout(g, player)
                wins += result
            if wins > best_score:
                best_score = wins
                best_move = move

        return best_move

    def _heuristic_rollout(self, game: QuoridorGame, perspective: int,
                           max_depth: int = 80) -> float:
        """
        Heuristic rollout following gorisanson's design:
        - 70% of the time: move pawn along shortest path
        - 30% of the time: place a random probable wall (if available)
        - If no walls left: move backward (penalty for wall exhaustion)
        """
        depth = 0
        while not game.is_over and depth < max_depth:
            cp = game.current_player

            if random.random() < 0.7:
                # Follow shortest path
                path = game.shortest_path(cp)
                if path and len(path) > 1:
                    move = ("move", path[1])
                    legal_pawns = game.get_legal_pawn_moves()
                    if move in legal_pawns:
                        game.make_move(move)
                        depth += 1
                        continue
                    # Path blocked by opponent pawn — try any pawn move
                    if legal_pawns:
                        game.make_move(random.choice(legal_pawns))
                        depth += 1
                        continue

                # Fallback: random pawn move
                pawn_moves = game.get_legal_pawn_moves()
                if pawn_moves:
                    game.make_move(random.choice(pawn_moves))
                    depth += 1
                    continue

            # Wall placement (30% of the time)
            if game.walls_remaining[cp] > 0:
                try:
                    walls = game.get_probable_walls()
                except AttributeError:
                    walls = game.get_legal_walls()
                if walls:
                    game.make_move(random.choice(walls))
                    depth += 1
                    continue

            # No walls left — just move pawn
            pawn_moves = game.get_legal_pawn_moves()
            if pawn_moves:
                game.make_move(random.choice(pawn_moves))
            else:
                moves = game.get_legal_moves()
                if moves:
                    game.make_move(random.choice(moves))
                else:
                    break
            depth += 1

        if game.winner == perspective:
            return 1.0
        elif game.winner is not None:
            return 0.0
        # Unfinished: evaluate by distance
        d_me = game.shortest_path_length(perspective) or 9
        d_opp = game.shortest_path_length(1 - perspective) or 9
        if d_me < d_opp:
            return 0.7
        elif d_me > d_opp:
            return 0.3
        return 0.5


class ExpertAgent:
    """
    Simpler heuristic agent for diversity in training data.
    Follows shortest path, occasionally blocks opponent.
    Weaker than MCTSExpert but faster and provides variety.
    """

    def choose_move(self, game: QuoridorGame) -> tuple:
        player = game.current_player
        opponent = (player + 1) % game.num_players  # works for 2 and 4 player

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
        should_wall = False
        if my_walls > 0:
            if opp_dist is not None and my_dist is not None:
                if opp_dist <= my_dist:
                    should_wall = random.random() < 0.5
                elif opp_dist <= my_dist + 2:
                    should_wall = random.random() < 0.25
                else:
                    should_wall = random.random() < 0.1
                if opp_dist <= 2:
                    should_wall = random.random() < 0.7

        if should_wall:
            wall = self._find_best_wall(game, player, opponent)
            if wall:
                return wall

        return self._advance(game, player)

    def _find_best_wall(self, game, player, opponent):
        opp_dist_before = game.shortest_path_length(opponent)
        if opp_dist_before is None:
            return None
        opp_path = game.shortest_path(opponent)
        if not opp_path or len(opp_path) < 2:
            return None

        best_wall = None
        best_increase = 0

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
                                my_new = g2.shortest_path_length(player)
                                my_old = game.shortest_path_length(player)
                                if my_new is not None and my_old is not None:
                                    net = increase - (my_new - my_old)
                                    if net > best_increase:
                                        best_increase = net
                                        best_wall = ("wall", (wr, wc, o))

        if best_wall and best_increase >= 1:
            return best_wall
        return None

    def _advance(self, game, player):
        path = game.shortest_path(player)
        if path and len(path) > 1:
            move = ("move", path[1])
            if move in game.get_legal_pawn_moves():
                if random.random() < 0.85:
                    return move
        pawn_moves = game.get_legal_pawn_moves()
        if pawn_moves:
            return random.choice(pawn_moves)
        return random.choice(game.get_legal_moves())


# =====================================================================
# Data generation
# =====================================================================

def _play_one_expert_game(args):
    """Worker function for parallel expert game generation."""
    game_idx, num_games, mcts_rollouts, use_mcts, num_players = args

    if use_mcts:
        agent = MCTSExpert(num_rollouts=mcts_rollouts)
    else:
        agent = ExpertAgent()

    game = QuoridorGame(num_players)
    trajectory = []
    move_count = 0

    while not game.is_over and move_count < 200:
        state = game.to_tensor()
        cp = game.current_player
        move = agent.choose_move(game)
        trajectory.append((state, move, cp))
        game.make_move(move)
        move_count += 1

    winner = game.winner if game.winner is not None else -1

    # Convert to training samples
    samples = []
    replay = QuoridorGame(2)
    for state, move, cp in trajectory:
        pawn_pos = replay.pawns[cp]
        policy = np.zeros(MoveEncoder.TOTAL_ACTIONS, dtype=np.float32)
        idx = MoveEncoder.encode(move, pawn_pos)
        policy[idx] = 0.85
        legal_mask = MoveEncoder.legal_mask(replay)
        legal_mask[idx] = 0
        n_other = legal_mask.sum()
        if n_other > 0:
            policy += (0.15 / n_other) * legal_mask

        if winner == -1:
            d0 = game.shortest_path_length(0) or 9
            d1 = game.shortest_path_length(1) or 9
            p0_adv = (d1 - d0) / (d0 + d1) if (d0 + d1) > 0 else 0
            value = p0_adv if cp == 0 else -p0_adv
        elif winner == cp:
            value = 1.0
        else:
            value = -1.0

        samples.append((state, policy, value))
        replay.make_move(move)

    tag = "MCTS" if use_mcts else "heur"
    w = f"P{winner}" if winner >= 0 else "draw"
    print(f"  Game {game_idx+1}/{num_games} [{tag}]: {move_count} moves, {w}")

    return samples, winner


def generate_expert_games(num_games: int = 1000,
                          max_moves: int = 200,
                          mcts_rollouts: int = 500,
                          mcts_fraction: float = 0.7,
                          num_workers: int = 4,
                          num_players: int = 2,
                          verbose: bool = True) -> list:
    """
    Play games between expert agents in parallel.

    70% MCTS games (strong), 30% heuristic games (fast, diverse).
    """
    import multiprocessing as mp

    # Build argument list
    args_list = []
    for i in range(num_games):
        use_mcts = random.random() < mcts_fraction
        args_list.append((i, num_games, mcts_rollouts, use_mcts, num_players))

    n_mcts = sum(1 for a in args_list if a[3])
    n_heur = num_games - n_mcts
    print(f"  Generating {num_games} games ({n_mcts} MCTS @ {mcts_rollouts} rollouts, "
          f"{n_heur} heuristic) with {num_workers} workers")

    t0 = time.time()

    with mp.Pool(num_workers) as pool:
        results = pool.map(_play_one_expert_game, args_list)

    all_samples = []
    wins = {0: 0, 1: 0, -1: 0}
    for samples, winner in results:
        all_samples.extend(samples)
        wins[winner] = wins.get(winner, 0) + 1

    elapsed = time.time() - t0
    print(f"\nGenerated {len(all_samples)} samples from {num_games} games "
          f"in {elapsed:.1f}s ({elapsed/num_games:.1f}s/game)")
    print(f"  P0 wins: {wins[0]}, P1 wins: {wins[1]}, draws: {wins.get(-1, 0)}")

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
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Pre-train on expert games")
    parser.add_argument("--games", type=int, default=1000,
                        help="Number of expert games to generate")
    parser.add_argument("--rollouts", type=int, default=500,
                        help="MCTS rollouts per move for expert (more = stronger, slower)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for game generation")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: heuristic only (no MCTS), 100%% speed")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Pre-training epochs")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate data, don't train")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--players", type=int, default=2, choices=[2, 4],
                        help="Number of players (2 or 4)")
    args = parser.parse_args()

    # Fast mode: all heuristic, no slow MCTS rollouts
    mcts_frac = 0.0 if args.fast else 0.7

    print("=" * 60)
    print(f"  Phase 1: Generating expert games ({args.players}-player)")
    print("=" * 60)
    samples = generate_expert_games(
        num_games=args.games,
        mcts_rollouts=args.rollouts,
        mcts_fraction=mcts_frac,
        num_workers=args.workers,
        num_players=args.players,
    )

    if not args.generate_only:
        print("\n" + "=" * 60)
        print("  Phase 2: Pre-training network")
        print("=" * 60)
        cfg = Config(num_players=args.players)
        device = args.device if args.device != "auto" else cfg.resolve_device()
        network = pretrain(samples, config=cfg, num_epochs=args.epochs,
                          device=device)

        print("\n" + "=" * 60)
        print("  Phase 3: Quick test")
        print("=" * 60)
        quick_test(network, device)

        print("\n" + "=" * 60)
        print("  Done! Next step:")
        print("  caffeinate -i python train_final.py --iterations 30 "
              f"--games 25 --simulations 150 --players {args.players}")
        print("=" * 60)
