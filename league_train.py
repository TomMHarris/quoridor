"""
League Training for Quoridor
==============================

Instead of pure self-play (which causes strategy collapse), the model
trains against a diverse pool of opponents:

  1. Current model (self-play) — discovers new strategies
  2. Past checkpoints — prevents forgetting how to beat older versions
  3. Warmup/expert model — maintains baseline competence
  4. Heuristic agents — ensures it can beat known strategies

This is inspired by AlphaStar's league training, adapted for a board game.

Usage:
    python league_train.py --iterations 30 --games 40 --workers 4
    python league_train.py --resume --iterations 20 --games 40
"""

import argparse
import glob
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
from mcts import MCTS, select_action
from config import Config


Sample = Tuple[np.ndarray, np.ndarray, float]


# =====================================================================
# Opponent pool
# =====================================================================

class HeuristicOpponent:
    """Greedy shortest-path agent with optional wall play."""

    def __init__(self, wall_prob=0.0, name="Greedy"):
        self.wall_prob = wall_prob
        self.name = name

    def choose_move(self, game):
        player = game.current_player
        opponent = 1 - player

        # Win immediately if possible
        if game.shortest_path_length(player) == 1:
            path = game.shortest_path(player)
            if path and len(path) > 1:
                move = ("move", path[1])
                if move in game.get_legal_pawn_moves():
                    return move

        # Occasionally place a wall
        if (self.wall_prob > 0 and random.random() < self.wall_prob
                and game.walls_remaining[player] > 0):
            wall = self._find_wall(game, player, opponent)
            if wall:
                return wall

        # Follow shortest path
        path = game.shortest_path(player)
        if path and len(path) > 1:
            move = ("move", path[1])
            if move in game.get_legal_pawn_moves():
                return move

        pawn_moves = game.get_legal_pawn_moves()
        if pawn_moves:
            return random.choice(pawn_moves)
        return random.choice(game.get_legal_moves())

    def _find_wall(self, game, player, opponent):
        opp_dist = game.shortest_path_length(opponent)
        opp_path = game.shortest_path(opponent)
        if not opp_path or opp_dist is None:
            return None

        best_wall = None
        best_net = 0
        for cell in opp_path[1:4]:
            tr, tc = cell
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    wr, wc = tr + dr, tc + dc
                    for o in ("H", "V"):
                        if game.is_valid_wall(wr, wc, o):
                            g2 = game.clone()
                            g2.make_move(("wall", (wr, wc, o)))
                            new_opp = g2.shortest_path_length(opponent)
                            new_me = g2.shortest_path_length(player)
                            if new_opp and new_me:
                                my_old = game.shortest_path_length(player)
                                net = (new_opp - opp_dist) - (new_me - (my_old or 8))
                                if net > best_net:
                                    best_net = net
                                    best_wall = ("wall", (wr, wc, o))
        return best_wall


class NetworkOpponent:
    """An older version of the network as opponent."""

    def __init__(self, network, device, simulations=50, name="Past model"):
        self.network = network
        self.device = device
        self.simulations = simulations
        self.name = name
        self.mcts = MCTS(
            network=network, device=device,
            num_simulations=simulations, c_puct=1.5,
        )

    def choose_move(self, game):
        probs = self.mcts.search(game, add_noise=False)
        idx = select_action(probs, temperature=0.1)
        pos = game.pawns[game.current_player]
        return MoveEncoder.decode(idx, pos)


class OpponentPool:
    """
    Manages the league of opponents.

    Distribution (tuned for post-warmup training where warmup = best model):
      30% self-play (current network vs itself — exploration)
      40% warmup/expert model (strongest anchor — prevents forgetting)
      20% past checkpoints (diversity)
      10% heuristic agents (ground truth signal)
    """

    def __init__(self, device):
        self.device = device
        self.checkpoints = []  # list of (path, iteration)
        self.heuristics = [
            HeuristicOpponent(wall_prob=0.0, name="Greedy"),
            HeuristicOpponent(wall_prob=0.3, name="SmartGreedy"),
            HeuristicOpponent(wall_prob=0.5, name="WallHeavy"),
        ]
        self.warmup_opponent = None

    def load_warmup(self, config):
        """Load the warmup model as a permanent league member."""
        warmup_path = os.path.join(config.checkpoint_dir, "model_iter_0000.pt")
        if os.path.exists(warmup_path):
            net = self._load_network(warmup_path, config)
            self.warmup_opponent = NetworkOpponent(
                net, self.device, simulations=50, name="Warmup"
            )
            print(f"  Loaded warmup opponent: {warmup_path}")

    def update_checkpoints(self, config):
        """Refresh the list of available past checkpoints."""
        files = sorted(glob.glob(
            os.path.join(config.checkpoint_dir, "model_iter_*.pt")))
        self.checkpoints = []
        for f in files:
            try:
                it = int(os.path.basename(f).split("_")[-1].split(".")[0])
                self.checkpoints.append((f, it))
            except ValueError:
                pass

    def sample_opponent(self, current_network, config):
        """
        Sample an opponent from the league.
        Returns (opponent, name) where opponent has a choose_move method.

        Distribution biased toward warmup model because it's the strongest
        known player — games against it give the clearest win/loss signal.
        """
        roll = random.random()

        if roll < 0.30:
            # Self-play (exploration)
            opp = NetworkOpponent(
                current_network, self.device,
                simulations=config.num_simulations, name="Self"
            )
            return opp, "self"

        elif roll < 0.70:
            # Warmup model — strongest anchor, 40% of games
            if self.warmup_opponent:
                return self.warmup_opponent, "warmup"
            # Fallback to self
            opp = NetworkOpponent(
                current_network, self.device,
                simulations=config.num_simulations, name="Self"
            )
            return opp, "self"

        elif roll < 0.90:
            # Past checkpoint (diversity)
            if self.checkpoints:
                path, it = random.choice(self.checkpoints)
                try:
                    net = self._load_network(path, config)
                    opp = NetworkOpponent(
                        net, self.device,
                        simulations=config.num_simulations,
                        name=f"Past-{it}"
                    )
                    return opp, f"past-{it}"
                except Exception:
                    pass
            # Fallback to warmup
            if self.warmup_opponent:
                return self.warmup_opponent, "warmup"
            opp = NetworkOpponent(
                current_network, self.device,
                simulations=config.num_simulations, name="Self"
            )
            return opp, "self"

        else:
            # Heuristic (ground truth signal)
            h = random.choice(self.heuristics)
            return h, h.name

    def _load_network(self, path, config):
        data = torch.load(path, map_location=self.device, weights_only=False)
        net = QuoridorNet(
            input_planes=config.input_planes,
            num_blocks=config.num_res_blocks,
            num_channels=config.num_channels,
        ).to(self.device)
        net.load_state_dict(data["model_state_dict"])
        net.eval()
        return net


# =====================================================================
# League self-play game
# =====================================================================

def play_league_game(network, opponent, config, device,
                     max_moves=200) -> Tuple[List[Sample], int, dict]:
    """
    Play a game: current network as one player, opponent as the other.
    The network always trains from its own perspective.
    Randomly assign network to P0 or P1 for fairness.
    """
    game = QuoridorGame(2)
    mcts = MCTS(
        network=network, device=device,
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        pawn_boost=False,  # warmup model already has good pawn priors
    )

    # Randomly assign sides
    network_player = random.choice([0, 1])

    trajectory = []  # only network's positions
    move_count = 0

    while not game.is_over and move_count < max_moves:
        state = game.to_tensor()
        cp = game.current_player

        if cp == network_player:
            # Network's turn — use MCTS
            legal_mask = MoveEncoder.legal_mask(game)
            raw_policy, _ = network.predict(state, legal_mask, device)
            action_probs = mcts.search(game, add_noise=True)
            temp = 1.0 if move_count < config.temperature_moves else 0.1
            action_idx = select_action(action_probs, temperature=temp)
            pawn_pos = game.pawns[cp]
            move = MoveEncoder.decode(action_idx, pawn_pos)

            # Only add to trajectory if MCTS meaningfully improved on raw policy.
            # KL(mcts || raw) > 0.01 means search found something the network missed.
            # This prevents circular training where MCTS ≈ network prior.
            kl = float(np.sum(
                action_probs * np.log((action_probs + 1e-8) / (raw_policy + 1e-8))
            ))
            if kl > 0.01:
                trajectory.append((state, action_probs, cp))
        else:
            # Opponent's turn
            move = opponent.choose_move(game)

        game.make_move(move)
        move_count += 1

    # Assign values from network's perspective
    winner = game.winner if game.winner is not None else -1
    samples = []

    for state, policy, player in trajectory:
        if winner == -1:
            d0 = game.shortest_path_length(0) or 9
            d1 = game.shortest_path_length(1) or 9
            p0_adv = (d1 - d0) / (d0 + d1) if (d0 + d1) > 0 else 0
            value = p0_adv if player == 0 else -p0_adv
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        samples.append((state, policy, value))

    result = "win" if winner == network_player else (
        "loss" if winner >= 0 else "draw")

    stats = {
        "winner": winner,
        "num_moves": move_count,
        "network_player": network_player,
        "result": result,
        "truncated": winner == -1,
        "final_state": game.to_dict(),
    }

    return samples, winner, stats


# =====================================================================
# League training loop
# =====================================================================

def train_network(network, replay_buffer, config, device, optimizer=None):
    """Train on replay buffer. Accepts persistent optimizer to retain Adam state."""
    network.train()
    states = np.array([s[0] for s in replay_buffer])
    policies = np.array([s[1] for s in replay_buffer])
    values = np.array([s[2] for s in replay_buffer], dtype=np.float32)

    dataset = TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(policies),
        torch.from_numpy(values).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, drop_last=False)

    if optimizer is None:
        optimizer = optim.Adam(network.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)

    # Fewer epochs to prevent overfitting on small, noisy buffer
    num_epochs = min(config.num_epochs, 2)

    total_loss = total_ploss = total_vloss = 0
    n = 0

    for epoch in range(num_epochs):
        for batch_s, batch_p, batch_v in loader:
            batch_s = batch_s.to(device)
            batch_p = batch_p.to(device)
            batch_v = batch_v.to(device)

            pred_p, pred_v = network(batch_s)
            log_p = torch.log(pred_p.clamp(min=1e-8))
            p_loss = -torch.sum(batch_p * log_p, dim=1).mean()
            v_loss = nn.MSELoss()(pred_v, batch_v)
            loss = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ploss += p_loss.item()
            total_vloss += v_loss.item()
            n += 1

    return {
        "loss": total_loss / max(n, 1),
        "policy_loss": total_ploss / max(n, 1),
        "value_loss": total_vloss / max(n, 1),
        "n_batches": n,
    }


def league_train(config=None, resume=False):
    if config is None:
        config = Config()

    device = config.resolve_device()
    print(f"Device: {device}")

    network = QuoridorNet(
        input_planes=config.input_planes,
        num_blocks=config.num_res_blocks,
        num_channels=config.num_channels,
    ).to(device)
    print(f"Network parameters: {count_parameters(network):,}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Set up opponent pool
    pool = OpponentPool(device)

    start_iteration = 1
    log = []

    _resume_opt_state = None
    if resume:
        checkpoints = sorted(glob.glob(
            os.path.join(config.checkpoint_dir, "model_iter_*.pt")))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"Resuming from {latest}")
            data = torch.load(latest, map_location=device, weights_only=False)
            network.load_state_dict(data["model_state_dict"])
            start_iteration = data["iteration"] + 1
            _resume_opt_state = data.get("optimizer_state_dict")

            log_path = os.path.join(config.log_dir, "training_log.json")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    log = json.load(f)

    pool.load_warmup(config)
    pool.update_checkpoints(config)

    # Small buffer: ~3 iterations of data to prevent stale-data poisoning.
    # With ~40 games × ~40 moves = ~1600 samples/iter, keep ~5000.
    buffer_size = max(3000, config.num_self_play_games * 50 * 3)
    replay_buffer = deque(maxlen=buffer_size)

    # Persistent optimizer so Adam retains momentum across iterations.
    # Lower LR (3e-4) to prevent catastrophic forgetting of warmup knowledge.
    optimizer = optim.Adam(network.parameters(), lr=3e-4,
                           weight_decay=config.weight_decay)
    if _resume_opt_state is not None:
        optimizer.load_state_dict(_resume_opt_state)

    end_iteration = start_iteration + config.num_iterations - 1

    for iteration in range(start_iteration, end_iteration + 1):
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration} (league training)")
        print(f"{'='*60}")

        # ── League self-play ──
        print("\n> Self-play vs league...")
        network.eval()
        all_samples = []
        all_stats = []
        opp_counts = {}

        t0 = time.time()
        for i in range(config.num_self_play_games):
            opponent, opp_name = pool.sample_opponent(network, config)
            opp_counts[opp_name] = opp_counts.get(opp_name, 0) + 1

            samples, winner, stats = play_league_game(
                network, opponent, config, device)
            stats["opponent"] = opp_name
            all_samples.extend(samples)
            all_stats.append(stats)

            r = stats["result"]
            print(f"  Game {i+1}/{config.num_self_play_games} "
                  f"vs {opp_name}: {stats['num_moves']} moves, {r}")

        self_play_time = time.time() - t0
        replay_buffer.extend(all_samples)

        # Stats summary
        results = [s["result"] for s in all_stats]
        wins = results.count("win")
        losses = results.count("loss")
        draws = results.count("draw")
        avg_moves = np.mean([s["num_moves"] for s in all_stats])

        print(f"\n  {len(all_samples)} samples, "
              f"buffer: {len(replay_buffer)}/{config.replay_buffer_size}")
        print(f"  Wins: {wins}, Losses: {losses}, Draws: {draws}")
        print(f"  Avg game: {avg_moves:.0f} moves, {self_play_time:.0f}s total")
        print(f"  Opponents: {dict(opp_counts)}")

        # ── Train ──
        if len(replay_buffer) >= config.min_replay_size:
            print("\n> Training...")
            t0 = time.time()
            train_stats = train_network(network, replay_buffer, config, device,
                                        optimizer=optimizer)
            train_time = time.time() - t0
            print(f"  Loss: {train_stats['loss']:.4f} "
                  f"(policy: {train_stats['policy_loss']:.4f}, "
                  f"value: {train_stats['value_loss']:.4f})")
        else:
            train_stats = {}
            train_time = 0

        # ── Log ──
        entry = {
            "iteration": iteration,
            "self_play_time": self_play_time,
            "train_time": train_time,
            "num_samples": len(all_samples),
            "buffer_size": len(replay_buffer),
            "avg_game_length": float(avg_moves),
            "wins": wins, "losses": losses, "draws": draws,
            "opponents": dict(opp_counts),
            **train_stats,
        }
        log.append(entry)

        with open(os.path.join(config.log_dir, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        # Save game histories for visualization
        games_path = os.path.join(config.log_dir, f"games_iter_{iteration:04d}.json")
        with open(games_path, "w") as f:
            json.dump(all_stats, f)

        # ── Checkpoint ──
        if iteration % config.save_every == 0 or iteration == end_iteration:
            path = os.path.join(
                config.checkpoint_dir, f"model_iter_{iteration:04d}.pt")
            torch.save({
                "iteration": iteration,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
            }, path)
            print(f"  Saved: {path}")

            # Update opponent pool with new checkpoint
            pool.update_checkpoints(config)

    print(f"\n{'='*60}")
    print("  League training complete!")
    print(f"{'='*60}")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="League training for Quoridor")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--games", type=int, default=40,
                        help="Games per iteration")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = Config()
    cfg.num_iterations = args.iterations
    cfg.num_self_play_games = args.games
    cfg.num_simulations = args.simulations
    cfg.device = args.device
    cfg.save_every = 5

    league_train(cfg, resume=args.resume)
