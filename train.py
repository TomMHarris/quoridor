"""
AlphaZero-style training loop for Quoridor.

Each iteration:
    1. Self-play: generate games using current network + MCTS
    2. Train: update network on replay buffer
    3. Checkpoint: save model + stats

Designed for M1 MacBook Air — small network, modest game counts,
with all the knobs in config.py.
"""

import os
import json
import time
import random
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
from mcts_gumbel import GumbelMCTS, select_action as gumbel_select_action
from config import Config


# ── Type alias for training samples ──────────────────────────────────────────
# (state_tensor, policy_target, value_target)
Sample = Tuple[np.ndarray, np.ndarray, float]


# ===========================================================================
# Self-play
# ===========================================================================

def self_play_game(network: QuoridorNet, config: Config,
                   device: str) -> Tuple[List[Sample], int, dict]:
    """
    Play one full game of self-play.

    Returns:
        samples: list of (state, mcts_policy, outcome) — outcome filled in
                 after game ends.
        winner:  player index or -1 for draw/truncation.
        stats:   dict with game metadata.
    """
    game = QuoridorGame(config.num_players)

    if config.use_gumbel:
        mcts = GumbelMCTS(
            network=network,
            device=device,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
        )
    else:
        mcts = MCTS(
            network=network,
            device=device,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
        )

    # Collect (state, policy, current_player) triples
    trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []
    move_count = 0

    while not game.is_over and move_count < config.max_game_length:
        state = game.to_tensor()
        action_probs = mcts.search(game, add_noise=True)

        # Temperature: exploratory for early moves, greedy later
        temp = 1.0 if move_count < config.temperature_moves else 0.1
        action_idx = select_action(action_probs, temperature=temp)

        trajectory.append((state, action_probs, game.current_player))

        # Execute the move
        pawn_pos = game.pawns[game.current_player]
        move = MoveEncoder.decode(action_idx, pawn_pos)
        game.make_move(move)
        move_count += 1

    # Assign outcomes
    winner = game.winner if game.winner is not None else -1
    samples: List[Sample] = []

    if winner == -1:
        # Truncated game: assign values based on who was closer to winning.
        # This teaches "advancing your pawn matters" even without a decisive finish.
        d0 = game.shortest_path_length(0) or 9
        d1 = game.shortest_path_length(1) or 9
        # Player closer to goal was "winning". Scale to [-1, 1].
        # If d0=2, d1=7 → P0 was much closer → P0 gets positive value
        if d0 + d1 > 0:
            p0_advantage = (d1 - d0) / (d0 + d1)  # range [-1, 1]
        else:
            p0_advantage = 0.0
        truncation_values = {0: p0_advantage, 1: -p0_advantage}

    for state, policy, player in trajectory:
        if winner == -1:
            value = truncation_values[player]
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        samples.append((state, policy, value))

    stats = {
        "winner": winner,
        "num_moves": move_count,
        "truncated": winner == -1,
        "final_state": game.to_dict(),
    }

    return samples, winner, stats


def run_self_play(network: QuoridorNet, config: Config,
                  device: str) -> Tuple[List[Sample], List[dict]]:
    """Generate a batch of self-play games (parallel when possible)."""
    network.eval()
    num_workers = min(config.num_parallel_games, config.num_self_play_games)

    # Parallel self-play using multiprocessing
    if num_workers > 1:
        return _run_self_play_parallel(network, config, device, num_workers)

    # Sequential fallback
    return _run_self_play_sequential(network, config, device)


def _run_self_play_sequential(network: QuoridorNet, config: Config,
                               device: str) -> Tuple[List[Sample], List[dict]]:
    all_samples: List[Sample] = []
    all_stats: List[dict] = []

    for i in range(config.num_self_play_games):
        t0 = time.time()
        samples, winner, stats = self_play_game(network, config, device)
        elapsed = time.time() - t0
        stats["time"] = elapsed

        all_samples.extend(samples)
        all_stats.append(stats)

        w_str = f"P{winner}" if winner >= 0 else "draw"
        print(f"  Game {i+1}/{config.num_self_play_games}: "
              f"{stats['num_moves']} moves, {w_str}, {elapsed:.1f}s")

    return all_samples, all_stats


def _play_one_game(args):
    """Worker function for parallel self-play."""
    state_dict, config_dict, game_idx, total = args
    from config import Config
    from network import QuoridorNet

    cfg = Config(**{k: v for k, v in config_dict.items()
                    if k in Config.__dataclass_fields__})

    # Each worker uses CPU to avoid MPS contention
    net = QuoridorNet(
        input_planes=cfg.input_planes,
        num_blocks=cfg.num_res_blocks,
        num_channels=cfg.num_channels,
    )
    net.load_state_dict(state_dict)
    net.eval()

    t0 = time.time()
    samples, winner, stats = self_play_game(net, cfg, "cpu")
    stats["time"] = time.time() - t0

    w_str = f"P{winner}" if winner >= 0 else "draw"
    print(f"  Game {game_idx+1}/{total}: "
          f"{stats['num_moves']} moves, {w_str}, {stats['time']:.1f}s")

    return samples, stats


def _run_self_play_parallel(network: QuoridorNet, config: Config,
                             device: str, num_workers: int
                             ) -> Tuple[List[Sample], List[dict]]:
    import multiprocessing as mp

    # Prepare args — send model weights (CPU) to each worker
    state_dict = {k: v.cpu() for k, v in network.state_dict().items()}
    config_dict = {k: v for k, v in asdict(config).items()}
    n = config.num_self_play_games

    args_list = [(state_dict, config_dict, i, n) for i in range(n)]

    print(f"  (parallel: {num_workers} workers)")
    with mp.Pool(num_workers) as pool:
        results = pool.map(_play_one_game, args_list)

    all_samples: List[Sample] = []
    all_stats: List[dict] = []
    for samples, stats in results:
        all_samples.extend(samples)
        all_stats.append(stats)

    return all_samples, all_stats


# ===========================================================================
# Training
# ===========================================================================

def train_network(network: QuoridorNet, replay_buffer: deque,
                  config: Config, device: str) -> dict:
    """
    Train the network on samples from the replay buffer.
    Returns dict with loss statistics.
    """
    network.train()

    # Unpack buffer
    states = np.array([s[0] for s in replay_buffer])
    policies = np.array([s[1] for s in replay_buffer])
    values = np.array([s[2] for s in replay_buffer], dtype=np.float32)

    dataset = TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(policies),
        torch.from_numpy(values).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        drop_last=False)

    optimizer = optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    n_batches = 0

    for epoch in range(config.num_epochs):
        for batch_states, batch_policies, batch_values in loader:
            batch_states = batch_states.to(device)
            batch_policies = batch_policies.to(device)
            batch_values = batch_values.to(device)

            # Forward (no legal mask during training — we train against
            # the MCTS policy which is already masked)
            pred_policy, pred_value = network(batch_states)

            # Policy loss: cross-entropy with MCTS target distribution
            # Avoid log(0) by clamping
            log_policy = torch.log(pred_policy.clamp(min=1e-8))
            policy_loss = -torch.sum(batch_policies * log_policy, dim=1).mean()

            # Value loss: MSE
            value_loss = nn.MSELoss()(pred_value, batch_values)

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "policy_loss": total_policy_loss / max(n_batches, 1),
        "value_loss": total_value_loss / max(n_batches, 1),
        "n_batches": n_batches,
        "buffer_size": len(replay_buffer),
    }


# ===========================================================================
# Main training loop
# ===========================================================================

def train(config: Config = None, resume: bool = False):
    if config is None:
        config = Config()

    device = config.resolve_device()
    print(f"Device: {device}")
    print(f"Config: {json.dumps(asdict(config), indent=2)}")

    # Initialise network
    network = QuoridorNet(
        input_planes=config.input_planes,
        num_blocks=config.num_res_blocks,
        num_channels=config.num_channels,
    ).to(device)
    print(f"Network parameters: {count_parameters(network):,}")

    # Create dirs
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Resume from checkpoint if requested
    start_iteration = 1
    log: List[dict] = []

    if resume:
        import glob
        checkpoints = sorted(glob.glob(
            os.path.join(config.checkpoint_dir, "model_iter_*.pt")))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"\n  Resuming from {latest}")
            data = torch.load(latest, map_location=device, weights_only=False)
            network.load_state_dict(data["model_state_dict"])
            start_iteration = data["iteration"] + 1
            print(f"  Continuing from iteration {start_iteration}")

            # Load existing log
            log_path = os.path.join(config.log_dir, "training_log.json")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    log = json.load(f)
        else:
            print("  No checkpoint found, starting fresh")

    # Replay buffer
    replay_buffer: deque = deque(maxlen=config.replay_buffer_size)

    end_iteration = start_iteration + config.num_iterations - 1

    for iteration in range(start_iteration, end_iteration + 1):
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration} (runs {start_iteration}-{end_iteration})")
        print(f"{'='*60}")

        # ── Self-play ──
        print("\n▸ Self-play...")
        t0 = time.time()
        samples, game_stats = run_self_play(network, config, device)
        self_play_time = time.time() - t0

        replay_buffer.extend(samples)
        print(f"  {len(samples)} new samples, "
              f"buffer: {len(replay_buffer)}/{config.replay_buffer_size}")

        # ── Self-play summary ──
        wins = [s["winner"] for s in game_stats]
        truncated = sum(1 for s in game_stats if s["truncated"])
        avg_moves = np.mean([s["num_moves"] for s in game_stats])
        avg_time = np.mean([s["time"] for s in game_stats])

        print(f"  Avg game: {avg_moves:.0f} moves, {avg_time:.1f}s")
        print(f"  Wins: P0={wins.count(0)}, P1={wins.count(1)}, "
              f"draws={truncated}")

        # ── Train ──
        if len(replay_buffer) >= config.min_replay_size:
            print("\n▸ Training...")
            t0 = time.time()
            train_stats = train_network(network, replay_buffer, config, device)
            train_time = time.time() - t0
            print(f"  Loss: {train_stats['loss']:.4f} "
                  f"(policy: {train_stats['policy_loss']:.4f}, "
                  f"value: {train_stats['value_loss']:.4f})")
            print(f"  {train_stats['n_batches']} batches in {train_time:.1f}s")
        else:
            print(f"\n▸ Skipping training (need {config.min_replay_size} "
                  f"samples, have {len(replay_buffer)})")
            train_stats = {}
            train_time = 0

        # ── Log ──
        entry = {
            "iteration": iteration,
            "self_play_time": self_play_time,
            "train_time": train_time,
            "num_samples": len(samples),
            "buffer_size": len(replay_buffer),
            "avg_game_length": float(avg_moves),
            "p0_wins": wins.count(0),
            "p1_wins": wins.count(1),
            "truncated": truncated,
            **train_stats,
        }
        log.append(entry)

        # Save log
        with open(os.path.join(config.log_dir, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        # Save game histories for visualization
        games_path = os.path.join(config.log_dir, f"games_iter_{iteration:04d}.json")
        with open(games_path, "w") as f:
            json.dump(game_stats, f)

        # ── Checkpoint ──
        if iteration % config.save_every == 0 or iteration == config.num_iterations:
            path = os.path.join(
                config.checkpoint_dir, f"model_iter_{iteration:04d}.pt"
            )
            torch.save({
                "iteration": iteration,
                "model_state_dict": network.state_dict(),
                "config": asdict(config),
            }, path)
            print(f"  Saved checkpoint: {path}")

    print(f"\n{'='*60}")
    print("  Training complete!")
    print(f"{'='*60}")
    return network, log


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # required on macOS

    parser = argparse.ArgumentParser(description="Train Quoridor AI")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--games", type=int, default=None,
                        help="Self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=None,
                        help="MCTS simulations per move")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel self-play workers (default: 4)")
    parser.add_argument("--gumbel", action="store_true",
                        help="Use Gumbel MCTS (better with fewer simulations)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--players", type=int, default=2, choices=[2, 4])
    args = parser.parse_args()

    cfg = Config(
        num_players=args.players,
        device=args.device,
    )
    if args.iterations is not None:
        cfg.num_iterations = args.iterations
    if args.games is not None:
        cfg.num_self_play_games = args.games
    if args.simulations is not None:
        cfg.num_simulations = args.simulations
    if args.workers is not None:
        cfg.num_parallel_games = args.workers
    if args.gumbel:
        cfg.use_gumbel = True

    train(cfg, resume=args.resume)
