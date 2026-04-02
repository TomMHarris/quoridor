"""
Complete training pipeline for Quoridor AI.

Phase 1 (warmup):  Pre-train on expert games via supervised learning.
Phase 2 (self-play): Improve via MCTS-guided self-play with evaluation gates.

Usage:
    python train.py                          # Full pipeline
    python train.py --skip-warmup            # Skip to self-play
    python train.py --warmup-only            # Just expert pre-training
    python train.py --warmup-only --games 20 # Quick test with few games
    python train.py --iterations 5           # Override iteration count
"""

import argparse
import os
import time
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from engine import QuoridorGame, MoveEncoder
from network import QuoridorNet
from mcts import GumbelMCTS, select_action, _fast_clone
from expert import (PathFollower, HeuristicExpert, MCTSExpert,
                    generate_expert_games)
from config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_net(config: Config) -> QuoridorNet:
    return QuoridorNet(config.input_planes, config.num_res_blocks,
                       config.num_channels, config.action_size)


def _save_checkpoint(net: QuoridorNet, optimizer, config: Config,
                     name: str, extra: dict = None) -> str:
    path = os.path.join(config.checkpoint_dir, f"{name}.pt")
    data = {
        "model_state": net.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "config": {
            "num_res_blocks": config.num_res_blocks,
            "num_channels": config.num_channels,
            "input_planes": config.input_planes,
            "action_size": config.action_size,
        },
    }
    if extra:
        data.update(extra)
    torch.save(data, path)
    return path


def _load_checkpoint(path: str, config: Config, device: torch.device):
    data = torch.load(path, map_location=device, weights_only=False)
    net = _make_net(config)
    net.load_state_dict(data["model_state"])
    net.to(device)
    return net, data


# ---------------------------------------------------------------------------
# Quick evaluation (used for sanity checks)
# ---------------------------------------------------------------------------

def quick_eval(net: QuoridorNet, device: torch.device, config: Config,
               opponent, num_games: int = 20, net_sims: int = 32) -> float:
    """Play net (with MCTS) vs opponent. Returns win rate for net."""
    net.eval()
    mcts = GumbelMCTS(net, device, config)
    wins = 0

    for g_idx in range(num_games):
        game = QuoridorGame(2)
        net_player = g_idx % 2  # alternate colors

        for step in range(config.max_game_length):
            if game.is_over:
                break

            if game.current_player == net_player:
                probs = mcts.search(game, num_simulations=net_sims,
                                    add_noise=False)
                action = select_action(probs, temperature=0.1)
                pos = game.pawns[game.current_player]
                move = MoveEncoder.decode(action, pos)
                # Validate move is legal
                if MoveEncoder.legal_mask(game)[action] < 0.5:
                    legal = game.get_legal_moves()
                    move = random.choice(legal)
            else:
                move = opponent.choose_move(game)

            game.make_move(move)

        if game.winner == net_player:
            wins += 1

    return wins / num_games


# ---------------------------------------------------------------------------
# Phase 1: Expert pre-training (warmup)
# ---------------------------------------------------------------------------

def warmup(config: Config, device: torch.device) -> QuoridorNet:
    """Pre-train network on expert games."""
    print("=" * 60)
    print("PHASE 1: Expert Pre-training")
    print("=" * 60)

    # Generate expert data
    print(f"\nGenerating {config.warmup_games} expert games...")
    t0 = time.time()
    samples = generate_expert_games(config, verbose=True)
    print(f"  Time: {time.time() - t0:.1f}s")

    # Build tensors
    states = torch.tensor(np.array([s[0] for s in samples]))
    policies = torch.tensor(np.array([s[1] for s in samples]))
    values = torch.tensor(np.array([s[2] for s in samples], dtype=np.float32))
    masks_all = torch.tensor(np.array([s[3] for s in samples]))

    dataset = TensorDataset(states, policies, values, masks_all)
    bs = min(config.warmup_batch_size, len(samples))
    loader = DataLoader(dataset, batch_size=bs,
                        shuffle=True, drop_last=False)

    # Create network and optimizer
    net = _make_net(config).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.warmup_lr,
                                 weight_decay=config.train_weight_decay)

    n_params = sum(p.numel() for p in net.parameters())
    print(f"\nNetwork: {n_params:,} parameters")
    print(f"Training: {len(samples)} samples, {config.warmup_epochs} epochs")
    print(f"Device: {device}\n")

    # Training loop
    for epoch in range(config.warmup_epochs):
        net.train()
        total_policy_loss = 0
        total_value_loss = 0
        n_batches = 0

        for batch_states, batch_policies, batch_values in loader:
            batch_states = batch_states.to(device)
            batch_policies = batch_policies.to(device)
            batch_values = batch_values.to(device)

            # Create legal masks from states (all expert moves are legal)
            masks = (batch_policies > 0).float()

            log_policy, value = net(batch_states, masks)

            # Policy loss: cross-entropy with soft targets
            policy_loss = -(batch_policies * log_policy).sum(dim=1).mean()
            # Value loss: MSE
            value_loss = F.mse_loss(value.squeeze(-1), batch_values)

            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

        avg_p = total_policy_loss / max(n_batches, 1)
        avg_v = total_value_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1:2d}/{config.warmup_epochs}: "
              f"policy_loss={avg_p:.4f}  value_loss={avg_v:.4f}")

    # Save warmup checkpoint
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = _save_checkpoint(net, optimizer, config, "model_warmup")
    print(f"\nSaved warmup model: {path}")

    # Sanity check
    print("\nSanity check: evaluating vs PathFollower...")
    wr = quick_eval(net, device, config, PathFollower(), num_games=20, net_sims=16)
    print(f"  Win rate vs PathFollower: {wr:.0%}")

    if wr < 0.75:
        print(f"\n  WARNING: Win rate {wr:.0%} < 75%. Warmup may be too weak.")
        print("  Consider: more MCTSExpert games, more epochs, or higher rollouts.")
        print("  Aborting to prevent wasting time on self-play.")
        return net

    print("\n  Warmup looks good!")
    return net


# ---------------------------------------------------------------------------
# Phase 2: Self-play improvement
# ---------------------------------------------------------------------------

def _self_play_game(net: QuoridorNet, opponent_net: QuoridorNet,
                    device: torch.device, config: Config,
                    net_player: int = 0) -> list:
    """Play one game, return training samples for the net's moves.

    Returns list of (state, mcts_policy, raw_policy, player).
    Value is assigned after the game ends.
    """
    mcts = GumbelMCTS(net, device, config)
    opp_mcts = GumbelMCTS(opponent_net, device, config)

    game = QuoridorGame(2)
    history = []  # (state, mcts_probs, raw_policy, player)

    for step in range(config.max_game_length):
        if game.is_over:
            break

        cp = game.current_player
        state = game.to_tensor()
        mask = MoveEncoder.legal_mask(game)

        if cp == net_player:
            # Our network's turn — collect training data
            raw_policy, _ = net.predict(state, mask, device)
            temp = 1.0 if step < config.temperature_moves * 2 else 0.1
            mcts_probs = mcts.search(game, add_noise=True)
            action = select_action(mcts_probs, temperature=temp)
            history.append((state, mcts_probs, raw_policy, cp))
        else:
            # Opponent's turn
            opp_probs = opp_mcts.search(game, add_noise=False)
            action = select_action(opp_probs, temperature=0.1)

        pos = game.pawns[cp]
        move = MoveEncoder.decode(action, pos)

        # Safety: validate move
        if mask[action] < 0.5:
            legal = game.get_legal_moves()
            move = random.choice(legal)

        game.make_move(move)

    return history, game


def _self_play_game_vs_heuristic(net: QuoridorNet, device: torch.device,
                                 config: Config, opponent,
                                 net_player: int = 0) -> tuple:
    """Play net vs heuristic opponent."""
    mcts = GumbelMCTS(net, device, config)
    game = QuoridorGame(2)
    history = []

    for step in range(config.max_game_length):
        if game.is_over:
            break

        cp = game.current_player
        state = game.to_tensor()
        mask = MoveEncoder.legal_mask(game)

        if cp == net_player:
            raw_policy, _ = net.predict(state, mask, device)
            temp = 1.0 if step < config.temperature_moves * 2 else 0.1
            mcts_probs = mcts.search(game, add_noise=True)
            action = select_action(mcts_probs, temperature=temp)
            history.append((state, mcts_probs, raw_policy, cp))
            pos = game.pawns[cp]
            move = MoveEncoder.decode(action, pos)
            if mask[action] < 0.5:
                move = random.choice(game.get_legal_moves())
        else:
            move = opponent.choose_move(game)

        game.make_move(move)

    return history, game


def self_play_phase(config: Config, device: torch.device,
                    warmup_net: QuoridorNet) -> QuoridorNet:
    """Phase 2: Improve via self-play with evaluation gates."""
    print("\n" + "=" * 60)
    print("PHASE 2: Self-play Improvement")
    print("=" * 60)

    # Current best network starts as warmup
    best_net = _make_net(config).to(device)
    best_net.load_state_dict(warmup_net.state_dict())

    # Candidate network
    candidate_net = _make_net(config).to(device)
    candidate_net.load_state_dict(warmup_net.state_dict())

    optimizer = torch.optim.Adam(candidate_net.parameters(), lr=config.train_lr,
                                 weight_decay=config.train_weight_decay)

    # Replay buffer
    replay_buffer = deque(maxlen=config.replay_buffer_size)

    # Past checkpoints for opponent pool
    past_checkpoints = []

    # Heuristic opponents
    heuristic_opponents = [PathFollower(), HeuristicExpert()]

    # Tracking
    best_policy_loss = float("inf")
    stale_count = 0
    consecutive_fails = 0

    for iteration in range(1, config.num_iterations + 1):
        t0 = time.time()
        print(f"\n--- Iteration {iteration}/{config.num_iterations} ---")

        # --- Self-play: generate games ---
        candidate_net.eval()
        best_net.eval()
        new_samples = []
        game_lengths = []

        for g_idx in range(config.games_per_iteration):
            net_player = g_idx % 2

            # Opponent selection
            r = random.random()
            if r < 0.30:
                # Self-play
                history, game = _self_play_game(
                    candidate_net, candidate_net, device, config, net_player)
            elif r < 0.65:
                # vs warmup/best
                history, game = _self_play_game(
                    candidate_net, best_net, device, config, net_player)
            elif r < 0.85 and past_checkpoints:
                # vs past checkpoint
                opp_path = random.choice(past_checkpoints)
                opp_net, _ = _load_checkpoint(opp_path, config, device)
                opp_net.eval()
                history, game = _self_play_game(
                    candidate_net, opp_net, device, config, net_player)
            else:
                # vs heuristic
                opp = random.choice(heuristic_opponents)
                history, game = _self_play_game_vs_heuristic(
                    candidate_net, device, config, opp, net_player)

            # Assign values
            winner = game.winner
            for state, mcts_probs, raw_policy, player in history:
                if winner is not None:
                    value = 1.0 if winner == player else -1.0
                else:
                    d0 = game.shortest_path_length(player) or 20
                    d1 = game.shortest_path_length(1 - player) or 20
                    value = float(np.tanh((d1 - d0) / 4.0))

                # KL filtering: only keep positions where MCTS improved policy
                kl = _kl_divergence(mcts_probs, raw_policy)
                if kl > config.kl_threshold:
                    new_samples.append((state, mcts_probs, value))

            game_lengths.append(len(game.move_history))

        replay_buffer.extend(new_samples)
        avg_len = np.mean(game_lengths) if game_lengths else 0
        print(f"  Games: {config.games_per_iteration}, "
              f"avg length: {avg_len:.0f}, "
              f"new samples (KL-filtered): {len(new_samples)}, "
              f"buffer: {len(replay_buffer)}")

        if avg_len > config.max_avg_game_length:
            print(f"  WARNING: Avg game length {avg_len:.0f} > {config.max_avg_game_length}. "
                  "Games may be too long — check wall play.")

        # --- Training ---
        if len(replay_buffer) < config.min_replay_for_training:
            print(f"  Buffer too small ({len(replay_buffer)} < "
                  f"{config.min_replay_for_training}), skipping training")
            continue

        candidate_net.train()
        buf = list(replay_buffer)
        random.shuffle(buf)

        states = torch.tensor(np.array([s[0] for s in buf]))
        policies = torch.tensor(np.array([s[1] for s in buf]))
        values = torch.tensor(np.array([s[2] for s in buf], dtype=np.float32))

        dataset = TensorDataset(states, policies, values)
        bs = min(config.train_batch_size, len(buf))
        loader = DataLoader(dataset, batch_size=bs,
                            shuffle=True, drop_last=False)

        total_ploss = 0
        total_vloss = 0
        n_batches = 0

        for epoch in range(config.train_epochs_per_iter):
            for batch_s, batch_p, batch_v in loader:
                batch_s = batch_s.to(device)
                batch_p = batch_p.to(device)
                batch_v = batch_v.to(device)

                masks = (batch_p > 0).float()
                log_policy, value = candidate_net(batch_s, masks)

                policy_loss = -(batch_p * log_policy).sum(dim=1).mean()
                value_loss = F.mse_loss(value.squeeze(-1), batch_v)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(candidate_net.parameters(),
                                               config.gradient_clip)
                optimizer.step()

                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                n_batches += 1

        avg_pl = total_ploss / max(n_batches, 1)
        avg_vl = total_vloss / max(n_batches, 1)
        print(f"  Training: policy_loss={avg_pl:.4f}  value_loss={avg_vl:.4f}")

        # Track policy loss stagnation
        if avg_pl < best_policy_loss - 0.001:
            best_policy_loss = avg_pl
            stale_count = 0
        else:
            stale_count += 1

        if stale_count >= config.max_stale_iterations:
            print(f"\n  HALT: Policy loss hasn't improved for "
                  f"{config.max_stale_iterations} iterations.")
            break

        # --- Evaluation gate ---
        candidate_net.eval()
        wr = quick_eval(candidate_net, device, config, PathFollower(),
                        num_games=config.eval_games, net_sims=32)
        wr_best = quick_eval(candidate_net, device, config, HeuristicExpert(),
                             num_games=config.eval_games // 2, net_sims=32)
        print(f"  Eval: vs PathFollower={wr:.0%}, vs HeuristicExpert={wr_best:.0%}")

        # Compare vs best network
        # Simple check: if candidate performs well against heuristics, accept it
        if wr >= 0.60 and wr_best >= config.eval_threshold:
            best_net.load_state_dict(candidate_net.state_dict())
            consecutive_fails = 0
            print(f"  -> ACCEPTED as new best")
        else:
            consecutive_fails += 1
            candidate_net.load_state_dict(best_net.state_dict())
            print(f"  -> REJECTED (reverting to best) [{consecutive_fails}/"
                  f"{config.consecutive_fails_abort}]")

        if consecutive_fails >= config.consecutive_fails_abort:
            if wr < config.min_win_rate_abort:
                print(f"\n  HALT: Win rate {wr:.0%} < {config.min_win_rate_abort:.0%} "
                      f"for {config.consecutive_fails_abort} consecutive iterations.")
                break
            else:
                consecutive_fails = 0  # Reset if still reasonable

        # --- Checkpoint ---
        if iteration % config.save_every == 0:
            path = _save_checkpoint(
                best_net, optimizer, config,
                f"model_iter_{iteration:04d}",
                extra={"iteration": iteration}
            )
            past_checkpoints.append(path)
            print(f"  Checkpoint: {path}")

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

    # Save final model
    path = _save_checkpoint(best_net, optimizer, config, "model_best")
    print(f"\nFinal model saved: {path}")

    return best_net


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) for probability distributions."""
    mask = p > 1e-8
    p_safe = np.clip(p[mask], 1e-8, 1.0)
    q_safe = np.clip(q[mask], 1e-8, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Quoridor AI")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip Phase 1 (requires existing checkpoint)")
    parser.add_argument("--warmup-only", action="store_true",
                        help="Only run Phase 1")
    parser.add_argument("--games", type=int, default=None,
                        help="Override warmup game count")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override self-play iteration count")
    parser.add_argument("--games-per-iter", type=int, default=None,
                        help="Override games per iteration")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from specific checkpoint")
    parser.add_argument("--fast", action="store_true",
                        help="Quick smoke test (heuristic-only experts, fewer epochs)")
    args = parser.parse_args()

    config = Config()
    if args.fast:
        config.warmup_mcts_fraction = 0.0  # no slow MCTS expert
        config.warmup_epochs = 3
        config.num_workers = 1
    if args.games:
        config.warmup_games = args.games
    if args.iterations:
        config.num_iterations = args.iterations
    if args.games_per_iter:
        config.games_per_iteration = args.games_per_iter

    device = config.device
    print(f"Device: {device}")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    if args.skip_warmup:
        # Load existing checkpoint
        ckpt_path = args.checkpoint or os.path.join(config.checkpoint_dir,
                                                     "model_warmup.pt")
        if not os.path.exists(ckpt_path):
            print(f"ERROR: No checkpoint found at {ckpt_path}")
            print("Run without --skip-warmup first.")
            return
        print(f"Loading checkpoint: {ckpt_path}")
        warmup_net, _ = _load_checkpoint(ckpt_path, config, device)
    else:
        warmup_net = warmup(config, device)

    if args.warmup_only:
        print("\nWarmup complete. Exiting (--warmup-only).")
        return

    self_play_phase(config, device, warmup_net)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
