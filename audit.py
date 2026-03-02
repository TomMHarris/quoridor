"""
Audit & Fix: diagnose exactly why self-play degrades the model.

Tests three hypotheses:
  1. MCTS search doesn't actually improve on raw network policy
  2. Pawn boost corrupts training targets (feedback loop)
  3. Replay buffer retains too much stale data

Then runs a quick fixed training test to verify.

Usage:
    python audit.py                 # diagnose problems
    python audit.py --fix --test    # apply fixes and run quick test
"""

import argparse
import glob
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict

import numpy as np
import torch

from game import QuoridorGame, MoveEncoder
from network import QuoridorNet, count_parameters
from mcts import MCTS, select_action
from config import Config


def load_latest(device="cpu"):
    files = sorted(glob.glob("checkpoints/model_iter_*.pt"))
    if not files:
        print("No checkpoints found")
        return None, None
    path = files[-1]
    data = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**{k: v for k, v in data["config"].items()
                    if k in Config.__dataclass_fields__})
    net = QuoridorNet(
        input_planes=cfg.input_planes,
        num_blocks=cfg.num_res_blocks,
        num_channels=cfg.num_channels,
    ).to(device)
    net.load_state_dict(data["model_state_dict"])
    net.eval()
    return net, data.get("iteration", "?")


# =================================================================
# TEST 1: Does MCTS actually improve move quality?
# =================================================================

def test_mcts_improvement(net, device="cpu"):
    """
    Compare raw network policy vs MCTS policy on key positions.
    If MCTS makes the policy WORSE, the training loop is broken.
    """
    print("\n" + "=" * 60)
    print("  TEST 1: Does MCTS improve on raw network?")
    print("=" * 60)

    positions = [
        ("Opening", QuoridorGame(2)),
    ]

    g2 = QuoridorGame(2)
    g2.make_move(("move", (7, 4)))
    g2.make_move(("move", (1, 4)))
    positions.append(("After 2 moves", g2))

    g3 = QuoridorGame(2)
    g3.pawns = [(1, 4), (7, 4)]
    positions.append(("P0 can win", g3))

    for name, game in positions:
        state = game.to_tensor()
        mask = MoveEncoder.legal_mask(game)
        raw_policy, raw_value = net.predict(state, mask, device)

        # Run MCTS with 100 sims (no pawn boost to see true effect)
        mcts = MCTS(net, device, num_simulations=100, c_puct=1.5)
        mcts_policy = mcts.search(game, add_noise=False)

        # Measure pawn % and forward move %
        pos = game.pawns[game.current_player]
        raw_pawn = sum(raw_policy[MoveEncoder.encode(m, pos)]
                       for m in game.get_legal_pawn_moves())
        mcts_pawn = sum(mcts_policy[MoveEncoder.encode(m, pos)]
                        for m in game.get_legal_pawn_moves())

        # Find forward move
        cp = game.current_player
        if cp == 0:
            fwd = (game.pawns[0][0] - 1, game.pawns[0][1])
        else:
            fwd = (game.pawns[1][0] + 1, game.pawns[1][1])
        fwd_move = ("move", fwd)
        fwd_idx = MoveEncoder.encode(fwd_move, pos) if fwd_move in game.get_legal_pawn_moves() else -1
        raw_fwd = raw_policy[fwd_idx] if fwd_idx >= 0 else 0
        mcts_fwd = mcts_policy[fwd_idx] if fwd_idx >= 0 else 0

        # Top MCTS move
        top_idx = np.argmax(mcts_policy)
        top_move = MoveEncoder.decode(top_idx, pos)

        print(f"\n  {name}:")
        print(f"    Raw network:  pawn={raw_pawn*100:.1f}%  fwd={raw_fwd*100:.1f}%  value={raw_value:.3f}")
        print(f"    MCTS (100):   pawn={mcts_pawn*100:.1f}%  fwd={mcts_fwd*100:.1f}%  top={top_move}")

        if mcts_pawn < raw_pawn * 0.5:
            print(f"    ⚠ WARNING: MCTS REDUCES pawn weight! Boost may be distorting search.")
        if mcts_fwd > raw_fwd:
            print(f"    ✓ MCTS improves forward move weight")
        else:
            print(f"    ⚠ MCTS doesn't improve forward move")


# =================================================================
# TEST 2: Does pawn boost corrupt training targets?
# =================================================================

def test_boost_corruption(net, device="cpu"):
    """
    Compare MCTS with boost vs without boost.
    If the boost changes what MCTS *concludes* (not just explores),
    it's corrupting the training signal.
    """
    print("\n" + "=" * 60)
    print("  TEST 2: Does pawn boost corrupt MCTS output?")
    print("=" * 60)

    game = QuoridorGame(2)
    pos = game.pawns[0]

    # MCTS with boost (current code)
    mcts_boost = MCTS(net, device, num_simulations=100, c_puct=1.5)
    policy_boost = mcts_boost.search(game, add_noise=False)

    # MCTS without boost — temporarily disable it
    original_expand = MCTS._expand

    def _expand_no_boost(self, node):
        game = node.game
        if game.is_over:
            return 0.0
        state = game.to_tensor()
        legal_mask = MoveEncoder.legal_mask(game)
        policy, value = self.network.predict(state, legal_mask, self.device)
        # NO BOOST
        probable_moves = game.get_probable_moves()
        pawn_pos = game.pawns[game.current_player]
        from mcts import MCTSNode
        for move in probable_moves:
            idx = MoveEncoder.encode(move, pawn_pos)
            child_game = game.clone()
            child_game.make_move(move)
            child = MCTSNode(game=child_game, parent=node, move=move, prior=policy[idx])
            node.children[move] = child
        return -value

    MCTS._expand = _expand_no_boost
    mcts_no_boost = MCTS(net, device, num_simulations=100, c_puct=1.5)
    policy_no_boost = mcts_no_boost.search(game, add_noise=False)
    MCTS._expand = original_expand  # restore

    # Compare
    boost_pawn = sum(policy_boost[MoveEncoder.encode(m, pos)]
                     for m in game.get_legal_pawn_moves())
    no_boost_pawn = sum(policy_no_boost[MoveEncoder.encode(m, pos)]
                        for m in game.get_legal_pawn_moves())

    print(f"\n  Opening position:")
    print(f"    With boost:    pawn visits = {boost_pawn*100:.1f}%")
    print(f"    Without boost: pawn visits = {no_boost_pawn*100:.1f}%")
    print(f"    Difference:    {(boost_pawn - no_boost_pawn)*100:+.1f}%")

    if abs(boost_pawn - no_boost_pawn) > 0.15:
        print(f"\n    ⚠ BOOST SIGNIFICANTLY CHANGES MCTS OUTPUT!")
        print(f"    The network trains on boosted targets, learns the boost,")
        print(f"    then boost deactivates, and model drifts back.")
        print(f"    → FIX: Remove boost. Warmup model doesn't need it.")
    else:
        print(f"\n    ✓ Boost has minimal effect on MCTS output (OK)")


# =================================================================
# TEST 3: Replay buffer staleness
# =================================================================

def test_buffer_staleness():
    """
    Check how much of the replay buffer is stale (from much older iterations).
    """
    print("\n" + "=" * 60)
    print("  TEST 3: Replay buffer analysis")
    print("=" * 60)

    cfg = Config()
    print(f"\n  Buffer size: {cfg.replay_buffer_size}")
    print(f"  Games per iteration: {cfg.num_self_play_games}")
    avg_game_len = 40  # rough estimate
    samples_per_iter = cfg.num_self_play_games * avg_game_len
    iters_in_buffer = cfg.replay_buffer_size / samples_per_iter

    print(f"  ~Samples per iteration: {samples_per_iter}")
    print(f"  ~Iterations retained: {iters_in_buffer:.0f}")
    print(f"  Training epochs: {cfg.num_epochs}")

    if iters_in_buffer > 5:
        print(f"\n  ⚠ BUFFER TOO LARGE!")
        print(f"    {iters_in_buffer:.0f} iterations of data, but model changes")
        print(f"    significantly each iteration. Old data has wrong targets.")
        print(f"    → FIX: Shrink to ~{3 * samples_per_iter} (3 iterations)")
    else:
        print(f"\n  ✓ Buffer size looks reasonable")

    if cfg.num_epochs > 2:
        print(f"\n  ⚠ TOO MANY EPOCHS ({cfg.num_epochs})")
        print(f"    Overfitting to stale data in the buffer.")
        print(f"    → FIX: Reduce to 2 epochs")


# =================================================================
# Quick fix + test
# =================================================================

def quick_fix_test(device="auto"):
    """
    Apply fixes and run 5 iterations to see if model improves.
    Compares iter 0 (warmup) vs iter 5 (after training) vs Greedy.
    """
    print("\n" + "=" * 60)
    print("  QUICK FIX TEST: 5 iterations with fixed settings")
    print("=" * 60)

    cfg = Config()
    if device == "auto":
        device = cfg.resolve_device()

    # Load warmup model
    warmup_path = "checkpoints/model_iter_0000.pt"
    if not os.path.exists(warmup_path):
        print("  No warmup model found. Run warmup.py first.")
        return

    data = torch.load(warmup_path, map_location=device, weights_only=False)
    network = QuoridorNet(
        input_planes=cfg.input_planes,
        num_blocks=cfg.num_res_blocks,
        num_channels=cfg.num_channels,
    ).to(device)
    network.load_state_dict(data["model_state_dict"])

    # ── FIXED SETTINGS ──
    num_sims = 100
    num_games = 20
    num_iterations = 5
    buffer_size = 3000       # FIX 1: small buffer (not 20,000)
    num_epochs = 2           # FIX 2: fewer epochs (not 4)
    # FIX 3: no pawn boost (warmup model doesn't need it)

    replay_buffer = deque(maxlen=buffer_size)

    # Benchmark warmup model first
    print("\n  Benchmarking warmup model (iter 0) vs Greedy...")
    warmup_wr = quick_benchmark(network, device, num_sims)
    print(f"  Warmup win rate vs Greedy: {warmup_wr*100:.0f}%")

    # Training loop (simplified, no boost, small buffer)
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    for iteration in range(1, num_iterations + 1):
        print(f"\n  --- Iteration {iteration}/{num_iterations} ---")
        network.eval()

        # Self-play (NO pawn boost)
        mcts = MCTS(network, device, num_simulations=num_sims, c_puct=1.5,
                     dirichlet_alpha=0.3, dirichlet_epsilon=0.25)

        # Monkey-patch to remove boost for this test
        original_expand = MCTS._expand
        def _expand_no_boost(self, node):
            game_node = node.game
            if game_node.is_over:
                return 0.0
            state = game_node.to_tensor()
            legal_mask = MoveEncoder.legal_mask(game_node)
            policy, value = self.network.predict(state, legal_mask, self.device)
            probable_moves = game_node.get_probable_moves()
            pawn_pos = game_node.pawns[game_node.current_player]
            from mcts import MCTSNode
            for move in probable_moves:
                idx = MoveEncoder.encode(move, pawn_pos)
                child_game = game_node.clone()
                child_game.make_move(move)
                child = MCTSNode(game=child_game, parent=node, move=move, prior=policy[idx])
                node.children[move] = child
            return -value
        MCTS._expand = _expand_no_boost

        samples = []
        wins = draws = 0
        total_moves = 0

        for g in range(num_games):
            game = QuoridorGame(2)
            trajectory = []
            move_count = 0

            while not game.is_over and move_count < 200:
                state = game.to_tensor()
                action_probs = mcts.search(game, add_noise=True)
                temp = 1.0 if move_count < 15 else 0.1
                action_idx = select_action(action_probs, temperature=temp)
                trajectory.append((state, action_probs, game.current_player))
                pawn_pos = game.pawns[game.current_player]
                move = MoveEncoder.decode(action_idx, pawn_pos)
                game.make_move(move)
                move_count += 1

            winner = game.winner if game.winner is not None else -1
            if winner >= 0:
                wins += 1
            else:
                draws += 1
            total_moves += move_count

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

        MCTS._expand = original_expand  # restore

        replay_buffer.extend(samples)
        avg_len = total_moves / num_games
        print(f"    {num_games} games, avg {avg_len:.0f} moves, "
              f"{wins} decisive, {draws} draws")

        # Train (fewer epochs)
        if len(replay_buffer) >= 200:
            network.train()
            states = np.array([s[0] for s in replay_buffer])
            policies = np.array([s[1] for s in replay_buffer])
            values = np.array([s[2] for s in replay_buffer], dtype=np.float32)

            dataset = TensorDataset(
                torch.from_numpy(states),
                torch.from_numpy(policies),
                torch.from_numpy(values).unsqueeze(1),
            )
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            optimizer = optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4)

            total_loss = 0
            n = 0
            for epoch in range(num_epochs):  # ONLY 2 epochs
                for bs, bp, bv in loader:
                    bs, bp, bv = bs.to(device), bp.to(device), bv.to(device)
                    pp, pv = network(bs)
                    log_p = torch.log(pp.clamp(min=1e-8))
                    p_loss = -torch.sum(bp * log_p, dim=1).mean()
                    v_loss = nn.MSELoss()(pv, bv)
                    loss = p_loss + v_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n += 1

            print(f"    Loss: {total_loss/n:.4f}, buffer: {len(replay_buffer)}")

    # Benchmark after training
    print(f"\n  Benchmarking trained model (iter {num_iterations}) vs Greedy...")
    network.eval()
    trained_wr = quick_benchmark(network, device, num_sims)
    print(f"  Trained win rate vs Greedy: {trained_wr*100:.0f}%")

    print(f"\n  {'='*50}")
    print(f"  RESULT:")
    print(f"    Warmup (iter 0):  {warmup_wr*100:.0f}% vs Greedy")
    print(f"    Trained (iter 5): {trained_wr*100:.0f}% vs Greedy")
    if trained_wr >= warmup_wr:
        print(f"    ✓ Self-play IMPROVED or MAINTAINED performance!")
    else:
        print(f"    ⚠ Self-play DEGRADED performance")
    print(f"  {'='*50}")


def quick_benchmark(network, device, num_sims, num_games=10):
    """Quick win rate vs Greedy."""
    from mcts import MCTS, select_action

    mcts = MCTS(network, device, num_simulations=num_sims, c_puct=1.5)
    wins = 0

    for g in range(num_games):
        game = QuoridorGame(2)
        ai_player = g % 2  # alternate sides

        while not game.is_over and len(game.move_history) < 200:
            if game.current_player == ai_player:
                probs = mcts.search(game, add_noise=False)
                idx = select_action(probs, temperature=0.1)
                pos = game.pawns[game.current_player]
                move = MoveEncoder.decode(idx, pos)
            else:
                # Greedy
                path = game.shortest_path(game.current_player)
                if path and len(path) > 1:
                    move = ("move", path[1])
                    if move not in game.get_legal_pawn_moves():
                        pawn_moves = game.get_legal_pawn_moves()
                        move = random.choice(pawn_moves) if pawn_moves else random.choice(game.get_legal_moves())
                else:
                    pawn_moves = game.get_legal_pawn_moves()
                    move = random.choice(pawn_moves) if pawn_moves else random.choice(game.get_legal_moves())

            game.make_move(move)

        if game.winner == ai_player:
            wins += 1

    return wins / num_games


# =================================================================
# Entry point
# =================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Quoridor training")
    parser.add_argument("--fix", action="store_true",
                        help="Apply fixes and run quick test")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = Config()
    device = args.device if args.device != "auto" else cfg.resolve_device()

    net, iteration = load_latest(device)
    if net is None:
        exit(1)

    print(f"Loaded iteration {iteration}")

    # Run diagnostics
    test_mcts_improvement(net, device)
    test_boost_corruption(net, device)
    test_buffer_staleness()

    # Run fix test if requested
    if args.fix:
        quick_fix_test(device)
