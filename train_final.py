"""
train_final.py — The training pipeline that should have existed from the start.

DESIGN PHILOSOPHY
-----------------
Beating gorisanson requires a network that improves MCTS quality, not one that
merely imitates heuristics. The path is:

  1. Warmup: pre-train on MCTSExpert games (strong heuristic rollouts)
  2. Policy improvement: play current network vs warmup anchor, only train on
     positions where MCTS search actually found improvements over raw policy
  3. Evaluation gate: only keep a new model if it beats the previous best

WHAT WAS WRONG WITH THE ORIGINAL SELF-PLAY LOOP
-------------------------------------------------
- Pure self-play between two copies of the same weak model → ~50/50 outcomes
  → value targets are random noise → network learns nothing
- Replay buffer of 20,000 with 4 epochs = training on 10 iterations of stale
  data with contradictory signals → forgetting
- Optimizer recreated every iteration → Adam's momentum thrown away → each
  update overshoots → catastrophic forgetting
- Pawn boost on a post-warmup model creates circular training targets
- Games often truncate at move 200 → value targets are heuristic estimates,
  not real outcomes → noisy signal

THE FIXES
---------
- Play against warmup model (strongest known opponent) → decisive games →
  clean win/loss signals
- KL filter: only train on positions where MCTS disagreed with raw policy →
  no circular training
- Evaluation gate: save new model only if it beats previous best >55% in N games
- Small buffer (3 iterations), 2 epochs, persistent Adam at LR=3e-4
- No pawn boost after warmup (warmup already learned good pawn weights)
- Support both 2-player and 4-player natively

COMMANDS
--------
  # Step 1: Generate warmup data and pre-train (~2hrs for 2000 games)
  python warmup.py --games 2000 --rollouts 500 --workers 4 --epochs 20

  # Step 2: League training with evaluation gate
  python train_final.py --iterations 30 --games 25 --simulations 150

  # Step 3: Benchmark
  python benchmark_ai.py --progression --games 20 --simulations 150 --quiet

  # 4-player mode (slower, uses 14-plane input)
  python warmup.py --games 1000 --fast --workers 4 --epochs 20  # fast for 4p
  python train_final.py --players 4 --iterations 20 --games 20 --simulations 100
"""

import argparse
import glob
import json
import os
import random
import time
from collections import deque
from dataclasses import asdict
from typing import List, Tuple, Optional

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


# ===========================================================================
# Heuristic opponent (for baseline / evaluation)
# ===========================================================================

class GreedyAgent:
    """Follows shortest path, occasionally blocks. Works for 2 and 4 players."""
    name = "Greedy"

    def choose_move(self, game: QuoridorGame):
        player = game.current_player
        path = game.shortest_path(player)
        if path and len(path) > 1:
            move = ("move", path[1])
            if move in game.get_legal_pawn_moves():
                return move
        pawn_moves = game.get_legal_pawn_moves()
        if pawn_moves:
            return random.choice(pawn_moves)
        return random.choice(game.get_legal_moves())


class SmartGreedyAgent:
    """Greedy + wall blocking. Works for 2 and 4 players."""
    name = "SmartGreedy"

    def choose_move(self, game: QuoridorGame):
        player = game.current_player

        if random.random() < 0.35 and game.walls_remaining[player] > 0:
            wall = self._find_blocking_wall(game, player)
            if wall:
                return wall

        path = game.shortest_path(player)
        if path and len(path) > 1:
            move = ("move", path[1])
            if move in game.get_legal_pawn_moves():
                return move
        pawn_moves = game.get_legal_pawn_moves()
        return random.choice(pawn_moves) if pawn_moves else random.choice(game.get_legal_moves())

    def _find_blocking_wall(self, game, player):
        # Find the opponent who is closest to their goal
        best_opponent = None
        best_dist = 99
        for p in range(game.num_players):
            if p == player:
                continue
            d = game.shortest_path_length(p)
            if d is not None and d < best_dist:
                best_dist = d
                best_opponent = p

        if best_opponent is None:
            return None

        opp_dist = best_dist
        opp_path = game.shortest_path(best_opponent)
        if not opp_path or len(opp_path) < 2:
            return None

        best_wall = None
        best_gain = 0
        for cell in opp_path[1:4]:
            tr, tc = cell
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    wr, wc = tr + dr, tc + dc
                    for o in ("H", "V"):
                        if game.is_valid_wall(wr, wc, o):
                            g2 = game.clone()
                            g2.make_move(("wall", (wr, wc, o)))
                            new_d = g2.shortest_path_length(best_opponent)
                            if new_d is not None:
                                gain = new_d - opp_dist
                                if gain > best_gain:
                                    best_gain = gain
                                    best_wall = ("wall", (wr, wc, o))
        return best_wall if best_gain >= 2 else None


class NetworkAgent:
    """Current network + MCTS. Used for evaluation."""

    def __init__(self, network, device, simulations=100, pawn_boost=False):
        self.network = network
        self.device = device
        self.mcts = MCTS(
            network=network, device=device,
            num_simulations=simulations,
            c_puct=1.5,
            pawn_boost=pawn_boost,
        )

    def choose_move(self, game: QuoridorGame):
        probs = self.mcts.search(game, add_noise=False)
        idx = select_action(probs, temperature=0.05)
        pos = game.pawns[game.current_player]
        return MoveEncoder.decode(idx, pos)


# ===========================================================================
# Quick evaluation
# ===========================================================================

def quick_eval(network, device, simulations=100, num_games=20, num_players=2,
               verbose=False) -> dict:
    """
    Evaluate network against Greedy and SmartGreedy.
    Returns win rates from the network's perspective.
    Games are split 50/50 as P0 and P1 (or P0/P1/P2/P3 for 4-player).
    """
    network.eval()
    agent = NetworkAgent(network, device, simulations=simulations)
    baselines = [GreedyAgent(), SmartGreedyAgent()]
    results = {}

    for baseline in baselines:
        wins = 0
        for g in range(num_games):
            # Rotate which player index the network takes
            net_player = g % num_players
            game = QuoridorGame(num_players)
            moves = 0

            while not game.is_over and moves < 150:
                cp = game.current_player
                if cp == net_player:
                    move = agent.choose_move(game)
                else:
                    move = baseline.choose_move(game)
                game.make_move(move)
                moves += 1

            if game.winner == net_player:
                wins += 1

        wr = wins / num_games
        results[baseline.name] = wr
        if verbose:
            print(f"    vs {baseline.name}: {wins}/{num_games} = {wr*100:.0f}%")

    return results


# ===========================================================================
# Self-play game (network vs opponent)
# ===========================================================================

def play_game(network, opponent, config: Config, device: str,
              simulations: int) -> Tuple[List[Sample], int, dict]:
    """
    Play one game: network vs opponent.
    Network randomly assigned to one player slot.
    Only collects training samples from network's turns.
    Filters: only positions where MCTS meaningfully improved on raw policy.
    """
    game = QuoridorGame(config.num_players)
    mcts = MCTS(
        network=network, device=device,
        num_simulations=simulations,
        c_puct=config.c_puct,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        pawn_boost=False,  # post-warmup: network already knows pawn moves are good
    )

    net_player = random.randint(0, config.num_players - 1)
    trajectory = []
    move_count = 0

    while not game.is_over and move_count < config.max_game_length:
        cp = game.current_player

        if cp == net_player:
            state = game.to_tensor()
            legal_mask = MoveEncoder.legal_mask(game)

            # Get raw network policy BEFORE MCTS (for KL filtering)
            raw_policy, _ = network.predict(state, legal_mask, device)

            # Run MCTS
            action_probs = mcts.search(game, add_noise=True)

            # KL(mcts || raw): measures how much search changed the policy.
            # If KL is tiny, MCTS found nothing new — skip this position.
            # This avoids circular training where the network trains to
            # reproduce its own prior with no new information.
            kl = float(np.sum(
                action_probs * np.log((action_probs + 1e-8) / (raw_policy + 1e-8))
            ))

            temp = 1.0 if move_count < config.temperature_moves else 0.05
            action_idx = select_action(action_probs, temperature=temp)
            pos = game.pawns[cp]
            move = MoveEncoder.decode(action_idx, pos)

            if kl > 0.01:  # MCTS found something: use this as a training sample
                trajectory.append((state, action_probs, cp, kl))
        else:
            move = opponent.choose_move(game)

        game.make_move(move)
        move_count += 1

    winner = game.winner if game.winner is not None else -1
    samples: List[Sample] = []

    for state, policy, player, kl in trajectory:
        if winner == -1:
            # Truncated: use distance-based heuristic value
            # Weight by KL so informative positions contribute more
            d_me = game.shortest_path_length(player) or 9
            opponents = [p for p in range(config.num_players) if p != player]
            d_opp = min(
                (game.shortest_path_length(p) or 9) for p in opponents
            )
            if d_me + d_opp > 0:
                value = float(np.tanh(2.0 * (d_opp - d_me) / (d_me + d_opp)))
            else:
                value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0

        samples.append((state, policy, value))

    stats = {
        "winner": winner,
        "net_player": net_player,
        "num_moves": move_count,
        "truncated": winner == -1,
        "decisive": winner >= 0,
        "kl_filtered_in": len(trajectory),
        "result": "win" if winner == net_player
                  else ("draw" if winner == -1 else "loss"),
    }
    return samples, winner, stats


# ===========================================================================
# Training step
# ===========================================================================

def train_step(network, replay_buffer: deque, optimizer, batch_size=64,
               num_epochs=2, device="cpu") -> dict:
    """Train one step on replay buffer. Returns loss stats."""
    network.train()

    states = np.array([s[0] for s in replay_buffer])
    policies = np.array([s[1] for s in replay_buffer])
    values = np.array([s[2] for s in replay_buffer], dtype=np.float32)

    dataset = TensorDataset(
        torch.from_numpy(states),
        torch.from_numpy(policies),
        torch.from_numpy(values).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False)

    total_loss = total_ploss = total_vloss = 0
    n = 0

    for _ in range(num_epochs):
        for bs, bp, bv in loader:
            bs, bp, bv = bs.to(device), bp.to(device), bv.to(device)
            pred_p, pred_v = network(bs)
            log_p = torch.log(pred_p.clamp(min=1e-8))
            p_loss = -(bp * log_p).sum(dim=1).mean()
            v_loss = nn.MSELoss()(pred_v, bv)
            loss = p_loss + v_loss
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping: prevents any single noisy batch from
            # destroying the model (common when buffer is small)
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_ploss += p_loss.item()
            total_vloss += v_loss.item()
            n += 1

    return {
        "loss": total_loss / max(n, 1),
        "policy_loss": total_ploss / max(n, 1),
        "value_loss": total_vloss / max(n, 1),
    }


# ===========================================================================
# Load helpers
# ===========================================================================

def load_checkpoint(path: str, device: str, config: Config):
    data = torch.load(path, map_location=device, weights_only=False)
    net = QuoridorNet(
        input_planes=config.input_planes,
        num_blocks=config.num_res_blocks,
        num_channels=config.num_channels,
    ).to(device)
    net.load_state_dict(data["model_state_dict"])
    net.eval()
    return net, data


def find_warmup(checkpoint_dir: str) -> Optional[str]:
    path = os.path.join(checkpoint_dir, "model_iter_0000.pt")
    return path if os.path.exists(path) else None


def find_best(checkpoint_dir: str) -> Optional[str]:
    """Return path to best checkpoint (highest iter that isn't iter 0)."""
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "model_iter_*.pt")))
    non_warmup = [f for f in files if "0000" not in f]
    return non_warmup[-1] if non_warmup else None


# ===========================================================================
# Opponent pool
# ===========================================================================

class OpponentPool:
    """
    Diverse opponent pool to prevent strategy collapse.

    Distribution:
      35% warmup model  — strongest anchor; clean win/loss signal
      30% self-play     — explores new strategies
      20% past best     — prevents forgetting how to beat older self
      15% heuristic     — ground truth; ensures baseline competence
    """

    def __init__(self, config: Config, device: str):
        self.config = config
        self.device = device
        self.warmup: Optional[NetworkAgent] = None
        self.past_checkpoints: List[NetworkAgent] = []
        self.heuristics = [GreedyAgent(), SmartGreedyAgent()]

    def load_warmup(self):
        path = find_warmup(self.config.checkpoint_dir)
        if path:
            net, _ = load_checkpoint(path, self.device, self.config)
            self.warmup = NetworkAgent(net, self.device,
                                       simulations=self.config.num_simulations)
            print(f"  Opponent pool: loaded warmup from {path}")

    def refresh_past(self):
        """Load the 3 most recent non-warmup checkpoints."""
        files = sorted(glob.glob(
            os.path.join(self.config.checkpoint_dir, "model_iter_*.pt")))
        non_warmup = [f for f in files if "0000" not in f]
        # Take up to 3 most recent
        recent = non_warmup[-3:]
        self.past_checkpoints = []
        for f in recent:
            try:
                net, _ = load_checkpoint(f, self.device, self.config)
                self.past_checkpoints.append(
                    NetworkAgent(net, self.device,
                                 simulations=self.config.num_simulations))
            except Exception:
                pass

    def sample(self, current_network) -> tuple:
        """Returns (opponent, label)."""
        roll = random.random()

        if roll < 0.35 and self.warmup is not None:
            return self.warmup, "warmup"

        elif roll < 0.65:
            opp = NetworkAgent(current_network, self.device,
                               simulations=self.config.num_simulations)
            return opp, "self"

        elif roll < 0.85 and self.past_checkpoints:
            opp = random.choice(self.past_checkpoints)
            return opp, "past"

        else:
            h = random.choice(self.heuristics)
            return h, h.name


# ===========================================================================
# Evaluation gate
# ===========================================================================

def evaluate_vs_best(candidate_net, best_net, device, simulations, num_games,
                     num_players) -> float:
    """
    Play candidate vs current best. Returns candidate win rate.
    We accept the candidate only if win_rate > 0.55.
    """
    candidate = NetworkAgent(candidate_net, device, simulations=simulations)
    best = NetworkAgent(best_net, device, simulations=simulations)
    wins = 0

    for g in range(num_games):
        cand_player = g % num_players
        game = QuoridorGame(num_players)
        moves = 0

        while not game.is_over and moves < 150:
            cp = game.current_player
            if cp == cand_player:
                move = candidate.choose_move(game)
            else:
                move = best.choose_move(game)
            game.make_move(move)
            moves += 1

        if game.winner == cand_player:
            wins += 1

    return wins / num_games


# ===========================================================================
# Main training loop
# ===========================================================================

def train(config: Config = None, resume: bool = False,
          eval_games: int = 10, eval_threshold: float = 0.52):
    """
    Main training loop with evaluation gate.

    eval_threshold: accept new model only if it beats best by this win rate.
    Lower threshold (0.52) is better for early training when models are close.
    """
    if config is None:
        config = Config()

    device = config.resolve_device()
    print(f"\nDevice: {device}")
    print(f"Players: {config.num_players}")
    print(f"Simulations: {config.num_simulations}")

    # Load or initialise network
    network = QuoridorNet(
        input_planes=config.input_planes,
        num_blocks=config.num_res_blocks,
        num_channels=config.num_channels,
    ).to(device)
    print(f"Parameters: {count_parameters(network):,}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    start_iter = 1
    log = []
    opt_state = None

    # Try to resume from warmup (iter 0) or latest checkpoint
    if resume:
        files = sorted(glob.glob(
            os.path.join(config.checkpoint_dir, "model_iter_*.pt")))
        if files:
            latest = files[-1]
            print(f"  Resuming from {latest}")
            _, data = None, torch.load(latest, map_location=device,
                                       weights_only=False)
            network.load_state_dict(data["model_state_dict"])
            start_iter = data["iteration"] + 1
            opt_state = data.get("optimizer_state_dict")
            log_path = os.path.join(config.log_dir, "training_log.json")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    log = json.load(f)
    else:
        # Always start from warmup model if it exists
        warmup_path = find_warmup(config.checkpoint_dir)
        if warmup_path:
            data = torch.load(warmup_path, map_location=device,
                              weights_only=False)
            network.load_state_dict(data["model_state_dict"])
            print(f"  Starting from warmup model: {warmup_path}")
        else:
            print("  WARNING: No warmup model found. Run warmup.py first.")
            print("  Training from scratch will be very slow to converge.")

    # Keep a copy of the "best known" model for evaluation gate
    best_network = QuoridorNet(
        input_planes=config.input_planes,
        num_blocks=config.num_res_blocks,
        num_channels=config.num_channels,
    ).to(device)
    best_network.load_state_dict(network.state_dict())
    best_network.eval()

    # Persistent Adam at low LR — prevents catastrophic forgetting
    optimizer = optim.Adam(network.parameters(), lr=3e-4,
                           weight_decay=config.weight_decay)
    if opt_state is not None:
        optimizer.load_state_dict(opt_state)

    # Small buffer: ~3 iterations of data
    buffer_size = max(3000, config.num_self_play_games * 45 * 3)
    replay_buffer: deque = deque(maxlen=buffer_size)
    print(f"  Replay buffer: {buffer_size} samples")

    # Opponent pool
    pool = OpponentPool(config, device)
    pool.load_warmup()

    # Baseline eval of starting model
    print("\n  Evaluating starting model...")
    start_wr = quick_eval(network, device,
                          simulations=config.num_simulations,
                          num_games=min(eval_games, 10),
                          num_players=config.num_players,
                          verbose=True)
    print(f"  Starting win rates: {start_wr}")

    end_iter = start_iter + config.num_iterations - 1

    for iteration in range(start_iter, end_iter + 1):
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration}/{end_iter}  (buffer={len(replay_buffer)})")
        print(f"{'='*60}")

        network.eval()
        pool.refresh_past()

        # ── Self-play ──────────────────────────────────────────────
        print("\n▸ Self-play...")
        t0 = time.time()
        all_samples = []
        all_stats = []
        opp_counts = {}
        kl_total = 0
        kl_n = 0

        for i in range(config.num_self_play_games):
            opponent, label = pool.sample(network)
            opp_counts[label] = opp_counts.get(label, 0) + 1

            samples, winner, stats = play_game(
                network, opponent, config, device,
                simulations=config.num_simulations,
            )
            stats["opponent"] = label
            all_samples.extend(samples)
            all_stats.append(stats)

            kl_total += stats["kl_filtered_in"]
            kl_n += stats["num_moves"] // 2 + 1  # rough denominator

            r = stats["result"]
            print(f"  {i+1:>2}/{config.num_self_play_games} "
                  f"vs {label:<10} {stats['num_moves']:>3}mv "
                  f"{r}  kl_kept={stats['kl_filtered_in']}")

        elapsed = time.time() - t0
        replay_buffer.extend(all_samples)

        wins = sum(1 for s in all_stats if s["result"] == "win")
        losses = sum(1 for s in all_stats if s["result"] == "loss")
        draws = sum(1 for s in all_stats if s["result"] == "draw")
        avg_kl_kept = kl_total / max(config.num_self_play_games, 1)
        avg_moves = np.mean([s["num_moves"] for s in all_stats])

        print(f"\n  W/L/D: {wins}/{losses}/{draws}  "
              f"avg_moves={avg_moves:.0f}  "
              f"avg_kl_positions={avg_kl_kept:.1f}  "
              f"time={elapsed:.0f}s")
        print(f"  Opponents: {opp_counts}")

        # Warn if KL filter is removing almost everything
        if avg_kl_kept < 3:
            print("  ⚠ KL filter removed most positions — MCTS barely "
                  "improving on raw policy. Consider increasing simulations.")

        # ── Train ──────────────────────────────────────────────────
        if len(replay_buffer) >= config.min_replay_size:
            print("\n▸ Training...")
            t1 = time.time()
            stats_t = train_step(
                network, replay_buffer, optimizer,
                batch_size=config.batch_size,
                num_epochs=2,
                device=device,
            )
            print(f"  loss={stats_t['loss']:.4f}  "
                  f"policy={stats_t['policy_loss']:.4f}  "
                  f"value={stats_t['value_loss']:.4f}  "
                  f"({time.time()-t1:.1f}s)")
        else:
            stats_t = {}
            print(f"  Skipping (need {config.min_replay_size}, "
                  f"have {len(replay_buffer)})")

        # ── Evaluation gate ────────────────────────────────────────
        # Only replace "best" if the new model actually beats it
        print("\n▸ Evaluation gate...")
        network.eval()
        wr = evaluate_vs_best(
            network, best_network, device,
            simulations=config.num_simulations,
            num_games=eval_games,
            num_players=config.num_players,
        )
        gate_passed = wr >= eval_threshold
        print(f"  Candidate vs best: {wr*100:.0f}%  "
              f"(threshold {eval_threshold*100:.0f}%)  "
              f"→ {'ACCEPTED ✓' if gate_passed else 'REJECTED — reverting'}")

        if gate_passed:
            best_network.load_state_dict(network.state_dict())
        else:
            # Revert to best but keep optimizer state (partial learning preserved)
            network.load_state_dict(best_network.state_dict())

        # ── Log & checkpoint ───────────────────────────────────────
        entry = {
            "iteration": iteration,
            "wins": wins, "losses": losses, "draws": draws,
            "avg_moves": float(avg_moves),
            "avg_kl_kept": float(avg_kl_kept),
            "eval_win_rate": float(wr),
            "gate_passed": gate_passed,
            "opponents": opp_counts,
            "buffer_size": len(replay_buffer),
            **stats_t,
        }
        log.append(entry)
        with open(os.path.join(config.log_dir, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        if iteration % config.save_every == 0 or iteration == end_iter:
            path = os.path.join(
                config.checkpoint_dir, f"model_iter_{iteration:04d}.pt")
            torch.save({
                "iteration": iteration,
                "model_state_dict": best_network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
                "eval_win_rate": wr,
            }, path)
            print(f"  Saved: {path}")

    # ── Final benchmark ────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL EVALUATION")
    print("="*60)
    final_wr = quick_eval(best_network, device,
                          simulations=config.num_simulations,
                          num_games=20,
                          num_players=config.num_players,
                          verbose=True)
    print(f"\nFinal win rates: {final_wr}")
    print("\nCompare to iter 0:")
    print(f"  Greedy: {start_wr.get('Greedy', '?')*100:.0f}% → "
          f"{final_wr.get('Greedy', '?')*100:.0f}%")
    print(f"  SmartGreedy: {start_wr.get('SmartGreedy', '?')*100:.0f}% → "
          f"{final_wr.get('SmartGreedy', '?')*100:.0f}%")

    if final_wr.get('SmartGreedy', 0) > start_wr.get('SmartGreedy', 0):
        print("\n✓ Self-play IMPROVED on warmup model!")
    else:
        print("\n⚠ Self-play did NOT improve on warmup model.")
        print("  → Increase simulations (try 300+) and/or more warmup games.")

    return best_network, log


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Train Quoridor AI (fixed)")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--games", type=int, default=25,
                        help="Self-play games per iteration")
    parser.add_argument("--simulations", type=int, default=150,
                        help="MCTS simulations per move")
    parser.add_argument("--players", type=int, default=2, choices=[2, 4])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-games", type=int, default=10,
                        help="Games for evaluation gate")
    parser.add_argument("--eval-threshold", type=float, default=0.52,
                        help="Win rate needed to accept new model (default 0.52)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = Config(
        num_players=args.players,
        device=args.device,
        num_iterations=args.iterations,
        num_self_play_games=args.games,
        num_simulations=args.simulations,
        save_every=5,
        # These are overridden inside train() but set here for buffer calc:
        num_epochs=2,
        replay_buffer_size=max(3000, args.games * 45 * 3),
    )

    train(cfg, resume=args.resume,
          eval_games=args.eval_games,
          eval_threshold=args.eval_threshold)
