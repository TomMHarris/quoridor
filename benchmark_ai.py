"""
Comprehensive Quoridor AI Benchmark
====================================

Modeled on gorisanson's evaluation methodology:
  - Self-play at multiple simulation counts (strength ladder)
  - Cross-checkpoint head-to-head (measuring training progress)
  - External baseline opponents
  - 50/50 split as P0/P1 for fairness
  - Wilson score confidence intervals (95%)
  - Clean tabular output

Usage:
    python benchmark_ai.py                        # full benchmark suite
    python benchmark_ai.py --quick                # fast version (10 games each)
    python benchmark_ai.py --ladder               # strength ladder only
    python benchmark_ai.py --progression          # checkpoint progression only
    python benchmark_ai.py --baselines            # baselines only
    python benchmark_ai.py --games 100            # 100 games per matchup
"""

import argparse
import glob
import json
import math
import os
import random
import time
from typing import List, Tuple, Optional

import numpy as np
import torch

from game import QuoridorGame, MoveEncoder
from network import QuoridorNet, count_parameters
from mcts import MCTS, select_action
from config import Config


# =====================================================================
# Statistics
# =====================================================================

def wilson_ci(wins: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return 0.0, 1.0
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0, center - spread), min(1, center + spread)


# =====================================================================
# Baseline agents
# =====================================================================

class RandomAgent:
    """Plays a uniformly random legal move."""
    name = "Random"

    def choose_move(self, game: QuoridorGame):
        return random.choice(game.get_legal_moves())


class GreedyAgent:
    """Always moves pawn along shortest path. Never places walls."""
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
    """
    Follows shortest path but also strategically places walls
    to block opponent's path with probability wall_prob.
    Only places a wall if it strictly increases opponent's distance.
    """
    name = "Smart Greedy"

    def __init__(self, wall_prob: float = 0.35):
        self.wall_prob = wall_prob

    def choose_move(self, game: QuoridorGame):
        player = game.current_player
        opponent = 1 - player

        # Sometimes try to block opponent
        if (random.random() < self.wall_prob
                and game.walls_remaining[player] > 0):
            opp_path = game.shortest_path(opponent)
            opp_dist = game.shortest_path_length(opponent)
            if opp_path and len(opp_path) > 1 and opp_dist is not None:
                # Try walls near opponent's next few steps
                best_wall = None
                best_increase = 0

                targets = opp_path[1:4]  # next 3 cells on their path
                for tr, tc in targets:
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            wr, wc = tr + dr, tc + dc
                            for o in ("H", "V"):
                                if game.is_valid_wall(wr, wc, o):
                                    g2 = game.clone()
                                    g2.make_move(("wall", (wr, wc, o)))
                                    new_dist = g2.shortest_path_length(opponent)
                                    if new_dist is not None:
                                        increase = new_dist - opp_dist
                                        if increase > best_increase:
                                            best_increase = increase
                                            best_wall = ("wall", (wr, wc, o))

                if best_wall and best_increase >= 2:
                    return best_wall

        # Default: follow shortest path
        path = game.shortest_path(player)
        if path and len(path) > 1:
            move = ("move", path[1])
            if move in game.get_legal_pawn_moves():
                return move
        pawn_moves = game.get_legal_pawn_moves()
        if pawn_moves:
            return random.choice(pawn_moves)
        return random.choice(game.get_legal_moves())


class PureMCTSAgent:
    """
    Pure MCTS with random rollouts (no neural network).
    Closer to gorisanson's approach but without their heuristics.
    """

    def __init__(self, num_rollouts: int = 500):
        self.num_rollouts = num_rollouts
        self.name = f"Pure MCTS ({num_rollouts})"

    def choose_move(self, game: QuoridorGame):
        # Consider pawn moves + probable walls
        pawn_moves = game.get_legal_pawn_moves()
        try:
            walls = game.get_probable_walls()
        except AttributeError:
            walls = game.get_legal_walls()
        candidates = pawn_moves + walls[:15]

        if not candidates:
            return random.choice(game.get_legal_moves())

        rollouts_per = max(1, self.num_rollouts // len(candidates))
        best_move = None
        best_score = -1

        for move in candidates:
            score = 0
            for _ in range(rollouts_per):
                g = game.clone()
                g.make_move(move)
                result = self._rollout(g, game.current_player)
                score += result
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _rollout(self, game: QuoridorGame, perspective: int,
                 max_depth: int = 100) -> float:
        depth = 0
        while not game.is_over and depth < max_depth:
            moves = game.get_legal_moves()
            pawn_moves = [m for m in moves if m[0] == "move"]
            if random.random() < 0.7 and pawn_moves:
                game.make_move(random.choice(pawn_moves))
            else:
                game.make_move(random.choice(moves))
            depth += 1
        if game.winner == perspective:
            return 1.0
        elif game.winner is not None:
            return 0.0
        return 0.5


class NNAgent:
    """Neural network + MCTS agent (our trained model)."""

    def __init__(self, network, device, num_simulations=100, name="AI"):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.name = name
        self.mcts = MCTS(
            network=network, device=device,
            num_simulations=num_simulations,
            c_puct=1.5,
        )

    def choose_move(self, game: QuoridorGame):
        action_probs = self.mcts.search(game, add_noise=False)
        action_idx = select_action(action_probs, temperature=0.1)
        pawn_pos = game.pawns[game.current_player]
        return MoveEncoder.decode(action_idx, pawn_pos)


# =====================================================================
# Match runner (gorisanson-style: 50/50 P0/P1 split)
# =====================================================================

def play_match(agent_a, agent_b, num_games: int = 100,
               max_moves: int = 120, verbose: bool = True) -> dict:
    """
    Play a match between agent_a and agent_b.
    Half games with agent_a as P0, half as P1.
    Returns results from agent_a's perspective.
    """
    games_per_side = num_games // 2
    wins_as_p0 = 0
    wins_as_p1 = 0
    losses = 0
    draws = 0
    game_lengths = []
    total_time = 0

    for half in range(2):
        # half 0: agent_a is P0, half 1: agent_a is P1
        a_is_p0 = (half == 0)

        for i in range(games_per_side):
            if a_is_p0:
                agents = [agent_a, agent_b]
            else:
                agents = [agent_b, agent_a]

            game = QuoridorGame(2)
            t0 = time.time()
            move_count = 0

            while not game.is_over and move_count < max_moves:
                agent = agents[game.current_player]
                move = agent.choose_move(game)
                game.make_move(move)
                move_count += 1

            elapsed = time.time() - t0
            total_time += elapsed
            game_lengths.append(move_count)

            # Determine outcome from agent_a's perspective
            if game.winner is None:
                draws += 1
                outcome_str = "draw"
            else:
                a_player = 0 if a_is_p0 else 1
                if game.winner == a_player:
                    if a_is_p0:
                        wins_as_p0 += 1
                    else:
                        wins_as_p1 += 1
                    outcome_str = "win"
                else:
                    losses += 1
                    outcome_str = "loss"

            game_num = i + 1 + (games_per_side if half == 1 else 0)
            side = "P0" if a_is_p0 else "P1"
            if verbose:
                print(f"    Game {game_num:>3}/{num_games} "
                      f"({agent_a.name} as {side}): "
                      f"{move_count:>3} moves, {outcome_str}, "
                      f"{elapsed:.1f}s")

    total_wins = wins_as_p0 + wins_as_p1
    total = total_wins + losses + draws
    win_rate = total_wins / total if total > 0 else 0
    ci_low, ci_high = wilson_ci(total_wins, total)

    return {
        "agent_a": agent_a.name,
        "agent_b": agent_b.name,
        "num_games": total,
        "wins_as_p0": wins_as_p0,
        "wins_as_p1": wins_as_p1,
        "total_wins": total_wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "avg_game_length": np.mean(game_lengths),
        "total_time": total_time,
    }


# =====================================================================
# Result formatting (gorisanson-style tables)
# =====================================================================

def print_table(title: str, reference_agent: str, results: list):
    """Print results in gorisanson's table format."""
    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"{'='*95}")
    print(f"  {'Opponent':<22} {'Games':>6} {'(as P0/P1)':>12} "
          f"{'Wins P0':>8} {'Wins P1':>8} {'Win%':>6} "
          f"{'CI Low':>7} {'CI High':>8}")
    print(f"  {'-'*22} {'-'*6} {'-'*12} {'-'*8} {'-'*8} {'-'*6} {'-'*7} {'-'*8}")

    for r in results:
        games_per_side = r["num_games"] // 2
        print(f"  {r['agent_b']:<22} {r['num_games']:>5}  "
              f"({games_per_side}/{games_per_side})      "
              f"{r['wins_as_p0']:>5}    {r['wins_as_p1']:>5}   "
              f"{r['win_rate']*100:5.1f}% "
              f"{r['ci_low']*100:6.1f}%  {r['ci_high']*100:6.1f}%")

    print()


def print_progression_table(title: str, results: list):
    """Print learning progression table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Checkpoint':<20} {'Win%':>6} {'95% CI':>18} "
          f"{'Avg Moves':>10} {'W':>4}/{' L':>3}/{' D':>3}")
    print(f"  {'-'*20} {'-'*6} {'-'*18} {'-'*10} {'-'*13}")

    for r in results:
        print(f"  {r['name']:<20} {r['win_rate']*100:5.1f}% "
              f"  [{r['ci_low']*100:5.1f}%, {r['ci_high']*100:5.1f}%]"
              f"  {r['avg_game_length']:>8.0f}  "
              f"  {r['total_wins']:>3}/ {r['losses']:>3}/ {r['draws']:>3}")
    print()


# =====================================================================
# Checkpoint loading
# =====================================================================

def load_checkpoint(path: str, device: str):
    data = torch.load(path, map_location=device, weights_only=False)
    cfg_data = data["config"]
    cfg = Config(**{k: v for k, v in cfg_data.items()
                    if k in Config.__dataclass_fields__})
    net = QuoridorNet(
        input_planes=cfg.input_planes,
        num_blocks=cfg.num_res_blocks,
        num_channels=cfg.num_channels,
    ).to(device)
    net.load_state_dict(data["model_state_dict"])
    net.eval()
    return net, cfg, data.get("iteration", "?")


def find_checkpoints(checkpoint_dir="checkpoints"):
    return sorted(glob.glob(os.path.join(checkpoint_dir, "model_iter_*.pt")))


# =====================================================================
# Benchmark modes
# =====================================================================

def strength_ladder(network, device: str, num_games: int = 20,
                    verbose: bool = True):
    """
    Self-play at different simulation counts.
    Like gorisanson testing 2.5k vs 7.5k vs 20k vs 60k.
    Our equivalent: vary the MCTS simulation budget.
    """
    print("\n" + "#" * 70)
    print("  STRENGTH LADDER: Same network, varying MCTS simulations")
    print("#" * 70)

    sim_levels = [
        (25,  "25-sim"),
        (50,  "50-sim"),
        (100, "100-sim"),
        (200, "200-sim"),
    ]

    # Reference agent: highest simulation count
    ref_sims, ref_name = sim_levels[-1]
    ref_agent = NNAgent(network, device, num_simulations=ref_sims,
                        name=ref_name)

    results = []
    for sims, name in sim_levels[:-1]:
        opponent = NNAgent(network, device, num_simulations=sims, name=name)
        print(f"\n  --- {ref_name} vs {name} ---")
        r = play_match(ref_agent, opponent, num_games=num_games,
                       verbose=verbose)
        results.append(r)

    print_table(
        f"Strength Ladder: {ref_name} (reference) vs weaker versions",
        ref_name, results
    )
    return results


def baseline_benchmark(network, device: str, num_games: int = 20,
                       simulations: int = 100, verbose: bool = True):
    """
    Test against external baseline agents of increasing difficulty.
    """
    print("\n" + "#" * 70)
    print("  BASELINE BENCHMARK: AI vs external opponents")
    print("#" * 70)

    ai = NNAgent(network, device, num_simulations=simulations,
                 name=f"AI ({simulations}-sim)")

    baselines = [
        RandomAgent(),
        GreedyAgent(),
        SmartGreedyAgent(wall_prob=0.35),
    ]

    # Only include Pure MCTS if doing a thorough run (it's slow)
    if num_games <= 20:
        baselines.append(PureMCTSAgent(num_rollouts=200))

    results = []
    for baseline in baselines:
        print(f"\n  --- {ai.name} vs {baseline.name} ---")
        r = play_match(ai, baseline, num_games=num_games, verbose=verbose)
        results.append(r)

    print_table(f"Baseline Results: {ai.name}", ai.name, results)
    return results


def checkpoint_progression(device: str, num_games: int = 10,
                           simulations: int = 100,
                           verbose: bool = False):
    """
    Test each saved checkpoint against a fixed baseline to measure
    learning over training. Like gorisanson's v0.2 vs v0.3 comparison
    but across all iterations.
    """
    print("\n" + "#" * 70)
    print("  LEARNING PROGRESSION: each checkpoint vs Greedy")
    print("#" * 70)

    checkpoints = find_checkpoints()
    if not checkpoints:
        print("  No checkpoints found!")
        return []

    baseline = GreedyAgent()
    results = []

    for path in checkpoints:
        net, cfg, iteration = load_checkpoint(path, device)
        name = f"iter {iteration}"
        ai = NNAgent(net, device, num_simulations=simulations, name=name)

        print(f"\n  --- {name} vs {baseline.name} ---")
        r = play_match(ai, baseline, num_games=num_games, verbose=verbose)
        results.append({
            "name": name,
            "iteration": iteration,
            **r,
        })

    print_progression_table(
        f"Learning Progression vs {baseline.name} ({num_games} games each)",
        results
    )

    # Also do head-to-head between earliest and latest checkpoint
    if len(checkpoints) >= 2:
        print("\n  --- Head-to-head: earliest vs latest checkpoint ---")
        net_early, _, iter_early = load_checkpoint(checkpoints[0], device)
        net_late, _, iter_late = load_checkpoint(checkpoints[-1], device)

        early_agent = NNAgent(net_early, device, num_simulations=simulations,
                              name=f"iter {iter_early}")
        late_agent = NNAgent(net_late, device, num_simulations=simulations,
                             name=f"iter {iter_late}")

        r = play_match(late_agent, early_agent,
                       num_games=num_games, verbose=verbose)
        print_table(
            f"Head-to-Head: iter {iter_late} vs iter {iter_early}",
            late_agent.name, [r]
        )

    return results


# =====================================================================
# Full benchmark
# =====================================================================

def full_benchmark(num_games: int = 20, simulations: int = 100,
                   device: str = "auto", verbose: bool = True):
    """Run the complete benchmark suite."""
    if device == "auto":
        device = Config().resolve_device()

    checkpoints = find_checkpoints()
    if not checkpoints:
        print("No checkpoints found. Run training first.")
        return

    # Load latest model
    path = checkpoints[-1]
    print(f"Loading latest model: {path}")
    net, cfg, iteration = load_checkpoint(path, device)
    print(f"Iteration: {iteration}")
    print(f"Device: {device}")
    print(f"Network parameters: {count_parameters(net):,}")
    print(f"Games per matchup: {num_games} ({num_games//2} as P0, "
          f"{num_games//2} as P1)")

    all_results = {}

    # 1. Baseline benchmark
    all_results["baselines"] = baseline_benchmark(
        net, device, num_games=num_games,
        simulations=simulations, verbose=verbose)

    # 2. Strength ladder
    all_results["ladder"] = strength_ladder(
        net, device, num_games=num_games, verbose=verbose)

    # 3. Learning progression
    prog_games = max(10, num_games // 2)
    all_results["progression"] = checkpoint_progression(
        device, num_games=prog_games,
        simulations=simulations, verbose=False)

    # Save results
    os.makedirs("logs", exist_ok=True)
    save_path = "logs/benchmark_results.json"

    # Convert for JSON serialization
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = []
        for r in v:
            sr = {key: val for key, val in r.items()
                  if isinstance(val, (int, float, str, bool, type(None)))}
            serializable[k].append(sr)

    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("  FULL BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Model: iteration {iteration} ({count_parameters(net):,} params)")
    print(f"  MCTS simulations: {simulations}")
    print()

    if all_results["baselines"]:
        print("  Baselines:")
        for r in all_results["baselines"]:
            print(f"    vs {r['agent_b']:<22} "
                  f"{r['win_rate']*100:5.1f}% "
                  f"[{r['ci_low']*100:.1f}%, {r['ci_high']*100:.1f}%]")

    if all_results["ladder"]:
        print("\n  Strength ladder (200-sim reference):")
        for r in all_results["ladder"]:
            print(f"    vs {r['agent_b']:<22} "
                  f"{r['win_rate']*100:5.1f}% "
                  f"[{r['ci_low']*100:.1f}%, {r['ci_high']*100:.1f}%]")

    if all_results["progression"]:
        print(f"\n  Progression vs Greedy:")
        for r in all_results["progression"]:
            print(f"    {r['name']:<20} "
                  f"{r['win_rate']*100:5.1f}% "
                  f"[{r['ci_low']*100:.1f}%, {r['ci_high']*100:.1f}%]")

    print()


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Quoridor AI (gorisanson-style)")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per matchup (gorisanson used 100)")
    parser.add_argument("--simulations", type=int, default=100,
                        help="MCTS sims per move for AI agent")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run (10 games, less verbose)")
    parser.add_argument("--ladder", action="store_true",
                        help="Strength ladder only")
    parser.add_argument("--baselines", action="store_true",
                        help="Baseline benchmark only")
    parser.add_argument("--progression", action="store_true",
                        help="Checkpoint progression only")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-game output")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = Config().resolve_device()

    if args.quick:
        args.games = 10

    verbose = not args.quiet and not args.quick

    if args.ladder or args.baselines or args.progression:
        # Run specific mode
        checkpoints = find_checkpoints()
        if not checkpoints:
            print("No checkpoints found.")
            exit(1)

        net, cfg, iteration = load_checkpoint(checkpoints[-1], device)
        print(f"Model: iteration {iteration}, device: {device}\n")

        if args.ladder:
            strength_ladder(net, device, num_games=args.games,
                           verbose=verbose)
        if args.baselines:
            baseline_benchmark(net, device, num_games=args.games,
                             simulations=args.simulations,
                             verbose=verbose)
        if args.progression:
            checkpoint_progression(device, num_games=args.games,
                                  simulations=args.simulations,
                                  verbose=verbose)
    else:
        # Full suite
        full_benchmark(
            num_games=args.games,
            simulations=args.simulations,
            device=device,
            verbose=verbose,
        )
