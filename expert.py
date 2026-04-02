"""
Heuristic expert agents for generating pre-training data.

Agents (weakest to strongest):
    PathFollower  - Always follows shortest path, never places walls.
    HeuristicExpert - Follows shortest path, places walls to block opponent.
    MCTSExpert - Pure MCTS with heuristic rollouts (no neural network).

Also provides generate_expert_games() for bulk data generation.
"""

import random
import math
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
from multiprocessing import Pool

from engine import QuoridorGame, MoveEncoder, Move, _USING_FAST_ENGINE, _PythonQuoridorGame
from config import Config


# ---------------------------------------------------------------------------
# PathFollower (baseline)
# ---------------------------------------------------------------------------

class PathFollower:
    """Always moves pawn along shortest path. Never places walls."""

    def choose_move(self, game: QuoridorGame) -> Move:
        path = game.shortest_path(game.current_player)
        legal_pawn = game.get_legal_pawn_moves()
        if path and len(path) > 1:
            candidate = ("move", path[1])
            if candidate in legal_pawn:
                return candidate
        return random.choice(legal_pawn) if legal_pawn else game.get_legal_moves()[0]


# ---------------------------------------------------------------------------
# HeuristicExpert
# ---------------------------------------------------------------------------

class HeuristicExpert:
    """Strategic agent: follows shortest path + places blocking walls.

    Wall placement strategy:
        1. Compute opponent's shortest path
        2. Try walls that cross the opponent's path
        3. Pick the wall that maximises (opponent_path_increase - my_path_increase)
        4. Only place wall if net gain >= 1 step
    """

    def __init__(self, wall_prob: float = 0.35):
        self.wall_prob = wall_prob

    def choose_move(self, game: QuoridorGame) -> Move:
        cp = game.current_player
        opp = 1 - cp

        # If no walls left, just move
        if game.walls_remaining[cp] <= 0:
            return self._best_pawn_move(game)

        my_dist = game.shortest_path_length(cp)
        opp_dist = game.shortest_path_length(opp)

        # If we're very close to goal, just run
        if my_dist is not None and my_dist <= 2:
            return self._best_pawn_move(game)

        # If opponent is close or we have distance advantage, consider walls
        should_wall = (
            random.random() < self.wall_prob
            and opp_dist is not None
            and (opp_dist <= my_dist + 2 if my_dist else True)
        )

        if should_wall:
            wall = self._find_best_wall(game, cp, opp)
            if wall is not None:
                return wall

        return self._best_pawn_move(game)

    def _best_pawn_move(self, game: QuoridorGame) -> Move:
        """Move that minimises distance to goal."""
        cp = game.current_player
        best_move = None
        best_dist = float("inf")
        for move in game.get_legal_pawn_moves():
            g = game.clone()
            g.make_move(move)
            d = g.shortest_path_length(cp)
            if d is not None and d < best_dist:
                best_dist = d
                best_move = move
        return best_move or game.get_legal_pawn_moves()[0]

    def _find_best_wall(self, game: QuoridorGame, cp: int, opp: int) -> Optional[Move]:
        """Find wall that maximises net path-length disruption."""
        opp_path = game.shortest_path(opp)
        if not opp_path or len(opp_path) < 2:
            return None

        my_dist_before = game.shortest_path_length(cp) or 0
        opp_dist_before = game.shortest_path_length(opp) or 0

        # Candidate walls: near opponent's path
        candidates = set()
        for r, c in opp_path:
            for dr in range(-1, 1):
                for dc in range(-1, 1):
                    wr, wc = r + dr, c + dc
                    if 0 <= wr < 8 and 0 <= wc < 8:
                        candidates.add((wr, wc))

        best_wall = None
        best_gain = 0

        for wr, wc in candidates:
            for o in ("H", "V"):
                if not game.is_valid_wall(wr, wc, o):
                    continue
                # Clone and place wall to evaluate
                g = game.clone()
                g.make_move(("wall", (wr, wc, o)))

                opp_dist_after = g.shortest_path_length(opp)
                my_dist_after = g.shortest_path_length(cp)

                if opp_dist_after is None or my_dist_after is None:
                    continue

                gain = (opp_dist_after - opp_dist_before) - (my_dist_after - my_dist_before)
                if gain > best_gain:
                    best_gain = gain
                    best_wall = ("wall", (wr, wc, o))

        return best_wall


# ---------------------------------------------------------------------------
# MCTSExpert (strongest, no neural network)
# ---------------------------------------------------------------------------

class MCTSExpert:
    """Pure MCTS with heuristic rollouts. Strongest heuristic agent."""

    def __init__(self, num_rollouts: int = 300):
        self.num_rollouts = num_rollouts

    def choose_move(self, game: QuoridorGame) -> Move:
        moves = game.get_probable_moves()
        if len(moves) == 1:
            return moves[0]

        scores = np.zeros(len(moves))
        counts = np.zeros(len(moves))

        for _ in range(self.num_rollouts):
            # UCB1 selection
            idx = self._select_ucb(scores, counts)
            move = moves[idx]

            # Simulate
            g = _fast_clone_for_expert(game)
            g.make_move(move)
            value = self._heuristic_rollout(g, game.current_player)

            scores[idx] += value
            counts[idx] += 1

        # Pick most-visited
        return moves[int(np.argmax(counts))]

    def _select_ucb(self, scores: np.ndarray, counts: np.ndarray) -> int:
        total = counts.sum()
        if total < len(counts):
            # Visit each at least once
            for i in range(len(counts)):
                if counts[i] == 0:
                    return i

        ucb = scores / np.maximum(counts, 1) + 1.4 * np.sqrt(
            np.log(total + 1) / np.maximum(counts, 1)
        )
        return int(np.argmax(ucb))

    def _heuristic_rollout(self, game: QuoridorGame, perspective: int,
                           max_depth: int = 60) -> float:
        """Play out game with heuristic policy, return value for perspective player."""
        g = game
        for _ in range(max_depth):
            if g.is_over:
                return 1.0 if g.winner == perspective else -1.0

            # Rollout policy: 70% shortest path, 30% random probable
            if random.random() < 0.7:
                path = g.shortest_path(g.current_player)
                legal_pawn = g.get_legal_pawn_moves()
                if path and len(path) > 1:
                    candidate = ("move", path[1])
                    # BFS path ignores occupancy — validate move is legal
                    if candidate in legal_pawn:
                        move = candidate
                    else:
                        move = random.choice(legal_pawn) if legal_pawn else g.get_legal_moves()[0]
                else:
                    move = random.choice(legal_pawn) if legal_pawn else g.get_legal_moves()[0]
            else:
                moves = g.get_probable_moves()
                move = random.choice(moves) if moves else g.get_legal_moves()[0]

            g = _fast_clone_for_expert(g)
            g.make_move(move)

        # Game didn't finish: use distance heuristic
        d_persp = g.shortest_path_length(perspective) or 20
        d_opp = g.shortest_path_length(1 - perspective) or 20
        return float(np.tanh((d_opp - d_persp) / 4.0))


if _USING_FAST_ENGINE:
    def _fast_clone_for_expert(game):
        return game.clone()
else:
    def _fast_clone_for_expert(game):
        """Minimal clone for rollouts (no move_history to save memory)."""
        g = _PythonQuoridorGame.__new__(_PythonQuoridorGame)
        g.num_players = game.num_players
        g.pawns = list(game.pawns)
        g.goals = game.goals
        g.walls_remaining = list(game.walls_remaining)
        g.current_player = game.current_player
        g.winner = game.winner
        g.walls_placed = set(game.walls_placed)
        g.wall_centers = set(game.wall_centers)
        g.h_blocked = set(game.h_blocked)
        g.v_blocked = set(game.v_blocked)
        g.move_history = []
        return g


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _play_expert_game(args) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Play one expert game and return training samples."""
    game_idx, use_mcts, mcts_rollouts = args
    random.seed(game_idx * 1000 + random.randint(0, 999))

    if use_mcts:
        agents = [MCTSExpert(mcts_rollouts), MCTSExpert(mcts_rollouts)]
    else:
        agents = [HeuristicExpert(), HeuristicExpert()]

    game = QuoridorGame(2)
    history = []  # (state_tensor, action_idx, player, legal_mask)

    for step in range(200):
        if game.is_over:
            break

        cp = game.current_player
        state = game.to_tensor()
        legal_mask = MoveEncoder.legal_mask(game)
        move = agents[cp].choose_move(game)
        pawn_pos = game.pawns[cp]
        action_idx = MoveEncoder.encode(move, pawn_pos)

        history.append((state, action_idx, cp, legal_mask))
        game.make_move(move)

    # Build training samples
    samples = []
    winner = game.winner

    for state, action_idx, player, legal_mask in history:
        # Policy target: smoothed over LEGAL moves only
        n_legal = max(legal_mask.sum(), 1)
        policy = np.zeros(MoveEncoder.TOTAL_ACTIONS, dtype=np.float32)
        # Spread 15% across all legal moves
        smoothing = 0.15 / n_legal
        policy[legal_mask > 0] = smoothing
        # Expert's chosen action gets the remaining 85% + its share of smoothing
        policy[action_idx] = 0.85 + smoothing
        policy /= policy.sum()

        # Value target
        if winner is not None:
            value = 1.0 if winner == player else -1.0
        else:
            # Unfinished game: use distance heuristic
            d0 = game.shortest_path_length(player) or 20
            d1 = game.shortest_path_length(1 - player) or 20
            value = float(np.tanh((d1 - d0) / 4.0))

        samples.append((state, policy, value, legal_mask))

    return samples


def generate_expert_games(config: Config, verbose: bool = True
                          ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Generate training data from expert games using multiprocessing."""
    n_mcts = int(config.warmup_games * config.warmup_mcts_fraction)
    n_heuristic = config.warmup_games - n_mcts

    args_list = []
    for i in range(n_mcts):
        args_list.append((i, True, config.warmup_mcts_rollouts))
    for i in range(n_heuristic):
        args_list.append((n_mcts + i, False, 0))

    random.shuffle(args_list)

    all_samples = []

    if config.num_workers > 1:
        with Pool(config.num_workers) as pool:
            for i, samples in enumerate(pool.imap_unordered(_play_expert_game, args_list)):
                all_samples.extend(samples)
                if verbose and (i + 1) % 50 == 0:
                    print(f"  Expert games: {i+1}/{config.warmup_games} "
                          f"({len(all_samples)} samples)")
    else:
        for i, args in enumerate(args_list):
            samples = _play_expert_game(args)
            all_samples.extend(samples)
            if verbose and (i + 1) % 50 == 0:
                print(f"  Expert games: {i+1}/{config.warmup_games} "
                      f"({len(all_samples)} samples)")

    if verbose:
        print(f"  Total: {config.warmup_games} games, {len(all_samples)} samples")

    return all_samples


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing expert agents...\n")

    game = QuoridorGame(2)

    # PathFollower
    pf = PathFollower()
    move = pf.choose_move(game)
    print(f"PathFollower: {move}")

    # HeuristicExpert
    he = HeuristicExpert()
    move = he.choose_move(game)
    print(f"HeuristicExpert: {move}")

    # MCTSExpert (small rollouts for speed)
    me = MCTSExpert(num_rollouts=20)
    move = me.choose_move(game)
    print(f"MCTSExpert (20 rollouts): {move}")

    # Play a quick game
    print("\nPlaying PathFollower vs HeuristicExpert...")
    game = QuoridorGame(2)
    agents = [PathFollower(), HeuristicExpert()]
    for step in range(200):
        if game.is_over:
            break
        move = agents[game.current_player].choose_move(game)
        game.make_move(move)
    print(f"  Result: {'P' + str(game.winner) + ' wins' if game.winner is not None else 'Draw'}")
    print(f"  Moves: {len(game.move_history)}")

    # Generate a few training samples
    print("\nGenerating 5 expert games (heuristic only)...")
    cfg = Config()
    cfg.warmup_games = 5
    cfg.warmup_mcts_fraction = 0.0
    cfg.num_workers = 1
    samples = generate_expert_games(cfg, verbose=False)
    print(f"  Got {len(samples)} samples")
    s, p, v = samples[0]
    print(f"  State shape: {s.shape}, Policy shape: {p.shape}, Value: {v:.2f}")
    print(f"  Policy sum: {p.sum():.4f}")
