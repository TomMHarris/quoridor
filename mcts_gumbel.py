"""
Gumbel AlphaZero MCTS.

Based on: "Policy improvement by planning with Gumbel" (Danihelka et al., 2022)

Key insight: Standard MCTS needs hundreds of simulations to get a good policy
because it uses visit counts as a proxy for action quality. Gumbel MCTS uses
the Gumbel-Top-k trick + Sequential Halving to efficiently allocate simulations
to the most promising actions, getting comparable policy quality from 16-50 sims.

This means ~4-10x more training games per hour with the same compute budget.
"""

import math
import numpy as np
from typing import Optional, Dict

from game import QuoridorGame, MoveEncoder, Move


def _boost_pawn_priors(policy, game, min_pawn_share=0.35):
    """Ensure pawn moves get at least min_pawn_share of probability mass."""
    pawn_indices = []
    wall_indices = []
    pawn_pos = game.pawns[game.current_player]
    for move in game.get_legal_pawn_moves():
        idx = MoveEncoder.encode(move, pawn_pos)
        pawn_indices.append(idx)
    for idx in range(MoveEncoder.TOTAL_ACTIONS):
        if idx < 128 and policy[idx] > 0 and idx not in pawn_indices:
            wall_indices.append(idx)
    if not pawn_indices:
        return policy
    pawn_total = sum(policy[i] for i in pawn_indices)
    if pawn_total >= min_pawn_share:
        return policy
    boosted = policy.copy()
    wall_total = sum(policy[i] for i in wall_indices)
    if wall_total > 0:
        wall_scale = (1.0 - min_pawn_share) / wall_total
        for i in wall_indices:
            boosted[i] *= wall_scale
    if pawn_total > 0:
        pawn_scale = min_pawn_share / pawn_total
        for i in pawn_indices:
            boosted[i] *= pawn_scale
    else:
        for i in pawn_indices:
            boosted[i] = min_pawn_share / len(pawn_indices)
    return boosted


class MCTSNode:
    """A node in the search tree."""
    __slots__ = [
        "game", "parent", "move", "children",
        "visit_count", "value_sum", "prior",
    ]

    def __init__(self, game, parent=None, move=None, prior=0.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.children: Dict[Move, "MCTSNode"] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        return self.game.is_over


class GumbelMCTS:
    """
    Gumbel AlphaZero MCTS.

    Key differences from standard MCTS:
    1. At root: Gumbel noise + Sequential Halving for action selection
    2. Non-root: standard PUCT (same as AlphaZero)
    3. Policy target: uses completed Q-values, not raw visit counts
    4. Works well with 16-50 simulations (vs 200+ for standard)

    Usage:
        mcts = GumbelMCTS(network, config)
        action_probs = mcts.search(game_state)
    """

    def __init__(self, network, device: str = "cpu",
                 num_simulations: int = 50,
                 c_puct: float = 1.5,
                 max_num_considered_actions: int = 16,
                 c_visit: float = 50.0,
                 c_scale: float = 1.0):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        # Gumbel-specific params
        self.max_num_considered = max_num_considered_actions
        self.c_visit = c_visit   # for mixing prior with Q
        self.c_scale = c_scale

    def search(self, game: QuoridorGame, add_noise: bool = True) -> np.ndarray:
        """
        Run Gumbel MCTS from the given game state.

        Returns:
            action_probs: (140,) improved policy (from completed Q-values).
        """
        root = MCTSNode(game.clone())

        # Get network evaluation at root
        state = game.to_tensor()
        legal_mask = MoveEncoder.legal_mask(game)
        raw_policy, root_value = self.network.predict(state, legal_mask, self.device)

        # Boost pawn priors (same cold-start fix as standard MCTS)
        raw_policy = _boost_pawn_priors(raw_policy, game)

        # Expand root
        probable_moves = game.get_probable_moves()
        pawn_pos = game.pawns[game.current_player]

        # Build action info
        move_to_idx = {}
        for move in probable_moves:
            idx = MoveEncoder.encode(move, pawn_pos)
            child_game = game.clone()
            child_game.make_move(move)
            child = MCTSNode(
                game=child_game, parent=root, move=move,
                prior=raw_policy[idx],
            )
            root.children[move] = child
            move_to_idx[move] = idx

        if not root.children:
            return np.zeros(MoveEncoder.TOTAL_ACTIONS, dtype=np.float32)

        # ── Gumbel sampling at root ──
        moves = list(root.children.keys())
        n_actions = len(moves)

        # Get log-priors (logits) for Gumbel trick
        logits = np.array([
            np.log(max(raw_policy[move_to_idx[m]], 1e-8)) for m in moves
        ], dtype=np.float64)

        if add_noise:
            # Sample Gumbel(0,1) noise
            gumbels = np.random.gumbel(size=n_actions)
        else:
            gumbels = np.zeros(n_actions)

        # Perturbed logits
        g_logits = logits + gumbels

        # ── Sequential Halving ──
        # Determine how many actions to consider and simulation budget per phase
        m = min(self.max_num_considered, n_actions)
        n_sims = self.num_simulations

        # Select top-m actions by perturbed logits
        top_indices = np.argsort(g_logits)[-m:]
        active_set = set(top_indices.tolist())

        # Allocate simulations via sequential halving
        if m <= 1:
            phases = [(list(active_set), n_sims)]
        else:
            phases = []
            remaining_sims = n_sims
            current_m = m
            while current_m > 1 and remaining_sims > 0:
                sims_per_action = max(1, remaining_sims // (current_m * int(np.ceil(np.log2(m)))))
                sims_this_phase = sims_per_action * current_m
                phases.append((list(active_set), sims_per_action))
                remaining_sims -= sims_this_phase

                # Compute scores and halve
                scores = []
                for idx in active_set:
                    child = root.children[moves[idx]]
                    q = self._completed_q(child, logits[idx])
                    scores.append((g_logits[idx] + q, idx))
                scores.sort(reverse=True)

                current_m = max(1, current_m // 2)
                active_set = set(s[1] for s in scores[:current_m])

            # Spend remaining sims on survivors
            if remaining_sims > 0 and active_set:
                sims_per = remaining_sims // max(len(active_set), 1)
                if sims_per > 0:
                    phases.append((list(active_set), sims_per))

        # ── Execute simulations ──
        for action_indices, sims_per in phases:
            for aidx in action_indices:
                move = moves[aidx]
                child = root.children[move]
                for _ in range(sims_per):
                    self._simulate(child)

        # ── Build improved policy from completed Q-values ──
        action_probs = np.zeros(MoveEncoder.TOTAL_ACTIONS, dtype=np.float32)

        completed_qs = np.full(n_actions, -np.inf, dtype=np.float64)
        for i, move in enumerate(moves):
            child = root.children[move]
            completed_qs[i] = logits[i] + self._completed_q(child, logits[i])

        # Softmax over completed Q-values
        completed_qs -= completed_qs.max()
        exp_qs = np.exp(completed_qs)
        total = exp_qs.sum()
        if total > 0:
            probs = exp_qs / total
        else:
            probs = np.ones(n_actions) / n_actions

        for i, move in enumerate(moves):
            idx = move_to_idx[move]
            action_probs[idx] = probs[i]

        return action_probs

    def _completed_q(self, child: MCTSNode, logit: float) -> float:
        """
        Completed Q-value: mix of prior value and search value.
        When visit_count is 0, uses the prior (network value).
        As visits increase, converges to the empirical Q.
        """
        if child.visit_count == 0:
            return 0.0  # no information beyond the prior
        return self.c_scale * child.q_value

    def _simulate(self, node: MCTSNode) -> None:
        """Run one simulation from this node downward."""
        # Walk to a leaf using PUCT
        path = [node]
        current = node

        while not current.is_leaf() and not current.is_terminal():
            current = self._select_child_puct(current)
            path.append(current)

        # Evaluate
        if current.is_terminal():
            value = 1.0 if current.game.winner is not None else 0.0
        else:
            value = self._expand(current)

        # Backprop
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    def _select_child_puct(self, node: MCTSNode) -> MCTSNode:
        """Standard PUCT selection for non-root nodes."""
        sqrt_parent = math.sqrt(max(node.visit_count, 1))
        best_score = -float("inf")
        best_child = None

        for child in node.children.values():
            q = child.q_value
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode) -> float:
        """Expand a leaf and return network value estimate."""
        game = node.game
        if game.is_over:
            return 0.0

        state = game.to_tensor()
        legal_mask = MoveEncoder.legal_mask(game)
        policy, value = self.network.predict(state, legal_mask, self.device)
        policy = _boost_pawn_priors(policy, game)

        probable_moves = game.get_probable_moves()
        pawn_pos = game.pawns[game.current_player]

        for move in probable_moves:
            idx = MoveEncoder.encode(move, pawn_pos)
            child_game = game.clone()
            child_game.make_move(move)
            child = MCTSNode(
                game=child_game, parent=node, move=move,
                prior=policy[idx],
            )
            node.children[move] = child

        return -value


def select_action(action_probs: np.ndarray, temperature: float = 1.0) -> int:
    """Choose an action index from the probability distribution."""
    if temperature < 1e-6:
        max_prob = action_probs.max()
        candidates = np.where(action_probs == max_prob)[0]
        return np.random.choice(candidates)

    probs = action_probs ** (1.0 / temperature)
    total = probs.sum()
    if total <= 0:
        nonzero = np.nonzero(action_probs)[0]
        return np.random.choice(nonzero) if len(nonzero) > 0 else 0
    probs /= total
    return np.random.choice(len(probs), p=probs)
