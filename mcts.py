"""
Monte Carlo Tree Search with neural network guidance (AlphaZero-style).

Key differences from pure MCTS:
    - No random rollouts. Leaf evaluation is done by the neural network.
    - The network's policy prior guides exploration via PUCT.
    - Dirichlet noise at the root for exploration during self-play.
"""

import math
import numpy as np
from typing import Optional, Dict, Tuple

from game import QuoridorGame, MoveEncoder, Move


class MCTSNode:
    """A node in the search tree."""

    __slots__ = [
        "game", "parent", "move", "children",
        "visit_count", "value_sum", "prior",
    ]

    def __init__(self, game: QuoridorGame, parent: Optional["MCTSNode"] = None,
                 move: Optional[Move] = None, prior: float = 0.0):
        self.game = game
        self.parent = parent
        self.move = move          # move that led to this node
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


class MCTS:
    """
    AlphaZero-style MCTS.

    Usage:
        mcts = MCTS(network, config)
        action_probs = mcts.search(game_state)
    """

    def __init__(self, network, device: str = "cpu",
                 num_simulations: int = 200,
                 c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, game: QuoridorGame, add_noise: bool = True) -> np.ndarray:
        """
        Run MCTS from the given game state.

        Returns:
            action_probs: (140,) array of visit-count-based action probabilities.
        """
        root = MCTSNode(game.clone())
        self._expand(root)

        # Add Dirichlet noise at root for exploration
        if add_noise and root.children:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * len(root.children)
            )
            for i, child in enumerate(root.children.values()):
                child.prior = (
                    (1 - self.dirichlet_epsilon) * child.prior
                    + self.dirichlet_epsilon * noise[i]
                )

        for _ in range(self.num_simulations):
            node = root

            # ── SELECT: walk down tree using PUCT ──
            while not node.is_leaf() and not node.is_terminal():
                node = self._select_child(node)

            # ── EVALUATE ──
            if node.is_terminal():
                # Game is over: value from perspective of node's parent's player
                # The player who just moved (parent) won
                value = 1.0 if node.game.winner is not None else 0.0
            else:
                # Expand and evaluate with network
                value = self._expand(node)

            # ── BACKPROP ──
            self._backpropagate(node, value)

        # Build action probability vector from visit counts
        action_probs = np.zeros(MoveEncoder.TOTAL_ACTIONS, dtype=np.float32)
        pawn_pos = game.pawns[game.current_player]
        for move, child in root.children.items():
            idx = MoveEncoder.encode(move, pawn_pos)
            action_probs[idx] = child.visit_count

        # Normalise
        total = action_probs.sum()
        if total > 0:
            action_probs /= total

        return action_probs

    def _select_child(self, node: MCTSNode) -> "MCTSNode":
        """Select child with highest PUCT score."""
        sqrt_parent = math.sqrt(node.visit_count)
        best_score = -float("inf")
        best_child = None

        for child in node.children.values():
            # Q is from the perspective of the player at `node`
            # Child's value_sum is from child's parent's perspective (= node's player)
            q = child.q_value
            # Exploration bonus
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode) -> float:
        """
        Expand a leaf node: create children and return the network's
        value estimate (from the perspective of node's current player).
        """
        game = node.game
        if game.is_over:
            return 0.0

        # Get network evaluation (mask over ALL legal moves for proper priors)
        state = game.to_tensor()
        legal_mask = MoveEncoder.legal_mask(game)
        policy, value = self.network.predict(state, legal_mask, self.device)

        # Boost pawn move priors to prevent wall-addiction cold start.
        # Without this, an untrained network puts ~97% on walls (128 wall
        # moves vs 3 pawn moves), MCTS never explores pawn moves, and the
        # network learns to be even MORE wall-obsessed. This ensures pawn
        # moves get enough exploration to discover that advancing wins games.
        policy = self._boost_pawn_priors(policy, game)

        # Create child nodes — only for PROBABLE moves (speed optimization)
        probable_moves = game.get_probable_moves()
        pawn_pos = game.pawns[game.current_player]

        for move in probable_moves:
            idx = MoveEncoder.encode(move, pawn_pos)
            child_game = game.clone()
            child_game.make_move(move)
            child = MCTSNode(
                game=child_game,
                parent=node,
                move=move,
                prior=policy[idx],
            )
            node.children[move] = child

        # Return value (negated because it's from current player's perspective,
        # and backprop expects value from parent's perspective)
        return -value

    @staticmethod
    def _boost_pawn_priors(policy: np.ndarray, game) -> np.ndarray:
        """
        Ensure pawn moves get at least min_pawn_share of total probability.
        This is a training aid, not a heuristic — it prevents the cold-start
        problem where the network never explores pawn moves because walls
        outnumber them ~40:1. As training progresses, the network learns
        appropriate pawn weights and this becomes a no-op.
        """
        min_pawn_share = 0.35

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
            return policy  # network already gives enough weight to pawns

        # Redistribute: give pawn moves min_pawn_share, scale walls down
        boosted = policy.copy()
        wall_total = sum(policy[i] for i in wall_indices)

        if wall_total > 0:
            wall_scale = (1.0 - min_pawn_share) / wall_total
            for i in wall_indices:
                boosted[i] *= wall_scale

        # Distribute min_pawn_share among pawn moves proportionally
        if pawn_total > 0:
            pawn_scale = min_pawn_share / pawn_total
            for i in pawn_indices:
                boosted[i] *= pawn_scale
        else:
            # Uniform over pawn moves if network gives them all zero
            for i in pawn_indices:
                boosted[i] = min_pawn_share / len(pawn_indices)

        return boosted

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Walk back to root, updating visit counts and value sums.
        Value alternates sign at each level (opponent's gain = my loss).
        """
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # flip perspective
            node = node.parent


def select_action(action_probs: np.ndarray, temperature: float = 1.0) -> int:
    """
    Choose an action index from the MCTS probability distribution.

    temperature = 1.0  → sample proportionally (exploratory)
    temperature → 0    → pick the most-visited action (greedy)
    """
    if temperature < 1e-6:
        # Greedy: break ties randomly
        max_prob = action_probs.max()
        candidates = np.where(action_probs == max_prob)[0]
        return np.random.choice(candidates)

    # Apply temperature
    probs = action_probs ** (1.0 / temperature)
    total = probs.sum()
    if total <= 0:
        # Fallback: uniform over nonzero
        nonzero = np.nonzero(action_probs)[0]
        return np.random.choice(nonzero) if len(nonzero) > 0 else 0
    probs /= total
    return np.random.choice(len(probs), p=probs)
