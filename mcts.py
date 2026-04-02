"""
Monte Carlo Tree Search for Quoridor.

Implements:
    - Standard PUCT MCTS (AlphaZero-style)
    - Gumbel MCTS with Sequential Halving (sample-efficient variant)

Based on "Policy improvement by planning with Gumbel" (Danihelka et al., 2022).
Gumbel MCTS gets comparable policy quality from 32-64 sims vs 200+ for standard.
"""

import math
import numpy as np
from typing import Optional, Dict, List

from engine import QuoridorGame, MoveEncoder, Move, _USING_FAST_ENGINE
from config import Config


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = [
        "game", "parent", "parent_action", "children",
        "visit_count", "value_sum", "prior", "is_expanded",
    ]

    def __init__(self, game: QuoridorGame, parent: Optional["MCTSNode"] = None,
                 parent_action: int = -1, prior: float = 0.0):
        self.game = game
        self.parent = parent
        self.parent_action = parent_action
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


# ---------------------------------------------------------------------------
# Fast game cloning (avoids deepcopy overhead)
# ---------------------------------------------------------------------------

if _USING_FAST_ENGINE:
    # Cython engine .clone() is already C-level fast
    def _fast_clone(game):
        return game.clone()
else:
    def _fast_clone(game):
        """Clone a game state without deepcopy. ~5-10x faster than deepcopy."""
        g = QuoridorGame.__new__(QuoridorGame)
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
        g.move_history = list(game.move_history)
        return g


# ---------------------------------------------------------------------------
# Standard PUCT MCTS
# ---------------------------------------------------------------------------

class MCTS:
    """AlphaZero-style MCTS with PUCT selection."""

    def __init__(self, network, device, config: Config):
        self.network = network
        self.device = device
        self.config = config

    def search(self, game: QuoridorGame, num_simulations: int = None,
               add_noise: bool = True) -> np.ndarray:
        """Run MCTS and return action probability distribution (140,)."""
        n_sims = num_simulations or self.config.num_simulations
        root = MCTSNode(game)
        value = self._expand(root)

        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(n_sims):
            node = root
            # SELECT
            while node.is_expanded and not node.game.is_over:
                node = self._select_child(node)
            # EXPAND + EVALUATE
            if not node.game.is_over:
                value = self._expand(node)
            else:
                # Terminal node: value from perspective of the node's parent's player
                if node.game.winner is not None:
                    # The player who just moved won
                    value = -1.0  # bad for the current player at this node
                else:
                    value = 0.0
            # BACKUP
            self._backup(node, value)

        return self._get_action_probs(root)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """PUCT selection."""
        best_score = -float("inf")
        best_child = None
        sqrt_parent = math.sqrt(node.visit_count)

        for child in node.children.values():
            q = -child.q_value  # negate: child's value is from opponent's perspective
            u = (self.config.c_puct * child.prior * sqrt_parent
                 / (1 + child.visit_count))
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand(self, node: MCTSNode) -> float:
        """Evaluate with network and create children for probable moves."""
        state = node.game.to_tensor()
        mask = MoveEncoder.legal_mask(node.game)
        policy, value = self.network.predict(state, mask, self.device)

        pawn_pos = node.game.pawns[node.game.current_player]
        for move in node.game.get_probable_moves():
            idx = MoveEncoder.encode(move, pawn_pos)
            child_game = _fast_clone(node.game)
            child_game.make_move(move)
            child = MCTSNode(child_game, parent=node,
                             parent_action=idx, prior=policy[idx])
            node.children[idx] = child

        node.is_expanded = True
        return value

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Propagate value up, negating at each level."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        """Add exploration noise to root priors."""
        if not node.children:
            return
        actions = list(node.children.keys())
        noise = np.random.dirichlet(
            [self.config.dirichlet_alpha] * len(actions)
        )
        eps = self.config.dirichlet_epsilon
        for a, n in zip(actions, noise):
            node.children[a].prior = (1 - eps) * node.children[a].prior + eps * n

    def _get_action_probs(self, root: MCTSNode) -> np.ndarray:
        """Convert visit counts to probabilities."""
        probs = np.zeros(MoveEncoder.TOTAL_ACTIONS, dtype=np.float32)
        for idx, child in root.children.items():
            probs[idx] = child.visit_count
        total = probs.sum()
        if total > 0:
            probs /= total
        return probs


# ---------------------------------------------------------------------------
# Gumbel MCTS with Sequential Halving
# ---------------------------------------------------------------------------

class GumbelMCTS:
    """Sample-efficient MCTS using Gumbel noise + Sequential Halving at root."""

    def __init__(self, network, device, config: Config):
        self.network = network
        self.device = device
        self.config = config

    def search(self, game: QuoridorGame, num_simulations: int = None,
               add_noise: bool = True) -> np.ndarray:
        """Run Gumbel MCTS. Returns action probabilities (140,)."""
        n_sims = num_simulations or self.config.num_simulations

        # Get network policy at root
        state = game.to_tensor()
        mask = MoveEncoder.legal_mask(game)
        policy, root_value = self.network.predict(state, mask, self.device)

        # Build root children for probable moves
        root = MCTSNode(game)
        pawn_pos = game.pawns[game.current_player]
        probable_moves = game.get_probable_moves()

        action_indices = []
        log_priors = []
        for move in probable_moves:
            idx = MoveEncoder.encode(move, pawn_pos)
            child_game = _fast_clone(game)
            child_game.make_move(move)
            child = MCTSNode(child_game, parent=root,
                             parent_action=idx, prior=policy[idx])
            root.children[idx] = child
            action_indices.append(idx)
            log_priors.append(math.log(max(policy[idx], 1e-8)))
        root.is_expanded = True

        if not action_indices:
            return np.zeros(MoveEncoder.TOTAL_ACTIONS, dtype=np.float32)

        n_actions = len(action_indices)
        log_priors = np.array(log_priors, dtype=np.float32)

        # Add Gumbel noise for exploration during training
        if add_noise:
            gumbel_noise = np.random.gumbel(size=n_actions).astype(np.float32)
        else:
            gumbel_noise = np.zeros(n_actions, dtype=np.float32)

        # Perturbed logits for action selection
        perturbed = log_priors + gumbel_noise

        # Sequential Halving
        k = min(self.config.max_considered_actions, n_actions)
        top_k_local = np.argsort(perturbed)[-k:]  # indices into action_indices

        # Allocate simulations via Sequential Halving
        remaining = list(top_k_local)
        sims_used = 0
        n_phases = max(1, int(math.ceil(math.log2(k))))
        sims_per_phase = max(1, n_sims // n_phases)

        for phase in range(n_phases):
            if len(remaining) <= 1:
                break
            sims_each = max(1, sims_per_phase // len(remaining))

            for local_idx in remaining:
                action_idx = action_indices[local_idx]
                child = root.children[action_idx]
                for _ in range(sims_each):
                    if sims_used >= n_sims:
                        break
                    self._simulate(child)
                    sims_used += 1

            # Compute scores and halve
            scores = []
            for local_idx in remaining:
                action_idx = action_indices[local_idx]
                child = root.children[action_idx]
                q = -child.q_value if child.visit_count > 0 else 0.0
                # Completed Q-transform
                score = perturbed[local_idx] + _sigma_q(q, child.visit_count)
                scores.append(score)

            # Keep top half
            sorted_pairs = sorted(zip(scores, remaining), reverse=True)
            half = max(1, len(remaining) // 2)
            remaining = [idx for _, idx in sorted_pairs[:half]]

        # Build improved policy from all visited children
        probs = np.zeros(MoveEncoder.TOTAL_ACTIONS, dtype=np.float32)
        for local_idx, action_idx in enumerate(action_indices):
            child = root.children[action_idx]
            if child.visit_count > 0:
                q = -child.q_value
                improved_logit = log_priors[local_idx] + _sigma_q(q, child.visit_count)
            else:
                improved_logit = log_priors[local_idx]
            probs[action_idx] = improved_logit

        # Softmax over actions that were considered
        considered = [action_indices[i] for i in range(n_actions)]
        logits = np.array([probs[a] for a in considered])
        logits -= logits.max()
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits / exp_logits.sum()

        result = np.zeros(MoveEncoder.TOTAL_ACTIONS, dtype=np.float32)
        for a, p in zip(considered, softmax_probs):
            result[a] = p

        return result

    def _simulate(self, node: MCTSNode) -> None:
        """Single MCTS simulation from a node (PUCT below root)."""
        path = [node]

        # SELECT down to leaf
        current = node
        while current.is_expanded and not current.game.is_over:
            current = self._select_child_puct(current)
            path.append(current)

        # EXPAND + EVALUATE
        if not current.game.is_over:
            value = self._expand_node(current)
        else:
            if current.game.winner is not None:
                value = -1.0  # bad for current player at this node
            else:
                value = 0.0

        # BACKUP
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    def _select_child_puct(self, node: MCTSNode) -> MCTSNode:
        best_score = -float("inf")
        best_child = None
        sqrt_parent = math.sqrt(node.visit_count + 1)

        for child in node.children.values():
            q = -child.q_value
            u = (self.config.c_puct * child.prior * sqrt_parent
                 / (1 + child.visit_count))
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _expand_node(self, node: MCTSNode) -> float:
        """Network evaluation + create children."""
        state = node.game.to_tensor()
        mask = MoveEncoder.legal_mask(node.game)
        policy, value = self.network.predict(state, mask, self.device)

        pawn_pos = node.game.pawns[node.game.current_player]
        for move in node.game.get_probable_moves():
            idx = MoveEncoder.encode(move, pawn_pos)
            child_game = _fast_clone(node.game)
            child_game.make_move(move)
            child = MCTSNode(child_game, parent=node,
                             parent_action=idx, prior=policy[idx])
            node.children[idx] = child

        node.is_expanded = True
        return value


def _sigma_q(q: float, visit_count: int, c: float = 50.0) -> float:
    """Transform Q-value for Gumbel policy improvement.

    Scales Q to be comparable to log-prior magnitudes.
    c controls the scaling — higher c means more trust in search results.
    """
    if visit_count == 0:
        return 0.0
    return c * q


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def select_action(probs: np.ndarray, temperature: float = 1.0) -> int:
    """Sample an action from probability distribution with temperature."""
    if temperature < 0.01:
        # Greedy
        return int(np.argmax(probs))

    # Apply temperature
    log_probs = np.log(np.maximum(probs, 1e-8))
    log_probs /= temperature
    log_probs -= log_probs.max()
    exp_probs = np.exp(log_probs)
    exp_probs /= exp_probs.sum()

    return int(np.random.choice(len(exp_probs), p=exp_probs))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    from network import QuoridorNet
    from config import Config

    cfg = Config()
    net = QuoridorNet(cfg.input_planes, cfg.num_res_blocks,
                      cfg.num_channels, cfg.action_size)
    device = torch.device("cpu")

    game = QuoridorGame(2)

    # Test standard MCTS with 8 sims
    print("Standard MCTS (8 sims):")
    mcts = MCTS(net, device, cfg)
    probs = mcts.search(game, num_simulations=8, add_noise=False)
    print(f"  Sum: {probs.sum():.4f}")
    print(f"  Non-zero actions: {(probs > 0).sum()}")
    top = np.argsort(probs)[-3:][::-1]
    for idx in top:
        pos = game.pawns[game.current_player]
        move = MoveEncoder.decode(idx, pos)
        print(f"  Action {idx}: {move} ({probs[idx]:.3f})")

    # Test Gumbel MCTS with 16 sims
    print("\nGumbel MCTS (16 sims):")
    gmcts = GumbelMCTS(net, device, cfg)
    probs = gmcts.search(game, num_simulations=16, add_noise=False)
    print(f"  Sum: {probs.sum():.4f}")
    print(f"  Non-zero actions: {(probs > 0).sum()}")
    top = np.argsort(probs)[-3:][::-1]
    for idx in top:
        pos = game.pawns[game.current_player]
        move = MoveEncoder.decode(idx, pos)
        print(f"  Action {idx}: {move} ({probs[idx]:.3f})")
