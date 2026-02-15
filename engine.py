"""
Quoridor Game Engine
====================
Supports 2-player and 4-player modes with full rule enforcement.

Coordinate system:
    Cells:  (row, col) on a 9x9 grid. Row 0 = top, row 8 = bottom.
    Walls:  Placed at groove intersections on an 8x8 grid.
            Each wall spans 2 cell-boundaries.
            Orientation: 'H' (horizontal) or 'V' (vertical).

    A horizontal wall at (wr, wc) blocks movement between:
        (wr, wc) <-> (wr+1, wc)   and   (wr, wc+1) <-> (wr+1, wc+1)

    A vertical wall at (wr, wc) blocks movement between:
        (wr, wc) <-> (wr, wc+1)   and   (wr+1, wc) <-> (wr+1, wc+1)

2-player setup:
    Player 0: starts (8, 4), goal = reach row 0, 10 walls
    Player 1: starts (0, 4), goal = reach row 8, 10 walls

4-player setup:
    Player 0: starts (8, 4), goal = reach row 0, 5 walls
    Player 1: starts (0, 4), goal = reach row 8, 5 walls
    Player 2: starts (4, 0), goal = reach col 8, 5 walls
    Player 3: starts (4, 8), goal = reach col 0, 5 walls
"""

from collections import deque
from copy import deepcopy
from typing import Optional, List, Tuple, Set, Dict

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Cell = Tuple[int, int]
Move = Tuple[str, tuple]  # ('move', (r, c))  or  ('wall', (r, c, 'H'/'V'))


# ===========================================================================
# QuoridorGame
# ===========================================================================

class QuoridorGame:
    """Full game state and rule enforcement for Quoridor."""

    BOARD_SIZE = 9
    WALL_GRID = 8  # groove intersections: 0..7

    SETUP = {
        2: {
            "starts": [(8, 4), (0, 4)],
            "goals": [{"row": 0}, {"row": 8}],
            "walls_per_player": 10,
        },
        4: {
            "starts": [(8, 4), (0, 4), (4, 0), (4, 8)],
            "goals": [{"row": 0}, {"row": 8}, {"col": 8}, {"col": 0}],
            "walls_per_player": 5,
        },
    }

    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self, num_players: int = 2):
        if num_players not in (2, 4):
            raise ValueError("num_players must be 2 or 4")

        setup = self.SETUP[num_players]
        self.num_players: int = num_players
        self.pawns: List[Cell] = list(setup["starts"])
        self.goals: List[Dict] = setup["goals"]
        self.walls_remaining: List[int] = [setup["walls_per_player"]] * num_players
        self.current_player: int = 0
        self.winner: Optional[int] = None

        # Placed walls -------------------------------------------------
        self.walls_placed: Set[Tuple[int, int, str]] = set()
        self.wall_centers: Set[Tuple[int, int]] = set()

        # Blocked edges (fast lookup) ----------------------------------
        #   h_blocked contains (r, c) => vertical movement between
        #       cell (r, c) and cell (r+1, c) is blocked.
        #   v_blocked contains (r, c) => horizontal movement between
        #       cell (r, c) and cell (r, c+1) is blocked.
        self.h_blocked: Set[Tuple[int, int]] = set()
        self.v_blocked: Set[Tuple[int, int]] = set()

        self.move_history: List[Move] = []

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _in_bounds(r: int, c: int) -> bool:
        return 0 <= r < 9 and 0 <= c < 9

    def _can_step(self, r1: int, c1: int, r2: int, c2: int) -> bool:
        """Return True if a single step from (r1,c1) to (r2,c2) is not
        blocked by a wall (does NOT check occupancy)."""
        if not self._in_bounds(r2, c2):
            return False
        dr, dc = r2 - r1, c2 - c1
        if dr == 1:
            return (r1, c1) not in self.h_blocked
        if dr == -1:
            return (r2, c2) not in self.h_blocked
        if dc == 1:
            return (r1, c1) not in self.v_blocked
        if dc == -1:
            return (r2, c2) not in self.v_blocked
        return False

    def _reached_goal(self, player: int) -> bool:
        r, c = self.pawns[player]
        g = self.goals[player]
        return ("row" in g and r == g["row"]) or ("col" in g and c == g["col"])

    # ------------------------------------------------------------------
    # Pawn movement (including jumps)
    # ------------------------------------------------------------------

    def _get_pawn_moves(self, player: int) -> List[Cell]:
        """All legal pawn destinations for *player*."""
        r, c = self.pawns[player]
        occupied = set(self.pawns)
        moves: List[Cell] = []

        for dr, dc in self.DIRECTIONS:
            nr, nc = r + dr, c + dc
            if not self._can_step(r, c, nr, nc):
                continue

            if (nr, nc) not in occupied:
                # Simple step to empty adjacent cell
                moves.append((nr, nc))
            else:
                # Adjacent cell has an opponent -> try jumping
                jr, jc = nr + dr, nc + dc  # straight jump target
                if (self._in_bounds(jr, jc)
                        and self._can_step(nr, nc, jr, jc)
                        and (jr, jc) not in occupied):
                    moves.append((jr, jc))
                else:
                    # Straight jump blocked -> diagonal jumps
                    for ddr, ddc in self.DIRECTIONS:
                        if (ddr, ddc) == (dr, dc) or (ddr, ddc) == (-dr, -dc):
                            continue
                        diag = (nr + ddr, nc + ddc)
                        if (self._in_bounds(*diag)
                                and self._can_step(nr, nc, *diag)
                                and diag not in occupied):
                            moves.append(diag)

        return moves

    # ------------------------------------------------------------------
    # Wall logic
    # ------------------------------------------------------------------

    def _wall_has_conflict(self, r: int, c: int, o: str) -> bool:
        """Does the wall at (r, c, o) overlap/cross an existing wall?"""
        if not (0 <= r < self.WALL_GRID and 0 <= c < self.WALL_GRID):
            return True
        if (r, c) in self.wall_centers:          # crossing at center
            return True
        if o == "H":
            return ((r, c - 1, "H") in self.walls_placed
                    or (r, c + 1, "H") in self.walls_placed)
        else:  # V
            return ((r - 1, c, "V") in self.walls_placed
                    or (r + 1, c, "V") in self.walls_placed)

    def _add_wall_edges(self, r: int, c: int, o: str) -> None:
        if o == "H":
            self.h_blocked.add((r, c))
            self.h_blocked.add((r, c + 1))
        else:
            self.v_blocked.add((r, c))
            self.v_blocked.add((r + 1, c))

    def _remove_wall_edges(self, r: int, c: int, o: str) -> None:
        if o == "H":
            self.h_blocked.discard((r, c))
            self.h_blocked.discard((r, c + 1))
        else:
            self.v_blocked.discard((r, c))
            self.v_blocked.discard((r + 1, c))

    def _has_path(self, player: int) -> bool:
        """BFS: can *player* reach their goal?"""
        start = self.pawns[player]
        goal = self.goals[player]
        visited = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            if ("row" in goal and r == goal["row"]) or \
               ("col" in goal and c == goal["col"]):
                return True
            for dr, dc in self.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and self._can_step(r, c, nr, nc):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    def _all_paths_ok(self) -> bool:
        return all(
            self._reached_goal(p) or self._has_path(p)
            for p in range(self.num_players)
        )

    def is_valid_wall(self, r: int, c: int, o: str) -> bool:
        """Full validation: no conflict + all players keep a path."""
        if self._wall_has_conflict(r, c, o):
            return False
        self._add_wall_edges(r, c, o)
        ok = self._all_paths_ok()
        self._remove_wall_edges(r, c, o)
        return ok

    # ------------------------------------------------------------------
    # Legal move generation
    # ------------------------------------------------------------------

    def get_legal_pawn_moves(self) -> List[Move]:
        return [("move", d) for d in self._get_pawn_moves(self.current_player)]

    def get_legal_walls(self) -> List[Move]:
        if self.walls_remaining[self.current_player] <= 0:
            return []
        return [
            ("wall", (r, c, o))
            for r in range(self.WALL_GRID)
            for c in range(self.WALL_GRID)
            for o in ("H", "V")
            if self.is_valid_wall(r, c, o)
        ]

    def get_probable_walls(self, radius: int = 1) -> List[Move]:
        """
        Like get_legal_walls but only considers positions near pawns,
        existing walls, or on opponents' shortest paths. Returns ~20-40
        moves instead of ~100+. The real speedup comes from MCTS creating
        fewer child nodes (fewer game clones per expansion).
        """
        if self.walls_remaining[self.current_player] <= 0:
            return []

        candidates: set = set()

        # Near all pawns (radius 1 = tight)
        for pr, pc in self.pawns:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    wr, wc = pr + dr, pc + dc
                    if 0 <= wr < self.WALL_GRID and 0 <= wc < self.WALL_GRID:
                        candidates.add((wr, wc))

        # Adjacent to existing walls (extend the maze)
        for wr, wc, _ in self.walls_placed:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nr, nc = wr + dr, wc + dc
                    if 0 <= nr < self.WALL_GRID and 0 <= nc < self.WALL_GRID:
                        candidates.add((nr, nc))

        # On opponents' shortest paths (strategically disruptive)
        cp = self.current_player
        for p in range(self.num_players):
            if p == cp:
                continue
            path = self.shortest_path(p)
            if path:
                for r, c in path:
                    for dr in range(-1, 1):
                        for dc in range(-1, 1):
                            wr, wc = r + dr, c + dc
                            if 0 <= wr < self.WALL_GRID and 0 <= wc < self.WALL_GRID:
                                candidates.add((wr, wc))

        return [
            ("wall", (r, c, o))
            for r, c in candidates
            for o in ("H", "V")
            if self.is_valid_wall(r, c, o)
        ]

    def get_probable_moves(self) -> List[Move]:
        """Pawn moves + probable walls. Use this in MCTS for speed."""
        if self.winner is not None:
            return []
        return self.get_legal_pawn_moves() + self.get_probable_walls()

    def get_legal_moves(self) -> List[Move]:
        """All legal moves for the current player."""
        if self.winner is not None:
            return []
        return self.get_legal_pawn_moves() + self.get_legal_walls()

    # ------------------------------------------------------------------
    # Applying moves
    # ------------------------------------------------------------------

    def make_move(self, move: Move) -> None:
        """Apply *move* and advance turn.  Raises ValueError on illegal moves."""
        if self.winner is not None:
            raise ValueError("Game is already over")

        action, data = move

        if action == "move":
            if data not in self._get_pawn_moves(self.current_player):
                raise ValueError(f"Illegal pawn move to {data}")
            self.pawns[self.current_player] = data

        elif action == "wall":
            r, c, o = data
            if self.walls_remaining[self.current_player] <= 0:
                raise ValueError("No walls remaining")
            if not self.is_valid_wall(r, c, o):
                raise ValueError(f"Illegal wall: {data}")
            self.walls_placed.add((r, c, o))
            self.wall_centers.add((r, c))
            self._add_wall_edges(r, c, o)
            self.walls_remaining[self.current_player] -= 1

        else:
            raise ValueError(f"Unknown action type: {action!r}")

        self.move_history.append(move)

        if self._reached_goal(self.current_player):
            self.winner = self.current_player
        else:
            self.current_player = (self.current_player + 1) % self.num_players

    # ------------------------------------------------------------------
    # Shortest path (for heuristics / evaluation)
    # ------------------------------------------------------------------

    def shortest_path_length(self, player: int) -> Optional[int]:
        """BFS distance from *player*'s pawn to their goal row/col."""
        start = self.pawns[player]
        goal = self.goals[player]
        dist = {start: 0}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            d = dist[(r, c)]
            if ("row" in goal and r == goal["row"]) or \
               ("col" in goal and c == goal["col"]):
                return d
            for dr, dc in self.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in dist and self._can_step(r, c, nr, nc):
                    dist[(nr, nc)] = d + 1
                    queue.append((nr, nc))
        return None  # no path (should never happen in a legal state)

    def shortest_path(self, player: int) -> Optional[List[Cell]]:
        """BFS to return one shortest path as a list of cells (incl. start)."""
        start = self.pawns[player]
        goal = self.goals[player]
        prev: Dict[Cell, Optional[Cell]] = {start: None}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            if ("row" in goal and r == goal["row"]) or \
               ("col" in goal and c == goal["col"]):
                # Reconstruct path
                path = []
                cur: Optional[Cell] = (r, c)
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                return path[::-1]
            for dr, dc in self.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in prev and self._can_step(r, c, nr, nc):
                    prev[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
        return None

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clone(self) -> "QuoridorGame":
        """Deep copy -- safe for MCTS tree expansion."""
        return deepcopy(self)

    @property
    def is_over(self) -> bool:
        return self.winner is not None

    # ------------------------------------------------------------------
    # Neural-network state tensor
    # ------------------------------------------------------------------

    def to_tensor(self) -> np.ndarray:
        """
        Encode the board as a float32 tensor of shape (C, 9, 9).

        2-player planes (C = 9):
            0  current player's pawn
            1  opponent's pawn
            2  h_blocked edges
            3  v_blocked edges
            4  current player walls remaining  (uniform fill, normalised)
            5  opponent walls remaining
            6  current-player-is-player-0 indicator
            7  current player distance to goal (normalised by 8)
            8  opponent distance to goal (normalised by 8)

        4-player planes (C = 14):
            0-3   pawns (current player first, then clockwise)
            4     h_blocked
            5     v_blocked
            6-9   walls remaining (same order as pawns)
            10-13 distance to goal (same order as pawns)
        """
        S = self.BOARD_SIZE
        cp = self.current_player

        if self.num_players == 2:
            planes = np.zeros((9, S, S), dtype=np.float32)
            opp = 1 - cp

            planes[0][self.pawns[cp]] = 1.0
            planes[1][self.pawns[opp]] = 1.0

            for r, c in self.h_blocked:
                planes[2, r, c] = 1.0
            for r, c in self.v_blocked:
                planes[3, r, c] = 1.0

            max_w = self.SETUP[2]["walls_per_player"]
            planes[4] = self.walls_remaining[cp] / max_w
            planes[5] = self.walls_remaining[opp] / max_w
            planes[6] = float(cp == 0)

            # Distance to goal — the key info a human sees instantly
            d_cp = self.shortest_path_length(cp)
            d_opp = self.shortest_path_length(opp)
            planes[7] = (d_cp if d_cp is not None else 8) / 8.0
            planes[8] = (d_opp if d_opp is not None else 8) / 8.0

        else:
            planes = np.zeros((14, S, S), dtype=np.float32)
            for i in range(4):
                p = (cp + i) % 4
                planes[i][self.pawns[p]] = 1.0

            for r, c in self.h_blocked:
                planes[4, r, c] = 1.0
            for r, c in self.v_blocked:
                planes[5, r, c] = 1.0

            max_w = self.SETUP[4]["walls_per_player"]
            for i in range(4):
                p = (cp + i) % 4
                planes[6 + i] = self.walls_remaining[p] / max_w
                d = self.shortest_path_length(p)
                planes[10 + i] = (d if d is not None else 8) / 8.0

        return planes

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "num_players": self.num_players,
            "pawns": [list(p) for p in self.pawns],
            "walls_remaining": list(self.walls_remaining),
            "walls_placed": [list(w) for w in self.walls_placed],
            "current_player": self.current_player,
            "winner": self.winner,
            "move_history": [(a, list(d)) for a, d in self.move_history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuoridorGame":
        game = cls(data["num_players"])
        game.pawns = [tuple(p) for p in data["pawns"]]
        game.walls_remaining = list(data["walls_remaining"])
        game.current_player = data["current_player"]
        game.winner = data["winner"]
        for w in data["walls_placed"]:
            r, c, o = w
            game.walls_placed.add((r, c, o))
            game.wall_centers.add((r, c))
            game._add_wall_edges(r, c, o)
        game.move_history = [(a, tuple(d)) for a, d in data["move_history"]]
        return game

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        S = self.BOARD_SIZE
        player_chars = ["0", "1", "2", "3"]
        lines = []

        lines.append("    " + "   ".join(str(c) for c in range(S)))
        lines.append("  +" + "---+" * S)

        for r in range(S):
            row = f"{r} |"
            for c in range(S):
                # Cell content
                ch = " . "
                for p in range(self.num_players):
                    if self.pawns[p] == (r, c):
                        ch = f" {player_chars[p]} "
                        break
                row += ch
                # Right edge
                if c < S - 1:
                    row += "|" if (r, c) in self.v_blocked else " "
                else:
                    row += "|"
            lines.append(row)

            # Horizontal divider
            if r < S - 1:
                div = "  +"
                for c in range(S):
                    div += "===" if (r, c) in self.h_blocked else "---"
                    div += "+"
                lines.append(div)

        lines.append("  +" + "---+" * S)

        # Info line
        if self.winner is not None:
            lines.append(f"\n  Player {self.winner} wins!")
        else:
            walls_info = "  ".join(
                f"P{i}:{w}w" for i, w in enumerate(self.walls_remaining)
            )
            lines.append(f"\n  Turn: Player {self.current_player}  |  {walls_info}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"QuoridorGame(players={self.num_players}, "
                f"turn={self.current_player}, "
                f"moves={len(self.move_history)}, "
                f"winner={self.winner})")


# ===========================================================================
# MoveEncoder -- maps moves <-> integer indices for policy networks
# ===========================================================================

class MoveEncoder:
    """
    Fixed action space of size 140:
        [0, 127]   -> wall placements:  index = r*16 + c*2 + (0=H, 1=V)
        [128, 139]  -> pawn moves as offsets from current position
    """

    PAWN_OFFSETS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),       # 4 adjacent
        (-2, 0), (2, 0), (0, -2), (0, 2),        # 4 straight jumps
        (-1, -1), (-1, 1), (1, -1), (1, 1),      # 4 diagonal jumps
    ]

    TOTAL_ACTIONS = 128 + 12  # 140

    @classmethod
    def encode(cls, move: Move, pawn_pos: Cell) -> int:
        action, data = move
        if action == "wall":
            r, c, o = data
            return r * 16 + c * 2 + (0 if o == "H" else 1)
        dr = data[0] - pawn_pos[0]
        dc = data[1] - pawn_pos[1]
        return 128 + cls.PAWN_OFFSETS.index((dr, dc))

    @classmethod
    def decode(cls, idx: int, pawn_pos: Cell) -> Move:
        if idx < 128:
            r = idx // 16
            c = (idx % 16) // 2
            o = "H" if idx % 2 == 0 else "V"
            return ("wall", (r, c, o))
        dr, dc = cls.PAWN_OFFSETS[idx - 128]
        return ("move", (pawn_pos[0] + dr, pawn_pos[1] + dc))

    @classmethod
    def legal_mask(cls, game: QuoridorGame) -> np.ndarray:
        """Binary mask over the action space (1 = legal)."""
        mask = np.zeros(cls.TOTAL_ACTIONS, dtype=np.float32)
        pos = game.pawns[game.current_player]
        for move in game.get_legal_moves():
            mask[cls.encode(move, pos)] = 1.0
        return mask


# ===========================================================================
# Quick smoke test
# ===========================================================================

if __name__ == "__main__":
    # --- 2-player demo ---
    g = QuoridorGame(2)
    print(g)
    print(f"\nLegal moves at start: {len(g.get_legal_moves())}")

    g.make_move(("move", (7, 4)))
    g.make_move(("move", (1, 4)))
    g.make_move(("wall", (1, 3, "H")))
    print("\nAfter 3 moves (P0 fwd, P1 fwd, P0 wall):")
    print(g)
    print(f"  P0 shortest path: {g.shortest_path_length(0)}")
    print(f"  P1 shortest path: {g.shortest_path_length(1)}")

    # --- 4-player demo ---
    print("\n" + "=" * 60)
    g4 = QuoridorGame(4)
    print(g4)
    print(f"\nLegal moves at start (4p): {len(g4.get_legal_moves())}")

    # --- Tensor shape ---
    print(f"\n2p tensor shape: {QuoridorGame(2).to_tensor().shape}")
    print(f"4p tensor shape: {QuoridorGame(4).to_tensor().shape}")

    # --- MoveEncoder round-trip ---
    g2 = QuoridorGame(2)
    pos = g2.pawns[0]
    for m in g2.get_legal_moves()[:5]:
        idx = MoveEncoder.encode(m, pos)
        m2 = MoveEncoder.decode(idx, pos)
        print(f"  {m}  ->  idx {idx}  ->  {m2}")
