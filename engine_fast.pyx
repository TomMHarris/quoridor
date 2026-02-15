# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Fast Quoridor engine in Cython.

Same API as engine.py but runs 50-100x faster by using:
  - C-typed arrays instead of Python sets/dicts
  - Pre-allocated BFS queue (no heap allocation)
  - Direct memory copy for cloning (no deepcopy)
  - Compiled C loops instead of interpreted Python

Build:  python setup.py build_ext --inplace
"""

import numpy as np
cimport numpy as np
from libc.string cimport memcpy, memset

# Board constants
DEF BOARD_SIZE = 9
DEF WALL_GRID = 8
DEF MAX_PLAYERS = 4
DEF MAX_WALLS_TOTAL = 40   # 4 players x 10 walls max
DEF MAX_MOVES = 300        # move history capacity
DEF BFS_QUEUE_SIZE = 81    # 9x9
DEF MAX_LEGAL_MOVES = 140  # pawn (12) + walls (128)

# Move types
DEF MOVE_PAWN = 0
DEF MOVE_WALL = 1

# Directions: up, down, left, right
cdef int DR[4]
cdef int DC[4]
DR[0] = -1; DR[1] = 1; DR[2] = 0; DR[3] = 0
DC[0] = 0; DC[1] = 0; DC[2] = -1; DC[3] = 1


# =====================================================================
# Compact move struct
# =====================================================================
ctypedef struct CMove:
    int action    # MOVE_PAWN or MOVE_WALL
    int r         # target row (pawn) or wall row
    int c         # target col (pawn) or wall col
    int orient    # 0=H, 1=V (only for walls)


# =====================================================================
# Core game state — all C types, no Python objects
# =====================================================================
ctypedef struct GameState:
    int num_players
    int current_player
    int winner              # -1 = no winner

    int pawn_r[MAX_PLAYERS]
    int pawn_c[MAX_PLAYERS]
    int walls_remaining[MAX_PLAYERS]

    # Blocked edges (bool arrays)
    # h_blocked[r][c] = 1 means movement between (r,c) and (r+1,c) is blocked
    # v_blocked[r][c] = 1 means movement between (r,c) and (r,c+1) is blocked
    bint h_blocked[BOARD_SIZE][BOARD_SIZE]
    bint v_blocked[BOARD_SIZE][BOARD_SIZE]

    # Wall tracking for conflict detection
    bint wall_centers[WALL_GRID][WALL_GRID]
    bint walls_h[WALL_GRID][WALL_GRID]  # horizontal wall placed at (r,c)
    bint walls_v[WALL_GRID][WALL_GRID]  # vertical wall placed at (r,c)

    int num_walls_placed
    # Store placed walls for serialization/display
    int placed_r[MAX_WALLS_TOTAL]
    int placed_c[MAX_WALLS_TOTAL]
    int placed_o[MAX_WALLS_TOTAL]  # 0=H, 1=V

    # Move history
    int num_moves
    CMove history[MAX_MOVES]

    # Goal info (encoded)
    # goal_type: 0 = row goal, 1 = col goal
    # goal_value: the target row or col number
    int goal_type[MAX_PLAYERS]
    int goal_val[MAX_PLAYERS]


# =====================================================================
# Low-level C functions
# =====================================================================

cdef inline bint in_bounds(int r, int c) noexcept nogil:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

cdef inline bint can_step(GameState* gs, int r1, int c1, int r2, int c2) noexcept nogil:
    """Check if step from (r1,c1) to (r2,c2) is not blocked by wall."""
    if not in_bounds(r2, c2):
        return 0
    cdef int dr = r2 - r1
    cdef int dc = c2 - c1
    if dr == 1:
        return not gs.h_blocked[r1][c1]
    if dr == -1:
        return not gs.h_blocked[r2][c2]
    if dc == 1:
        return not gs.v_blocked[r1][c1]
    if dc == -1:
        return not gs.v_blocked[r2][c2]
    return 0

cdef inline bint reached_goal(GameState* gs, int player) noexcept nogil:
    if gs.goal_type[player] == 0:  # row goal
        return gs.pawn_r[player] == gs.goal_val[player]
    else:  # col goal
        return gs.pawn_c[player] == gs.goal_val[player]

cdef bint has_path(GameState* gs, int player) noexcept nogil:
    """BFS: can player reach their goal?"""
    cdef int start_r = gs.pawn_r[player]
    cdef int start_c = gs.pawn_c[player]
    cdef int gt = gs.goal_type[player]
    cdef int gv = gs.goal_val[player]

    cdef bint visited[BOARD_SIZE][BOARD_SIZE]
    memset(visited, 0, sizeof(visited))

    cdef int queue_r[BFS_QUEUE_SIZE]
    cdef int queue_c[BFS_QUEUE_SIZE]
    cdef int head = 0, tail = 0

    queue_r[tail] = start_r
    queue_c[tail] = start_c
    tail += 1
    visited[start_r][start_c] = 1

    cdef int r, c, nr, nc, d

    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1

        # Check goal
        if gt == 0 and r == gv:
            return 1
        if gt == 1 and c == gv:
            return 1

        # Expand neighbors
        for d in range(4):
            nr = r + DR[d]
            nc = c + DC[d]
            if in_bounds(nr, nc) and not visited[nr][nc] and can_step(gs, r, c, nr, nc):
                visited[nr][nc] = 1
                queue_r[tail] = nr
                queue_c[tail] = nc
                tail += 1

    return 0

cdef bint all_paths_ok(GameState* gs) noexcept nogil:
    cdef int p
    for p in range(gs.num_players):
        if not reached_goal(gs, p) and not has_path(gs, p):
            return 0
    return 1

cdef void add_wall_edges(GameState* gs, int r, int c, int orient) noexcept nogil:
    if orient == 0:  # H
        gs.h_blocked[r][c] = 1
        gs.h_blocked[r][c + 1] = 1
    else:  # V
        gs.v_blocked[r][c] = 1
        gs.v_blocked[r + 1][c] = 1

cdef void remove_wall_edges(GameState* gs, int r, int c, int orient) noexcept nogil:
    if orient == 0:  # H
        gs.h_blocked[r][c] = 0
        gs.h_blocked[r][c + 1] = 0
    else:  # V
        gs.v_blocked[r][c] = 0
        gs.v_blocked[r + 1][c] = 0

cdef bint wall_has_conflict(GameState* gs, int r, int c, int orient) noexcept nogil:
    if r < 0 or r >= WALL_GRID or c < 0 or c >= WALL_GRID:
        return 1
    if gs.wall_centers[r][c]:
        return 1
    if orient == 0:  # H
        if c > 0 and gs.walls_h[r][c - 1]:
            return 1
        if c < WALL_GRID - 1 and gs.walls_h[r][c + 1]:
            return 1
    else:  # V
        if r > 0 and gs.walls_v[r - 1][c]:
            return 1
        if r < WALL_GRID - 1 and gs.walls_v[r + 1][c]:
            return 1
    return 0

cdef bint is_valid_wall(GameState* gs, int r, int c, int orient) noexcept nogil:
    if wall_has_conflict(gs, r, c, orient):
        return 0
    add_wall_edges(gs, r, c, orient)
    cdef bint ok = all_paths_ok(gs)
    remove_wall_edges(gs, r, c, orient)
    return ok

cdef int get_pawn_moves(GameState* gs, int player, CMove* out) noexcept nogil:
    """Write legal pawn moves into out[], return count."""
    cdef int r = gs.pawn_r[player]
    cdef int c = gs.pawn_c[player]
    cdef int count = 0
    cdef int nr, nc, jr, jc, d, dd, p
    cdef bint occupied, straight_ok, jump_occ, diag_occ
    cdef int dr2, dc2

    for d in range(4):
        nr = r + DR[d]
        nc = c + DC[d]
        if not can_step(gs, r, c, nr, nc):
            continue

        # Check if occupied by any pawn
        occupied = 0
        for p in range(gs.num_players):
            if gs.pawn_r[p] == nr and gs.pawn_c[p] == nc:
                occupied = 1
                break

        if not occupied:
            out[count].action = MOVE_PAWN
            out[count].r = nr
            out[count].c = nc
            out[count].orient = 0
            count += 1
        else:
            # Try straight jump
            jr = nr + DR[d]
            jc = nc + DC[d]
            straight_ok = 0
            if in_bounds(jr, jc) and can_step(gs, nr, nc, jr, jc):
                # Check not occupied
                jump_occ = 0
                for p in range(gs.num_players):
                    if gs.pawn_r[p] == jr and gs.pawn_c[p] == jc:
                        jump_occ = 1
                        break
                if not jump_occ:
                    out[count].action = MOVE_PAWN
                    out[count].r = jr
                    out[count].c = jc
                    out[count].orient = 0
                    count += 1
                    straight_ok = 1

            if not straight_ok:
                # Diagonal jumps
                for dd in range(4):
                    if dd == d or (DR[dd] == -DR[d] and DC[dd] == -DC[d]):
                        continue
                    dr2 = nr + DR[dd]
                    dc2 = nc + DC[dd]
                    if in_bounds(dr2, dc2) and can_step(gs, nr, nc, dr2, dc2):
                        diag_occ = 0
                        for p in range(gs.num_players):
                            if gs.pawn_r[p] == dr2 and gs.pawn_c[p] == dc2:
                                diag_occ = 1
                                break
                        if not diag_occ:
                            out[count].action = MOVE_PAWN
                            out[count].r = dr2
                            out[count].c = dc2
                            out[count].orient = 0
                            count += 1
    return count

cdef int shortest_path_length(GameState* gs, int player) noexcept nogil:
    """BFS distance from player's pawn to their goal. Returns -1 if no path."""
    cdef int start_r = gs.pawn_r[player]
    cdef int start_c = gs.pawn_c[player]
    cdef int gt = gs.goal_type[player]
    cdef int gv = gs.goal_val[player]

    cdef int dist[BOARD_SIZE][BOARD_SIZE]
    memset(dist, -1, sizeof(dist))

    cdef int queue_r[BFS_QUEUE_SIZE]
    cdef int queue_c[BFS_QUEUE_SIZE]
    cdef int head = 0, tail = 0

    queue_r[tail] = start_r
    queue_c[tail] = start_c
    tail += 1
    dist[start_r][start_c] = 0

    cdef int r, c, nr, nc, d, cur_dist

    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        cur_dist = dist[r][c]

        if gt == 0 and r == gv:
            return cur_dist
        if gt == 1 and c == gv:
            return cur_dist

        for d in range(4):
            nr = r + DR[d]
            nc = c + DC[d]
            if in_bounds(nr, nc) and dist[nr][nc] == -1 and can_step(gs, r, c, nr, nc):
                dist[nr][nc] = cur_dist + 1
                queue_r[tail] = nr
                queue_c[tail] = nc
                tail += 1

    return -1


# =====================================================================
# Init helpers
# =====================================================================

cdef void init_game(GameState* gs, int num_players) noexcept nogil:
    memset(gs, 0, sizeof(GameState))
    gs.num_players = num_players
    gs.current_player = 0
    gs.winner = -1
    gs.num_walls_placed = 0
    gs.num_moves = 0

    if num_players == 2:
        gs.pawn_r[0] = 8; gs.pawn_c[0] = 4
        gs.pawn_r[1] = 0; gs.pawn_c[1] = 4
        gs.goal_type[0] = 0; gs.goal_val[0] = 0  # row 0
        gs.goal_type[1] = 0; gs.goal_val[1] = 8  # row 8
        gs.walls_remaining[0] = 10
        gs.walls_remaining[1] = 10
    else:  # 4 players
        gs.pawn_r[0] = 8; gs.pawn_c[0] = 4
        gs.pawn_r[1] = 0; gs.pawn_c[1] = 4
        gs.pawn_r[2] = 4; gs.pawn_c[2] = 0
        gs.pawn_r[3] = 4; gs.pawn_c[3] = 8
        gs.goal_type[0] = 0; gs.goal_val[0] = 0
        gs.goal_type[1] = 0; gs.goal_val[1] = 8
        gs.goal_type[2] = 1; gs.goal_val[2] = 8
        gs.goal_type[3] = 1; gs.goal_val[3] = 0
        for i in range(4):
            gs.walls_remaining[i] = 5


# =====================================================================
# Python wrapper class — same API as engine.QuoridorGame
# =====================================================================

cdef class FastQuoridorGame:
    cdef GameState state

    def __init__(self, int num_players=2):
        if num_players not in (2, 4):
            raise ValueError("num_players must be 2 or 4")
        init_game(&self.state, num_players)

    @property
    def num_players(self):
        return self.state.num_players

    @property
    def current_player(self):
        return self.state.current_player

    @current_player.setter
    def current_player(self, int val):
        self.state.current_player = val

    @property
    def winner(self):
        return self.state.winner if self.state.winner >= 0 else None

    @property
    def is_over(self):
        return self.state.winner >= 0

    @staticmethod
    def get_board_size():
        return 9

    @staticmethod
    def get_wall_grid():
        return 8

    @property
    def pawns(self):
        return [(self.state.pawn_r[i], self.state.pawn_c[i])
                for i in range(self.state.num_players)]

    @pawns.setter
    def pawns(self, val):
        for i, (r, c) in enumerate(val):
            self.state.pawn_r[i] = r
            self.state.pawn_c[i] = c

    @property
    def walls_remaining(self):
        return [self.state.walls_remaining[i]
                for i in range(self.state.num_players)]

    @walls_remaining.setter
    def walls_remaining(self, val):
        for i, w in enumerate(val):
            self.state.walls_remaining[i] = w

    @property
    def walls_placed(self):
        result = set()
        for i in range(self.state.num_walls_placed):
            o = "H" if self.state.placed_o[i] == 0 else "V"
            result.add((self.state.placed_r[i], self.state.placed_c[i], o))
        return result

    @property
    def move_history(self):
        result = []
        for i in range(self.state.num_moves):
            m = self.state.history[i]
            if m.action == MOVE_PAWN:
                result.append(("move", (m.r, m.c)))
            else:
                o = "H" if m.orient == 0 else "V"
                result.append(("wall", (m.r, m.c, o)))
        return result

    # Class-level constants for API compatibility with pure Python engine
    # (Can't use BOARD_SIZE/WALL_GRID as they clash with DEF constants)
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

    # ── Clone (fast memcpy) ──

    def clone(self):
        cdef FastQuoridorGame new = FastQuoridorGame.__new__(FastQuoridorGame)
        memcpy(&new.state, &self.state, sizeof(GameState))
        return new

    # ── Move generation ──

    def _get_pawn_moves(self, int player):
        cdef CMove moves[12]
        cdef int n = get_pawn_moves(&self.state, player, moves)
        return [(moves[i].r, moves[i].c) for i in range(n)]

    def get_legal_pawn_moves(self):
        cdef CMove moves[12]
        cdef int n = get_pawn_moves(&self.state, self.state.current_player, moves)
        return [("move", (moves[i].r, moves[i].c)) for i in range(n)]

    def get_legal_walls(self):
        if self.state.walls_remaining[self.state.current_player] <= 0:
            return []
        result = []
        cdef int r, c, o
        for r in range(WALL_GRID):
            for c in range(WALL_GRID):
                for o in range(2):
                    if is_valid_wall(&self.state, r, c, o):
                        orient = "H" if o == 0 else "V"
                        result.append(("wall", (r, c, orient)))
        return result

    def get_probable_walls(self, int radius=1):
        if self.state.walls_remaining[self.state.current_player] <= 0:
            return []

        cdef bint candidates[WALL_GRID][WALL_GRID]
        memset(candidates, 0, sizeof(candidates))

        cdef int r, c, dr, dc, wr, wc, i, p

        # Near pawns
        for p in range(self.state.num_players):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    wr = self.state.pawn_r[p] + dr
                    wc = self.state.pawn_c[p] + dc
                    if 0 <= wr < WALL_GRID and 0 <= wc < WALL_GRID:
                        candidates[wr][wc] = 1

        # Adjacent to placed walls
        for i in range(self.state.num_walls_placed):
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    wr = self.state.placed_r[i] + dr
                    wc = self.state.placed_c[i] + dc
                    if 0 <= wr < WALL_GRID and 0 <= wc < WALL_GRID:
                        candidates[wr][wc] = 1

        # On opponents' shortest paths
        cdef int cp = self.state.current_player
        # Use BFS to find path cells for opponents
        for p in range(self.state.num_players):
            if p == cp:
                continue
            self._add_path_candidates(&self.state, p, candidates)

        result = []
        cdef int o
        for r in range(WALL_GRID):
            for c in range(WALL_GRID):
                if candidates[r][c]:
                    for o in range(2):
                        if is_valid_wall(&self.state, r, c, o):
                            orient = "H" if o == 0 else "V"
                            result.append(("wall", (r, c, orient)))
        return result

    cdef void _add_path_candidates(self, GameState* gs, int player,
                                    bint candidates[WALL_GRID][WALL_GRID]) noexcept:
        """Add wall candidate positions along player's shortest path."""
        cdef int start_r = gs.pawn_r[player]
        cdef int start_c = gs.pawn_c[player]
        cdef int gt = gs.goal_type[player]
        cdef int gv = gs.goal_val[player]

        cdef int prev_r[BOARD_SIZE][BOARD_SIZE]
        cdef int prev_c[BOARD_SIZE][BOARD_SIZE]
        memset(prev_r, -1, sizeof(prev_r))
        memset(prev_c, -1, sizeof(prev_c))

        cdef int queue_r[BFS_QUEUE_SIZE]
        cdef int queue_c[BFS_QUEUE_SIZE]
        cdef int head = 0, tail = 0

        queue_r[tail] = start_r
        queue_c[tail] = start_c
        tail += 1
        prev_r[start_r][start_c] = start_r
        prev_c[start_r][start_c] = start_c

        cdef int r, c, nr, nc, d, wr, wc, dr, dc, pr, pc

        while head < tail:
            r = queue_r[head]
            c = queue_c[head]
            head += 1

            if (gt == 0 and r == gv) or (gt == 1 and c == gv):
                # Trace back path and add to candidates
                while not (r == start_r and c == start_c):
                    for dr in range(-1, 1):
                        for dc in range(-1, 1):
                            wr = r + dr
                            wc = c + dc
                            if 0 <= wr < WALL_GRID and 0 <= wc < WALL_GRID:
                                candidates[wr][wc] = 1
                    pr = prev_r[r][c]
                    pc = prev_c[r][c]
                    r = pr
                    c = pc
                return

            for d in range(4):
                nr = r + DR[d]
                nc = c + DC[d]
                if in_bounds(nr, nc) and prev_r[nr][nc] == -1 and can_step(gs, r, c, nr, nc):
                    prev_r[nr][nc] = r
                    prev_c[nr][nc] = c
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    def get_probable_moves(self):
        if self.state.winner >= 0:
            return []
        return self.get_legal_pawn_moves() + self.get_probable_walls()

    def get_legal_moves(self):
        if self.state.winner >= 0:
            return []
        return self.get_legal_pawn_moves() + self.get_legal_walls()

    def is_valid_wall(self, int r, int c, str o):
        cdef int orient = 0 if o == "H" else 1
        return is_valid_wall(&self.state, r, c, orient)

    # ── Make move ──

    def make_move(self, move):
        if self.state.winner >= 0:
            raise ValueError("Game is already over")

        action, data = move
        cdef int cp = self.state.current_player
        cdef CMove cmove
        cdef CMove pawn_moves[12]
        cdef int n, idx, orient
        cdef bint valid

        if action == "move":
            r, c = data
            # Validate
            n = get_pawn_moves(&self.state, cp, pawn_moves)
            valid = 0
            for i in range(n):
                if pawn_moves[i].r == r and pawn_moves[i].c == c:
                    valid = 1
                    break
            if not valid:
                raise ValueError(f"Illegal pawn move to {data}")

            self.state.pawn_r[cp] = r
            self.state.pawn_c[cp] = c

            cmove.action = MOVE_PAWN
            cmove.r = r
            cmove.c = c
            cmove.orient = 0

        elif action == "wall":
            r, c, o = data
            orient = 0 if o == "H" else 1

            if self.state.walls_remaining[cp] <= 0:
                raise ValueError("No walls remaining")
            if not is_valid_wall(&self.state, r, c, orient):
                raise ValueError(f"Illegal wall: {data}")

            # Place the wall
            add_wall_edges(&self.state, r, c, orient)
            self.state.wall_centers[r][c] = 1
            if orient == 0:
                self.state.walls_h[r][c] = 1
            else:
                self.state.walls_v[r][c] = 1

            idx = self.state.num_walls_placed
            self.state.placed_r[idx] = r
            self.state.placed_c[idx] = c
            self.state.placed_o[idx] = orient
            self.state.num_walls_placed += 1
            self.state.walls_remaining[cp] -= 1

            cmove.action = MOVE_WALL
            cmove.r = r
            cmove.c = c
            cmove.orient = orient
        else:
            raise ValueError(f"Unknown action: {action}")

        # Record move
        if self.state.num_moves < MAX_MOVES:
            self.state.history[self.state.num_moves] = cmove
            self.state.num_moves += 1

        # Check win
        if reached_goal(&self.state, cp):
            self.state.winner = cp
        else:
            self.state.current_player = (cp + 1) % self.state.num_players

    # ── Shortest path ──

    def shortest_path_length(self, int player):
        cdef int d = shortest_path_length(&self.state, player)
        return d if d >= 0 else None

    def shortest_path(self, int player):
        """BFS shortest path as list of (r,c) cells."""
        cdef int start_r = self.state.pawn_r[player]
        cdef int start_c = self.state.pawn_c[player]
        cdef int gt = self.state.goal_type[player]
        cdef int gv = self.state.goal_val[player]

        cdef int prev_r[BOARD_SIZE][BOARD_SIZE]
        cdef int prev_c[BOARD_SIZE][BOARD_SIZE]
        memset(prev_r, -1, sizeof(prev_r))
        prev_r[start_r][start_c] = start_r
        prev_c[start_r][start_c] = start_c

        cdef int queue_r[BFS_QUEUE_SIZE]
        cdef int queue_c[BFS_QUEUE_SIZE]
        cdef int head = 0, tail = 0

        queue_r[tail] = start_r
        queue_c[tail] = start_c
        tail += 1

        cdef int r, c, nr, nc, d

        while head < tail:
            r = queue_r[head]
            c = queue_c[head]
            head += 1

            if (gt == 0 and r == gv) or (gt == 1 and c == gv):
                # Reconstruct
                path = []
                while not (r == start_r and c == start_c):
                    path.append((r, c))
                    pr = prev_r[r][c]
                    pc = prev_c[r][c]
                    r = pr
                    c = pc
                path.append((start_r, start_c))
                path.reverse()
                return path

            for d in range(4):
                nr = r + DR[d]
                nc = c + DC[d]
                if in_bounds(nr, nc) and prev_r[nr][nc] == -1 and can_step(&self.state, r, c, nr, nc):
                    prev_r[nr][nc] = r
                    prev_c[nr][nc] = c
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

        return None

    # ── Tensor encoding ──

    def to_tensor(self):
        cdef int S = BOARD_SIZE
        cdef int cp = self.state.current_player
        cdef int opp, d_cp, d_opp, d
        cdef float max_w, max_w4

        if self.state.num_players == 2:
            planes = np.zeros((9, S, S), dtype=np.float32)
            opp = 1 - cp

            planes[0, self.state.pawn_r[cp], self.state.pawn_c[cp]] = 1.0
            planes[1, self.state.pawn_r[opp], self.state.pawn_c[opp]] = 1.0

            for r in range(S):
                for c in range(S):
                    if self.state.h_blocked[r][c]:
                        planes[2, r, c] = 1.0
                    if self.state.v_blocked[r][c]:
                        planes[3, r, c] = 1.0

            max_w = 10.0
            planes[4] = self.state.walls_remaining[cp] / max_w
            planes[5] = self.state.walls_remaining[opp] / max_w
            planes[6] = 1.0 if cp == 0 else 0.0

            # Distance to goal
            d_cp = shortest_path_length(&self.state, cp)
            d_opp = shortest_path_length(&self.state, opp)
            planes[7] = (d_cp if d_cp >= 0 else 8) / 8.0
            planes[8] = (d_opp if d_opp >= 0 else 8) / 8.0
        else:
            planes = np.zeros((14, S, S), dtype=np.float32)
            for i in range(4):
                p = (cp + i) % 4
                planes[i, self.state.pawn_r[p], self.state.pawn_c[p]] = 1.0

            for r in range(S):
                for c in range(S):
                    if self.state.h_blocked[r][c]:
                        planes[4, r, c] = 1.0
                    if self.state.v_blocked[r][c]:
                        planes[5, r, c] = 1.0

            max_w4 = 5.0
            for i in range(4):
                p = (cp + i) % 4
                planes[6 + i] = self.state.walls_remaining[p] / max_w4
                d = shortest_path_length(&self.state, p)
                planes[10 + i] = (d if d >= 0 else 8) / 8.0

        return planes

    # ── Serialization (compatible with pure Python engine) ──

    def to_dict(self):
        return {
            "num_players": self.state.num_players,
            "pawns": [[self.state.pawn_r[i], self.state.pawn_c[i]]
                      for i in range(self.state.num_players)],
            "walls_remaining": [self.state.walls_remaining[i]
                                for i in range(self.state.num_players)],
            "walls_placed": [[self.state.placed_r[i], self.state.placed_c[i],
                              "H" if self.state.placed_o[i] == 0 else "V"]
                             for i in range(self.state.num_walls_placed)],
            "current_player": self.state.current_player,
            "winner": self.state.winner if self.state.winner >= 0 else None,
            "move_history": self.move_history,
        }

    @classmethod
    def from_dict(cls, data):
        cdef FastQuoridorGame game = cls(data["num_players"])
        for i, p in enumerate(data["pawns"]):
            game.state.pawn_r[i] = p[0]
            game.state.pawn_c[i] = p[1]
        for i, w in enumerate(data["walls_remaining"]):
            game.state.walls_remaining[i] = w
        game.state.current_player = data["current_player"]
        game.state.winner = data["winner"] if data["winner"] is not None else -1
        for w in data["walls_placed"]:
            r, c, o = w[0], w[1], w[2]
            game._place_wall_raw(r, c, o)
        for a, d in data.get("move_history", []):
            if game.state.num_moves < MAX_MOVES:
                if a == "move":
                    game.state.history[game.state.num_moves].action = MOVE_PAWN
                    game.state.history[game.state.num_moves].r = d[0]
                    game.state.history[game.state.num_moves].c = d[1]
                else:
                    game.state.history[game.state.num_moves].action = MOVE_WALL
                    game.state.history[game.state.num_moves].r = d[0]
                    game.state.history[game.state.num_moves].c = d[1]
                    game.state.history[game.state.num_moves].orient = 0 if d[2] == "H" else 1
                game.state.num_moves += 1
        return game

    cdef void _place_wall_raw(self, int r, int c, str o):
        """Place a wall without validation (for deserialization)."""
        cdef int orient = 0 if o == "H" else 1
        cdef int idx
        add_wall_edges(&self.state, r, c, orient)
        self.state.wall_centers[r][c] = 1
        if orient == 0:
            self.state.walls_h[r][c] = 1
        else:
            self.state.walls_v[r][c] = 1
        idx = self.state.num_walls_placed
        self.state.placed_r[idx] = r
        self.state.placed_c[idx] = c
        self.state.placed_o[idx] = orient
        self.state.num_walls_placed += 1

    def __repr__(self):
        w = self.state.winner if self.state.winner >= 0 else None
        return (f"FastQuoridorGame(players={self.state.num_players}, "
                f"turn={self.state.current_player}, "
                f"moves={self.state.num_moves}, winner={w})")
