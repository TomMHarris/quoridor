import numpy as np
from collections import deque

class QuoridorGame:
    def __init__(self, size=9):
        self.size = size
        # Board: 0=Empty. We track players separately.
        # Walls: 0=None, 1=Wall. grids are (size-1)x(size-1)
        self.walls_v = np.zeros((size-1, size-1), dtype=int)
        self.walls_h = np.zeros((size-1, size-1), dtype=int)
        
        self.p1_pos = (0, size // 2)
        self.p2_pos = (size - 1, size // 2)
        self.p1_walls = 10
        self.p2_walls = 10
        
        self.current_player = 1
        self.moves_count = 0
        self.max_moves = 200

    def get_state(self):
        # Input to Neural Net: 4 Channels
        # 1. P1 Position (1s)
        # 2. P2 Position (1s)
        # 3. Vertical Walls
        # 4. Horizontal Walls
        
        state = np.zeros((4, self.size, self.size))
        
        # Channel 0: Current Player
        p_curr = self.p1_pos if self.current_player == 1 else self.p2_pos
        state[0, p_curr[0], p_curr[1]] = 1
        
        # Channel 1: Opponent
        p_opp = self.p2_pos if self.current_player == 1 else self.p1_pos
        state[1, p_opp[0], p_opp[1]] = 1
        
        # Channels 2 & 3: Walls (Pad to 9x9)
        # We pad the 8x8 wall grids to 9x9 for easier CNN processing
        state[2, :self.size-1, :self.size-1] = self.walls_v
        state[3, :self.size-1, :self.size-1] = self.walls_h
        
        if self.current_player == -1:
            # Flip board for P2 perspective
            state = np.flip(state, axis=1).copy() # Flip rows
            # Note: We don't swap channels 0/1 because 0 is ALWAYS "My pos"
            
        return state

    def is_connected(self, pos, target_row):
        # BFS to check if path exists
        q = deque([pos])
        visited = {pos}
        
        while q:
            r, c = q.popleft()
            if r == target_row: return True
            
            # Check neighbors
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if (nr, nc) not in visited:
                        # Check wall blocking
                        if self.can_step((r,c), (nr,nc)):
                            visited.add((nr, nc))
                            q.append((nr, nc))
        return False

    def can_step(self, curr, next_pos):
        r, c = curr
        nr, nc = next_pos
        
        if r == nr: # Horizontal move
            # Blocked if vertical wall at min(c, nc)
            wc = min(c, nc)
            if 0 <= wc < self.size-1 and 0 <= r < self.size-1:
                 if self.walls_v[r, wc] == 1: return False
            if 0 <= wc < self.size-1 and 0 <= r-1 < self.size-1:
                 if self.walls_v[r-1, wc] == 1: return False
                 
        elif c == nc: # Vertical move
            wr = min(r, nr)
            if 0 <= wr < self.size-1 and 0 <= c < self.size-1:
                if self.walls_h[wr, c] == 1: return False
            if 0 <= wr < self.size-1 and 0 <= c-1 < self.size-1:
                if self.walls_h[wr, c-1] == 1: return False
        
        return True

    def get_valid_moves(self):
        moves = []
        
        # 1. Player Movement
        curr = self.p1_pos if self.current_player == 1 else self.p2_pos
        opp = self.p2_pos if self.current_player == 1 else self.p1_pos
        
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = curr[0]+dr, curr[1]+dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self.can_step(curr, (nr, nc)):
                    if (nr, nc) == opp:
                        # Jump logic (simplified: jump over)
                        nnr, nnc = nr+dr, nc+dc
                        if 0 <= nnr < self.size and 0 <= nnc < self.size:
                             if self.can_step((nr,nc), (nnr,nnc)):
                                 moves.append((0, nnr, nnc))
                    else:
                        moves.append((0, nr, nc))

        # 2. Wall Placement (Type 1=Vertical, 2=Horizontal)
        walls_left = self.p1_walls if self.current_player == 1 else self.p2_walls
        if walls_left > 0:
            for r in range(self.size - 1):
                for c in range(self.size - 1):
                    # Try Vertical
                    if self.walls_v[r, c] == 0 and self.walls_h[r, c] == 0:
                         # Check overlap
                         if not (r > 0 and self.walls_v[r-1, c] == 1) and not (r < self.size-2 and self.walls_v[r+1, c] == 1):
                             # PATH CHECK
                             self.walls_v[r,c] = 1
                             if self.is_connected(self.p1_pos, self.size-1) and self.is_connected(self.p2_pos, 0):
                                 moves.append((1, r, c))
                             self.walls_v[r,c] = 0
                             
                    # Try Horizontal
                    if self.walls_h[r, c] == 0 and self.walls_v[r, c] == 0:
                         if not (c > 0 and self.walls_h[r, c-1] == 1) and not (c < self.size-2 and self.walls_h[r, c+1] == 1):
                             self.walls_h[r,c] = 1
                             if self.is_connected(self.p1_pos, self.size-1) and self.is_connected(self.p2_pos, 0):
                                 moves.append((2, r, c))
                             self.walls_h[r,c] = 0
        return moves

    def step(self, action):
        # Action format: (type, r, c)
        # type 0: move, 1: v-wall, 2: h-wall
        act_type, r, c = action
        
        if act_type == 0:
            if self.current_player == 1: self.p1_pos = (r, c)
            else: self.p2_pos = (r, c)
        elif act_type == 1:
            self.walls_v[r, c] = 1
            if self.current_player == 1: self.p1_walls -= 1
            else: self.p2_walls -= 1
        elif act_type == 2:
            self.walls_h[r, c] = 1
            if self.current_player == 1: self.p1_walls -= 1
            else: self.p2_walls -= 1

        self.moves_count += 1
        self.current_player *= -1
        
        if self.p1_pos[0] == self.size - 1: return 1
        if self.p2_pos[0] == 0: return -1
        if self.moves_count >= self.max_moves: return 0
        return None

    def clone(self):
        g = QuoridorGame(self.size)
        g.walls_v = self.walls_v.copy()
        g.walls_h = self.walls_h.copy()
        g.p1_pos = self.p1_pos
        g.p2_pos = self.p2_pos
        g.p1_walls = self.p1_walls
        g.p2_walls = self.p2_walls
        g.current_player = self.current_player
        g.moves_count = self.moves_count
        return g