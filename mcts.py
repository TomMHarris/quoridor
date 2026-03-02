import math
import numpy as np
import torch

class MCTS:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.Qsa = {}  # Quality of move
        self.Nsa = {}  # Count of move
        self.Ns = {}   # Count of state visits
        self.Ps = {}   # Initial policy probability
        self.Es = {}   # End game status
        self.Vs = {}   # Valid moves

    def search(self, game, depth=0):
        # HARD LIMIT: Stop recursion if game is too long
        if depth > 50:
            return 0

        s = str(game.get_state())

        # Check if this state is terminal (Win/Loss/Draw)
        if s not in self.Es:
            if game.p1_pos[0] == game.size - 1: 
                self.Es[s] = 1      # P1 Won
            elif game.p2_pos[0] == 0: 
                self.Es[s] = -1     # P2 Won
            elif game.moves_count >= game.max_moves: 
                self.Es[s] = 0      # Draw
            else: 
                self.Es[s] = None   # Game continues

        # If terminal, return the result
        if self.Es[s] is not None:
            return -self.Es[s] if game.current_player == -1 else self.Es[s]

        if s not in self.Ps:
            # LEAF NODE: Predict with model
            board_tensor = torch.FloatTensor(game.get_state()).unsqueeze(0).unsqueeze(0).to(self.device)
            
            policy, v = self.model(board_tensor)
            policy = torch.exp(policy).data.cpu().numpy()[0]
            v = v.item()

            # Mask invalid moves
            valids = game.get_valid_moves()
            mask = np.zeros(game.size * game.size)
            for x, y in valids:
                mask[x * game.size + y] = 1
            
            policy = policy * mask
            sum_policy = np.sum(policy)
            
            if sum_policy > 0:
                policy /= sum_policy
            else:
                policy = mask / np.sum(mask)

            self.Ps[s] = policy
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        # SELECTION: Pick best move
        best_uct = -float('inf')
        best_move = None
        
        for x, y in self.Vs[s]:
            a = x * game.size + y
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + 1.0 * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = 1.0 * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
            
            if u > best_uct:
                best_uct = u
                best_move = (x, y)

        if best_move is None: return 0 # Should not happen

        # SIMULATION
        a = best_move[0] * game.size + best_move[1]
        next_game = game.clone()
        next_game.step(best_move)
        
        # Recurse with depth + 1
        v = self.search(next_game, depth=depth+1)

        # BACKPROPAGATION
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v