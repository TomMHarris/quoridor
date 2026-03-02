import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from game import QuoridorGame
from model import AlphaZeroNet, get_device
from mcts import MCTS

# --- Configuration ---
BOARD_SIZE = 9  # REAL SIZE
ITERATIONS = 50 
EPISODES = 5    # Fewer games per iter because 9x9 takes longer
MCTS_SIMS = 25
DEVICE = get_device()

def encode_action(action, size):
    # Map (type, r, c) to integer index
    t, r, c = action
    return t * (size * size) + r * size + c

def decode_action(idx, size):
    # Map integer index to (type, r, c)
    t = idx // (size * size)
    rem = idx % (size * size)
    r = rem // size
    c = rem % size
    return (t, r, c)

def visualize_progress(histories, filename="training_panel.png"):
    rows = int(np.ceil(len(histories) / 5))
    cols = 5
    if rows == 0: return

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    fig.patch.set_facecolor('#F5F5DC')
    if rows == 1: axes = np.array([axes])
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx >= len(histories):
            ax.axis('off')
            continue
            
        data = histories[idx]
        moves = data['moves']
        
        ax.set_facecolor('#F5F5DC')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Iter {data['iter']}", fontsize=10)
        
        # Grid
        for i in range(BOARD_SIZE + 1):
            ax.axvline(i - 0.5, color='#D2B48C', lw=1)
            ax.axhline(i - 0.5, color='#D2B48C', lw=1)
            
        # Draw Walls and Path
        curr = (0, BOARD_SIZE // 2)
        path_x, path_y = [curr[1]], [curr[0]]
        
        for i, m in enumerate(moves):
            t, r, c = m
            if t == 0: # Move
                if i % 2 == 0: # Only plot P1 path
                    path_x.append(c)
                    path_y.append(r)
            elif t == 1: # V-Wall
                ax.vlines(c - 0.5, r - 0.5, r + 1.5, colors='red', lw=2)
            elif t == 2: # H-Wall
                ax.hlines(r - 0.5, c - 0.5, c + 1.5, colors='red', lw=2)
                
        ax.plot(path_x, path_y, color='#5C4033', linewidth=2, marker='o', markersize=3)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def self_play(net):
    game = QuoridorGame(BOARD_SIZE)
    mcts = MCTS(net, DEVICE)
    training_data = []
    
    while True:
        for _ in range(MCTS_SIMS):
            mcts.search(game)
        
        s = str(game.get_state())
        # Map MCTS keys (flat indices) to full probability vector
        counts = np.zeros(3 * BOARD_SIZE * BOARD_SIZE)
        
        # Valid moves in this state
        valids = game.get_valid_moves()
        valid_indices = [encode_action(m, BOARD_SIZE) for m in valids]
        
        for idx in valid_indices:
            if (s, idx) in mcts.Nsa:
                counts[idx] = mcts.Nsa[(s, idx)]
        
        if np.sum(counts) == 0:
            probs = np.zeros_like(counts)
            probs[valid_indices] = 1 / len(valid_indices)
        else:
            probs = counts / np.sum(counts)
        
        training_data.append((game.get_state(), probs))
        
        action_idx = np.random.choice(len(probs), p=probs)
        action = decode_action(action_idx, BOARD_SIZE)
        
        res = game.step(action)
        if res is not None:
            return training_data, res

def train():
    net = AlphaZeroNet(BOARD_SIZE).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    eval_history = []

    print(f"🚀 Training 9x9 Quoridor on {DEVICE}...")

    for i in range(ITERATIONS):
        batch_states, batch_probs, batch_vs = [], [], []
        for _ in tqdm(range(EPISODES), desc=f"Iter {i+1}"):
            data, winner = self_play(net)
            for state, p in data:
                batch_states.append(state)
                batch_probs.append(p)
                batch_vs.append(winner)
        
        if len(batch_states) > 0:
            net.train()
            states = torch.FloatTensor(np.array(batch_states)).to(DEVICE)
            target_pis = torch.FloatTensor(np.array(batch_probs)).to(DEVICE)
            target_vs = torch.FloatTensor(np.array(batch_vs)).view(-1, 1).to(DEVICE)

            out_pi, out_v = net(states)
            loss_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
            loss_v = torch.sum((target_vs - out_v) ** 2) / target_vs.size(0)
            
            optimizer.zero_grad()
            (loss_pi + loss_v).backward()
            optimizer.step()

        # Evaluation
        net.eval()
        game = QuoridorGame(BOARD_SIZE)
        mcts = MCTS(net, DEVICE)
        moves = []
        while True:
            for _ in range(10): mcts.search(game) # Lower sims for speed
            
            # Select best move
            best_n = -1
            best_action = None
            valids = game.get_valid_moves()
            if not valids: break 
            
            s = str(game.get_state())
            for m in valids:
                idx = encode_action(m, BOARD_SIZE)
                n = mcts.Nsa.get((s, idx), 0)
                if n > best_n:
                    best_n = n
                    best_action = m
            
            # Fallback if no exploration
            if best_action is None: best_action = valids[0]
            
            res = game.step(best_action)
            moves.append(best_action)
            if res is not None: break
            
        eval_history.append({'iter': i+1, 'moves': moves})
        visualize_progress(eval_history)

    print("✅ Done! Check training_panel.png")

if __name__ == "__main__":
    train()