# Quoridor AI

AlphaZero-style Quoridor AI with 2-player and 4-player support.

## Setup (macOS Apple Silicon)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CPU + MPS acceleration on Apple Silicon)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```

## Quick start

### Train
```bash
# Default settings (tuned for M1 MacBook Air)
python train.py

# Faster test run (fewer games, fewer simulations)
python train.py --iterations 5 --games 10 --simulations 50

# Full training with more compute
python train.py --iterations 50 --games 100 --simulations 400
```

### Play against the AI
```bash
python play.py                      # vs latest checkpoint
python play.py --simulations 400    # stronger (slower)
python play.py --ai-first           # AI moves first
```

### Run tests
```bash
python test_engine.py
```

## Architecture

```
engine.py      Game logic, rules, move encoding, state tensors
network.py     ResNet (4 blocks × 64 channels ≈ 200K params)
mcts.py        AlphaZero MCTS with neural network leaf evaluation
train.py       Self-play → train loop with replay buffer
play.py        Interactive terminal play vs trained model
config.py      All hyperparameters in one place
```

### Network
- **Input**: (C, 9, 9) tensor — pawn positions, wall edges, walls remaining
- **Backbone**: Conv → 4× ResBlock (conv-bn-relu-conv-bn + skip)
- **Policy head**: Conv 1×1 → FC → 140 logits (128 wall placements + 12 pawn moves)
- **Value head**: Conv 1×1 → FC → tanh → scalar ∈ [-1, 1]

### Training loop
Each iteration:
1. **Self-play**: N games using current network + MCTS (200 sims/move)
2. **Train**: SGD on replay buffer (cross-entropy policy + MSE value)
3. **Checkpoint**: save model weights + training log

### Performance expectations (M1 Air)

| Setting | Time per game | Time per iteration |
|---|---|---|
| 50 sims/move | ~30s | ~25 min (50 games) |
| 200 sims/move | ~2 min | ~1.5 hrs (50 games) |
| 400 sims/move | ~4 min | ~3.5 hrs (50 games) |

The main bottleneck is the Python game engine (BFS path validation for every
candidate wall). For serious training, porting `engine.py` to C/Cython would
give 50-100× speedup on the game simulation side.

## Scaling up

To train a stronger agent:
1. **More simulations** (400-800 per move)
2. **More self-play games** (100-200 per iteration)
3. **Bigger network** (8 blocks × 128 channels ≈ 1.5M params)
4. **Port engine to C++** for 50-100× faster game simulation
5. **Parallel self-play** across CPU cores

Edit `config.py` to adjust all hyperparameters.
