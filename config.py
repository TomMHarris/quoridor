"""
Configuration for Quoridor AI training.
Tuned for Apple Silicon M1 MacBook Air (8-16GB RAM).
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Game ──────────────────────────────────────────────────────────
    num_players: int = 2

    # ── Network ───────────────────────────────────────────────────────
    num_res_blocks: int = 4        # ResNet depth (AlphaZero used 19-40)
    num_channels: int = 64         # Filter width (AlphaZero used 256)
    # Input planes: 7 for 2-player, 10 for 4-player
    # Determined automatically from num_players

    # ── MCTS ──────────────────────────────────────────────────────────
    num_simulations: int = 200     # per move (AlphaZero: 800)
    c_puct: float = 1.5            # exploration constant
    dirichlet_alpha: float = 0.3   # noise at root for exploration
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 20    # use temp=1 for first N moves, then temp->0
    use_gumbel: bool = False       # use Gumbel MCTS (better with fewer sims)

    # ── Self-play ─────────────────────────────────────────────────────
    num_self_play_games: int = 50  # games per iteration
    max_game_length: int = 200     # real Quoridor always ends; this is just a safety net
    num_parallel_games: int = 4    # batch self-play (limited by RAM)

    # ── Training ──────────────────────────────────────────────────────
    num_iterations: int = 30       # total train iterations
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 4            # passes over replay buffer per iteration
    replay_buffer_size: int = 20_000
    min_replay_size: int = 200     # start training after this many samples

    # ── System ────────────────────────────────────────────────────────
    device: str = "auto"           # "auto" picks mps > cuda > cpu
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 5            # save checkpoint every N iterations

    @property
    def input_planes(self) -> int:
        return 9 if self.num_players == 2 else 14

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
