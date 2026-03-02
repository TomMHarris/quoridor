"""Configuration for Quoridor AI training."""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    # Game
    num_players: int = 2

    # Network
    num_res_blocks: int = 8
    num_channels: int = 128
    input_planes: int = 9       # from engine.py to_tensor (2-player)
    action_size: int = 140      # from engine.py MoveEncoder

    # MCTS
    num_simulations: int = 64
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 15     # temp=1 for first N moves, then 0.1
    max_considered_actions: int = 16  # Gumbel Sequential Halving top-K

    # Expert pre-training (Phase 1)
    warmup_games: int = 2000
    warmup_mcts_rollouts: int = 300
    warmup_mcts_fraction: float = 0.6   # fraction of games using MCTSExpert
    warmup_epochs: int = 15
    warmup_lr: float = 1e-3
    warmup_batch_size: int = 128

    # Self-play (Phase 2)
    num_iterations: int = 40
    games_per_iteration: int = 30
    max_game_length: int = 150

    # Training
    train_batch_size: int = 128
    train_lr: float = 3e-4
    train_weight_decay: float = 1e-4
    train_epochs_per_iter: int = 2
    replay_buffer_size: int = 50_000
    min_replay_for_training: int = 1000
    gradient_clip: float = 1.0
    kl_threshold: float = 0.02

    # Evaluation
    eval_games: int = 20
    eval_threshold: float = 0.55

    # System
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_every: int = 5

    # Early stopping
    max_stale_iterations: int = 10      # halt if no policy loss improvement
    min_win_rate_abort: float = 0.40    # halt if win rate drops below this
    consecutive_fails_abort: int = 3    # for N consecutive checkpoints
    max_avg_game_length: int = 120      # warn if games are too long

    @property
    def device(self) -> torch.device:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
