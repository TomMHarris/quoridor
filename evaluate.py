"""
Evaluate trained Quoridor AI models against various opponents.

Usage:
    python evaluate.py                          # Eval latest checkpoint
    python evaluate.py --checkpoint model_best  # Eval specific checkpoint
    python evaluate.py --games 50               # More games for accuracy
"""

import argparse
import os
import random
import time

import numpy as np
import torch

from engine import QuoridorGame, MoveEncoder
from network import QuoridorNet
from mcts import GumbelMCTS, select_action
from expert import PathFollower, HeuristicExpert, MCTSExpert
from config import Config


def load_model(checkpoint_name: str, config: Config, device: torch.device):
    """Load a model from checkpoints directory."""
    path = os.path.join(config.checkpoint_dir, f"{checkpoint_name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data = torch.load(path, map_location=device, weights_only=False)
    net = QuoridorNet(config.input_planes, config.num_res_blocks,
                      config.num_channels, config.action_size)
    net.load_state_dict(data["model_state"])
    net.to(device)
    net.eval()
    return net


def play_game(net, device, config, opponent, net_player=0,
              num_sims=64, verbose=False):
    """Play one game. Returns (winner, num_moves, game)."""
    mcts = GumbelMCTS(net, device, config)
    game = QuoridorGame(2)

    for step in range(config.max_game_length):
        if game.is_over:
            break

        if game.current_player == net_player:
            probs = mcts.search(game, num_simulations=num_sims, add_noise=False)
            action = select_action(probs, temperature=0.1)
            pos = game.pawns[game.current_player]
            move = MoveEncoder.decode(action, pos)
            mask = MoveEncoder.legal_mask(game)
            if mask[action] < 0.5:
                move = random.choice(game.get_legal_moves())
        else:
            move = opponent.choose_move(game)

        if verbose and step < 10:
            print(f"  Move {step}: P{game.current_player} -> {move}")

        game.make_move(move)

    return game.winner, len(game.move_history), game


def evaluate_vs_opponent(net, device, config, opponent, opponent_name,
                         num_games=20, num_sims=64):
    """Full evaluation against an opponent."""
    wins = 0
    losses = 0
    draws = 0
    total_moves = []

    print(f"\n  vs {opponent_name} ({num_games} games, {num_sims} sims):")

    for g_idx in range(num_games):
        net_player = g_idx % 2
        winner, n_moves, _ = play_game(
            net, device, config, opponent,
            net_player=net_player, num_sims=num_sims
        )
        total_moves.append(n_moves)

        if winner == net_player:
            wins += 1
        elif winner is not None:
            losses += 1
        else:
            draws += 1

        if (g_idx + 1) % 10 == 0:
            print(f"    {g_idx+1}/{num_games}: "
                  f"W={wins} L={losses} D={draws}")

    wr = wins / num_games
    avg_moves = np.mean(total_moves)
    print(f"    Final: {wins}W {losses}L {draws}D "
          f"(win rate: {wr:.0%}, avg moves: {avg_moves:.0f})")

    return {"wins": wins, "losses": losses, "draws": draws,
            "win_rate": wr, "avg_moves": avg_moves}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quoridor AI")
    parser.add_argument("--checkpoint", type=str, default="model_best",
                        help="Checkpoint name (without .pt)")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per opponent")
    parser.add_argument("--sims", type=int, default=64,
                        help="MCTS simulations per move")
    args = parser.parse_args()

    config = Config()
    device = config.device
    print(f"Device: {device}")

    # Try to load model
    try:
        net = load_model(args.checkpoint, config, device)
    except FileNotFoundError:
        # Try warmup model as fallback
        try:
            net = load_model("model_warmup", config, device)
            print("Using warmup model (no best model found)")
        except FileNotFoundError:
            print("ERROR: No trained model found in checkpoints/")
            print("Run 'python train.py' first.")
            return

    print(f"\nLoaded: {args.checkpoint}")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters: {n_params:,}")

    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)

    results = {}

    # vs PathFollower
    results["PathFollower"] = evaluate_vs_opponent(
        net, device, config, PathFollower(),
        "PathFollower", args.games, args.sims
    )

    # vs HeuristicExpert
    results["HeuristicExpert"] = evaluate_vs_opponent(
        net, device, config, HeuristicExpert(),
        "HeuristicExpert", args.games, args.sims
    )

    # vs MCTSExpert (fewer games — it's slow)
    mcts_games = max(4, args.games // 4)
    results["MCTSExpert"] = evaluate_vs_opponent(
        net, device, config, MCTSExpert(num_rollouts=100),
        "MCTSExpert(100)", mcts_games, args.sims
    )

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, r in results.items():
        print(f"  {name:20s}: {r['win_rate']:.0%} "
              f"({r['wins']}W/{r['losses']}L/{r['draws']}D)")


if __name__ == "__main__":
    main()
