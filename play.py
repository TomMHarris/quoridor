"""
Play against the Quoridor AI in the terminal.

Usage:
    python play.py                          # play vs latest checkpoint
    python play.py --checkpoint path.pt     # play vs specific checkpoint
    python play.py --simulations 400        # stronger AI (slower)
    python play.py --human-first            # you move first (default)
    python play.py --ai-first               # AI moves first
"""

import argparse
import glob
import os

import numpy as np
import torch

from game import QuoridorGame, MoveEncoder
from network import QuoridorNet
from mcts import MCTS, select_action
from config import Config


def load_model(checkpoint_path: str, device: str):
    """Load network from checkpoint."""
    data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = Config(**{k: v for k, v in data["config"].items()
                    if k in Config.__dataclass_fields__})
    net = QuoridorNet(
        input_planes=cfg.input_planes,
        num_blocks=cfg.num_res_blocks,
        num_channels=cfg.num_channels,
    ).to(device)
    net.load_state_dict(data["model_state_dict"])
    net.eval()
    return net, cfg


def find_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> str:
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "model_iter_*.pt")))
    if not files:
        raise FileNotFoundError(
            f"No checkpoints in {checkpoint_dir}/. Run train.py first.")
    return files[-1]


def parse_human_move(text: str, game: QuoridorGame):
    """
    Parse human input:
        Pawn:  'e5' or '4 4' or 'move 4 4'
        Wall:  'e5h' or 'wall 4 4 H' or '4 4 h'
    """
    text = text.strip().lower()

    if not text:
        return None

    # Try algebraic: letter + digit + optional h/v
    if len(text) in (2, 3) and text[0].isalpha() and text[1].isdigit():
        col = ord(text[0]) - ord('a')
        row = int(text[1]) - 1
        if len(text) == 3 and text[2] in ('h', 'v'):
            return ("wall", (row, col, text[2].upper()))
        return ("move", (row, col))

    parts = text.replace(",", " ").split()

    # Strip leading 'move' or 'wall'
    action = None
    if parts[0] in ("move", "m"):
        action = "move"
        parts = parts[1:]
    elif parts[0] in ("wall", "w"):
        action = "wall"
        parts = parts[1:]

    if len(parts) == 2 and action != "wall":
        r, c = int(parts[0]), int(parts[1])
        return ("move", (r, c))
    elif len(parts) == 3:
        r, c = int(parts[0]), int(parts[1])
        o = parts[2].upper()
        return ("wall", (r, c, o))

    return None


def play(checkpoint_path: str = None, simulations: int = 200,
         human_player: int = 0, device: str = "auto"):

    if device == "auto":
        device = Config().resolve_device()

    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    print(f"Loading: {checkpoint_path}")
    print(f"Device:  {device}")
    net, cfg = load_model(checkpoint_path, device)

    mcts = MCTS(
        network=net, device=device,
        num_simulations=simulations,
        c_puct=cfg.c_puct,
    )

    game = QuoridorGame(2)
    ai_player = 1 - human_player

    print(f"\nYou are Player {human_player} "
          f"({'top→bottom' if human_player == 1 else 'bottom→top'})")
    print("Enter moves as: row col  (pawn) or  row col H/V  (wall)")
    print("Example: '7 3' to move pawn, '3 4 H' to place horizontal wall")
    print("Or algebraic: 'e5' for pawn, 'e3h' for wall")
    print("Type 'quit' to exit, 'undo' to take back.\n")

    while not game.is_over:
        print(game)
        print()

        if game.current_player == human_player:
            # Human turn
            while True:
                try:
                    text = input(f"Your move (P{human_player}): ")
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    return

                if text.strip().lower() in ("quit", "q", "exit"):
                    print("Goodbye!")
                    return

                move = parse_human_move(text, game)
                if move is None:
                    print("  Could not parse. Try: row col  or  row col H/V")
                    continue

                if move not in game.get_legal_moves():
                    print(f"  Illegal move: {move}")
                    continue

                game.make_move(move)
                break
        else:
            # AI turn
            print("AI thinking...")
            action_probs = mcts.search(game, add_noise=False)
            action_idx = select_action(action_probs, temperature=0.1)
            pawn_pos = game.pawns[ai_player]
            move = MoveEncoder.decode(action_idx, pawn_pos)
            game.make_move(move)
            print(f"  AI plays: {move}")

        print()

    print(game)
    if game.winner == human_player:
        print("\n🎉 You win!")
    elif game.winner == ai_player:
        print("\n🤖 AI wins!")
    else:
        print("\nDraw!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Quoridor vs AI")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--ai-first", action="store_true")
    parser.add_argument("--human-first", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    human = 1 if args.ai_first else 0
    play(
        checkpoint_path=args.checkpoint,
        simulations=args.simulations,
        human_player=human,
        device=args.device,
    )