"""
Quick diagnostic: what does the trained model actually think?
Run: python diagnose.py
"""

import glob
import torch
import numpy as np
from game import QuoridorGame, MoveEncoder
from network import QuoridorNet
from config import Config


def load_latest():
    files = sorted(glob.glob("checkpoints/model_iter_*.pt"))
    if not files:
        print("No checkpoints found!")
        return None, None, None
    path = files[-1]
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**{k: v for k, v in data["config"].items()
                    if k in Config.__dataclass_fields__})
    net = QuoridorNet(
        input_planes=cfg.input_planes,
        num_blocks=cfg.num_res_blocks,
        num_channels=cfg.num_channels,
    )
    net.load_state_dict(data["model_state_dict"])
    net.eval()
    return net, cfg, data.get("iteration", "?")


def diagnose_position(net, game, label=""):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Current player: P{game.current_player}")
    print(f"  P0 at {game.pawns[0]}, P1 at {game.pawns[1]}")
    print(f"  Walls remaining: P0={game.walls_remaining[0]}, P1={game.walls_remaining[1]}")
    print(f"  P0 shortest path: {game.shortest_path_length(0)} moves")
    print(f"  P1 shortest path: {game.shortest_path_length(1)} moves")

    state = game.to_tensor()
    mask = MoveEncoder.legal_mask(game)
    policy, value = net.predict(state, mask)

    print(f"\n  Network value estimate: {value:.4f}")
    print(f"    (positive = good for current player)")

    # Decode top moves
    pawn_pos = game.pawns[game.current_player]
    move_probs = []
    for idx in range(MoveEncoder.TOTAL_ACTIONS):
        if mask[idx] > 0:
            move = MoveEncoder.decode(idx, pawn_pos)
            move_probs.append((policy[idx], move))

    move_probs.sort(reverse=True)

    # Separate pawn vs wall moves
    pawn_total = sum(p for p, m in move_probs if m[0] == "move")
    wall_total = sum(p for p, m in move_probs if m[0] == "wall")
    n_pawn = sum(1 for _, m in move_probs if m[0] == "move")
    n_wall = sum(1 for _, m in move_probs if m[0] == "wall")

    print(f"\n  Policy distribution:")
    print(f"    Pawn moves: {pawn_total*100:.1f}% across {n_pawn} moves")
    print(f"    Wall moves: {wall_total*100:.1f}% across {n_wall} moves")

    print(f"\n  Top 10 moves:")
    for prob, move in move_probs[:10]:
        action, data = move
        if action == "move":
            marker = "<-- PAWN"
        else:
            marker = ""
        print(f"    {prob*100:5.1f}%  {move}  {marker}")

    # Check if "walk forward" is in the top moves
    if game.current_player == 0:
        forward = (game.pawns[0][0] - 1, game.pawns[0][1])
    else:
        forward = (game.pawns[1][0] + 1, game.pawns[1][1])

    forward_move = ("move", forward)
    forward_prob = 0
    for p, m in move_probs:
        if m == forward_move:
            forward_prob = p
            break

    print(f"\n  'Walk forward' move {forward_move}: {forward_prob*100:.1f}%")
    if forward_prob < 0.1:
        print(f"    WARNING: model puts <10% on the obvious forward move")
    print()


def main():
    net, cfg, iteration = load_latest()
    if net is None:
        return

    print(f"Loaded iteration {iteration}")
    print(f"Network: {sum(p.numel() for p in net.parameters()):,} params")

    # Position 1: Starting position (P0 to move)
    g = QuoridorGame(2)
    diagnose_position(net, g, "Starting position (P0 to move)")

    # Position 2: After a few moves
    g2 = QuoridorGame(2)
    g2.make_move(("move", (7, 4)))
    g2.make_move(("move", (1, 4)))
    g2.make_move(("move", (6, 4)))
    g2.make_move(("move", (2, 4)))
    diagnose_position(net, g2, "After 4 moves (both advancing, P0 to move)")

    # Position 3: P0 almost winning
    g3 = QuoridorGame(2)
    g3.pawns = [(1, 4), (7, 4)]
    diagnose_position(net, g3, "P0 one step from goal")

    # Position 4: P1 almost winning, P0 to move
    g4 = QuoridorGame(2)
    g4.pawns = [(5, 4), (7, 4)]
    diagnose_position(net, g4, "P1 close to goal, P0 to move (should be urgent)")

    # Check training log
    try:
        import json
        with open("logs/training_log.json") as f:
            log = json.load(f)
        print(f"\n{'='*60}")
        print(f"  TRAINING LOG SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Iter':>5} {'Loss':>8} {'P.Loss':>8} {'V.Loss':>8} "
              f"{'Avg Moves':>10} {'P0 W':>5} {'P1 W':>5} {'Draw':>5}")
        print(f"  {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*5} {'-'*5} {'-'*5}")
        for e in log:
            print(f"  {e.get('iteration','?'):>5} "
                  f"{e.get('loss', 0):>8.4f} "
                  f"{e.get('policy_loss', 0):>8.4f} "
                  f"{e.get('value_loss', 0):>8.4f} "
                  f"{e.get('avg_game_length', 0):>10.0f} "
                  f"{e.get('p0_wins', 0):>5} "
                  f"{e.get('p1_wins', 0):>5} "
                  f"{e.get('truncated', 0):>5}")
    except Exception as e:
        print(f"\nCouldn't read training log: {e}")


if __name__ == "__main__":
    main()
