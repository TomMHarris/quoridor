"""
Visualize Quoridor AI training progression.

Generates a panel of final board positions from training games,
showing how the AI's play evolves over iterations.

Usage:
    python visualize.py                         # from saved training data
    python visualize.py --demo                  # generate demo with random games
    python visualize.py --output poster.pdf     # save as PDF (print quality)

Requires: pip install matplotlib
"""

import argparse
import glob
import json
import os
import random
from typing import List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
except ImportError:
    print("This script requires matplotlib: pip install matplotlib")
    raise

from engine import QuoridorGame


# ===========================================================================
# Quoridor palette -- warm wood tones matching the physical game
# ===========================================================================

COLORS = {
    # Background
    "bg":             "#FFFFFF",
    "board_bg":       "#F5F0E8",
    "board_border":   "#D6CEBF",
    "grid":           "#DDD7CA",

    # Walls -- natural beech wood
    "wall":           "#C4A56E",
    "wall_edge":      "#A8894F",
    "wall_shadow":    "#00000012",

    # Pawns -- matching the physical pieces
    "p0":             "#F0DDB8",       # light beech / cream
    "p0_edge":        "#CEBB8E",
    "p0_highlight":   "#FFF8EC",
    "p1":             "#5C3A1E",       # dark walnut / brown
    "p1_edge":        "#3D2510",
    "p1_highlight":   "#7A5535",

    # Paths -- translucent versions of pawn colors
    "path_p0":        "#E8D0A0",
    "path_p1":        "#5C3A1E",

    # Text
    "text":           "#2C2C2C",
    "text_light":     "#9C9688",
    "text_mid":       "#6B665C",
    "accent":         "#C03020",
}


# ===========================================================================
# Reconstruct pawn trails from move history
# ===========================================================================

def _extract_pawn_trail(game: QuoridorGame, player: int) -> list:
    """Replay the game to extract the trail of positions for a given pawn."""
    setup = game.SETUP[game.num_players]
    replay = QuoridorGame(game.num_players)
    trails = {p: [setup["starts"][p]] for p in range(game.num_players)}

    for move in game.move_history:
        cp = replay.current_player
        action, data = move
        if action == "move":
            trails[cp].append(data)
        replay.make_move(move)

    return trails.get(player, [setup["starts"][player]])


# ===========================================================================
# Board rendering
# ===========================================================================

def draw_board(ax, game: QuoridorGame, cell_size: float = 1.0,
               label: str = "", sublabel: str = ""):
    """Draw a single Quoridor board state on the given axes."""
    S = game.BOARD_SIZE
    total = S * cell_size

    ax.set_xlim(-0.12, total + 0.12)
    ax.set_ylim(-0.7, total + 0.12)
    ax.set_aspect("equal")
    ax.axis("off")

    # Board background
    board_rect = patches.FancyBboxPatch(
        (-0.04, -0.04), total + 0.08, total + 0.08,
        boxstyle="round,pad=0.03",
        facecolor=COLORS["board_bg"],
        edgecolor=COLORS["board_border"],
        linewidth=0.6,
    )
    ax.add_patch(board_rect)

    # Grid grooves
    for i in range(1, S):
        ax.plot([i * cell_size, i * cell_size], [0, total],
                color=COLORS["grid"], linewidth=0.35, zorder=1)
        ax.plot([0, total], [i * cell_size, i * cell_size],
                color=COLORS["grid"], linewidth=0.35, zorder=1)

    # -- Pawn trailing paths (always shown) --
    for p in range(min(game.num_players, 2)):
        trail = _extract_pawn_trail(game, p)
        if trail and len(trail) > 1:
            path_color = COLORS[f"path_p{p}"]
            xs = [c * cell_size + cell_size / 2 for _, c in trail]
            ys = [(S - 1 - r) * cell_size + cell_size / 2 for r, _ in trail]
            ax.plot(xs, ys, color=path_color,
                    alpha=0.22 if p == 0 else 0.16,
                    linewidth=cell_size * 3.2,
                    solid_capstyle="round", solid_joinstyle="round",
                    zorder=2)

    # -- Walls -- warm wood blocks --
    wall_width = cell_size * 0.16

    for wr, wc, orient in game.walls_placed:
        if orient == "H":
            x = wc * cell_size
            y = (S - 1 - wr) * cell_size
            w = 2 * cell_size
            h = wall_width

            # Shadow
            ax.add_patch(patches.FancyBboxPatch(
                (x + 0.02, y - h / 2 - 0.02), w, h,
                boxstyle="round,pad=0.02",
                facecolor=COLORS["wall_shadow"], edgecolor="none", zorder=3))

            # Body
            ax.add_patch(patches.FancyBboxPatch(
                (x, y - h / 2), w, h,
                boxstyle="round,pad=0.02",
                facecolor=COLORS["wall"], edgecolor=COLORS["wall_edge"],
                linewidth=0.4, zorder=4))

            # Highlight
            ax.add_patch(patches.FancyBboxPatch(
                (x + 0.06, y - h / 2 + h * 0.5), w - 0.12, h * 0.25,
                boxstyle="round,pad=0.01",
                facecolor="#FFFFFF20", edgecolor="none", zorder=5))

        else:  # V
            x = (wc + 1) * cell_size
            y = (S - 2 - wr) * cell_size
            w = wall_width
            h = 2 * cell_size

            ax.add_patch(patches.FancyBboxPatch(
                (x - w / 2 + 0.02, y - 0.02), w, h,
                boxstyle="round,pad=0.02",
                facecolor=COLORS["wall_shadow"], edgecolor="none", zorder=3))

            ax.add_patch(patches.FancyBboxPatch(
                (x - w / 2, y), w, h,
                boxstyle="round,pad=0.02",
                facecolor=COLORS["wall"], edgecolor=COLORS["wall_edge"],
                linewidth=0.4, zorder=4))

            ax.add_patch(patches.FancyBboxPatch(
                (x - w / 2 + w * 0.5, y + 0.06), w * 0.25, h - 0.12,
                boxstyle="round,pad=0.01",
                facecolor="#FFFFFF20", edgecolor="none", zorder=5))

    # -- Pawns -- rounded wooden pieces with 3D shading --
    for p in range(game.num_players):
        r, c = game.pawns[p]
        cx = c * cell_size + cell_size / 2
        cy = (S - 1 - r) * cell_size + cell_size / 2
        radius = cell_size * 0.30
        pidx = min(p, 1)

        base = COLORS[f"p{pidx}"]
        edge = COLORS[f"p{pidx}_edge"]
        hi = COLORS[f"p{pidx}_highlight"]

        # Shadow beneath
        ax.add_patch(plt.Circle(
            (cx + 0.03, cy - 0.03), radius * 1.1,
            color="#00000015", zorder=6))

        # Main body
        ax.add_patch(plt.Circle(
            (cx, cy), radius,
            facecolor=base, edgecolor=edge,
            linewidth=0.6, zorder=7))

        # Dome highlight
        ax.add_patch(plt.Circle(
            (cx - radius * 0.15, cy + radius * 0.15), radius * 0.35,
            facecolor=hi, edgecolor="none",
            alpha=0.5, zorder=8))

        # Specular
        ax.add_patch(plt.Circle(
            (cx - radius * 0.08, cy + radius * 0.22), radius * 0.1,
            facecolor="#FFFFFF",  edgecolor="none",
            alpha=0.45 if pidx == 0 else 0.3, zorder=9))

    # -- Label --
    if label or sublabel:
        combined = f"{label}   {sublabel}" if sublabel else label
        ax.text(total / 2, -0.3, combined,
                ha="center", va="top",
                fontsize=4.0,
                color=COLORS["text_mid"],
                fontfamily="serif",
                fontstyle="italic")


# ===========================================================================
# Panel layout
# ===========================================================================

def create_panel(games_by_iteration: dict, output_path: str = "training_panel.png",
                 title: str = "quoridor",
                 subtitle: str = "",
                 cols: int = 10, max_games: int = 50):
    """Create the main visualization panel."""

    all_games = []
    iterations = sorted(games_by_iteration.keys())

    for it in iterations:
        games = games_by_iteration[it]
        finished = [g for g in games if g["game"].winner is not None]
        truncated = [g for g in games if g["game"].winner is None]
        selected = (finished + truncated)[:4]
        for g in selected:
            all_games.append({"iteration": it, **g})

    if len(all_games) > max_games:
        step = len(all_games) / max_games
        all_games = [all_games[int(i * step)] for i in range(max_games)]

    n = len(all_games)
    if n == 0:
        print("No games to visualize!")
        return

    rows = (n + cols - 1) // cols
    actual_cols = min(n, cols)

    board_inches = 1.35
    fig_w = actual_cols * board_inches + 1.0
    fig_h = rows * (board_inches + 0.12) + 2.4

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=COLORS["bg"])

    # -- Title: lowercase serif, like the Gigamic box --
    fig.text(0.5, 1 - 0.4 / fig_h, title,
             ha="center", va="top",
             fontsize=24, fontweight="300",
             color=COLORS["text"],
             fontfamily="serif")

    # Thin red accent line beneath title
    line_y = 1 - 0.7 / fig_h
    fig.add_artist(plt.Line2D(
        [0.38, 0.62], [line_y, line_y],
        color=COLORS["accent"], linewidth=0.8,
        transform=fig.transFigure, clip_on=False))

    # Subtitle
    n_iters = len(iterations)
    sub = subtitle or f"learning to play  |  {n} games across {n_iters} iterations"
    fig.text(0.5, 1 - 0.95 / fig_h, sub,
             ha="center", va="top",
             fontsize=7,
             color=COLORS["text_light"],
             fontfamily="serif",
             fontstyle="italic")

    # -- Board grid --
    left_margin = 0.5 / fig_w
    right_margin = 0.5 / fig_w
    top_margin = 1.4 / fig_h
    bottom_margin = 0.8 / fig_h

    usable_w = 1 - left_margin - right_margin
    usable_h = 1 - top_margin - bottom_margin
    cell_w = usable_w / actual_cols
    cell_h = usable_h / rows

    for idx, gdata in enumerate(all_games):
        row_idx = idx // actual_cols
        col_idx = idx % actual_cols

        x = left_margin + col_idx * cell_w
        y = 1 - top_margin - (row_idx + 1) * cell_h

        pad = 0.006
        ax = fig.add_axes([x + pad, y + pad, cell_w - 2 * pad, cell_h - 2 * pad])

        game = gdata["game"]
        it = gdata["iteration"]
        n_moves = gdata["num_moves"]
        winner = game.winner

        label = f"iter {it}"
        if winner is not None:
            result = "light" if winner == 0 else "dark"
            sublabel = f"{n_moves} moves \u2014 {result} wins"
        else:
            sublabel = f"{n_moves} moves \u2014 draw"

        draw_board(ax, game, label=label, sublabel=sublabel)

    # -- Footer --
    fig.text(0.5, 0.22 / fig_h,
             "early \u2192 late   |   "
             "walls = wood bars   |   "
             "light vs dark   |   "
             "trails show path taken",
             ha="center", va="bottom",
             fontsize=5,
             color=COLORS["text_light"],
             fontfamily="serif",
             fontstyle="italic")

    plt.savefig(output_path, dpi=220, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")


# ===========================================================================
# Data loading
# ===========================================================================

def load_training_games(log_dir: str = "logs") -> dict:
    """Load saved game histories from training."""
    games_by_iteration = {}
    files = sorted(glob.glob(os.path.join(log_dir, "games_iter_*.json")))

    if not files:
        print(f"No game files found in {log_dir}/")
        print("Run training first, or use --demo for sample data.")
        return {}

    for f in files:
        basename = os.path.basename(f)
        it = int(basename.split("_")[-1].split(".")[0])
        with open(f) as fh:
            game_stats = json.load(fh)

        games = []
        for gs in game_stats:
            if "final_state" in gs:
                game = QuoridorGame.from_dict(gs["final_state"])
                games.append({
                    "game": game,
                    "num_moves": gs["num_moves"],
                    "winner": gs["winner"],
                })
        if games:
            games_by_iteration[it] = games

    return games_by_iteration


def generate_demo_games(n_iterations: int = 10,
                        games_per_iter: int = 5) -> dict:
    """Generate random games for demo/testing the visualization."""
    games_by_iteration = {}

    for it in range(1, n_iterations + 1):
        games = []
        for _ in range(games_per_iter):
            g = QuoridorGame(2)
            move_count = 0
            max_moves = max(20, 200 - it * 15)

            while not g.is_over and move_count < max_moves:
                legal = g.get_legal_moves()
                pawn_moves = [m for m in legal if m[0] == "move"]
                pawn_prob = 0.5 + it * 0.04
                if random.random() < pawn_prob and pawn_moves:
                    g.make_move(random.choice(pawn_moves))
                else:
                    g.make_move(random.choice(legal))
                move_count += 1

            games.append({
                "game": g,
                "num_moves": move_count,
                "winner": g.winner if g.winner is not None else -1,
            })

        games_by_iteration[it] = games

    return games_by_iteration


# ===========================================================================
# Single board
# ===========================================================================

def visualize_single_game(game: QuoridorGame, output_path: str = "board.png"):
    """Render a single board state."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor=COLORS["bg"])
    draw_board(ax, game, cell_size=1.0)
    plt.savefig(output_path, dpi=220, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"Saved: {output_path}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Quoridor AI training")
    parser.add_argument("--demo", action="store_true",
                        help="Generate demo with random games")
    parser.add_argument("--output", type=str, default="training_panel.png",
                        help="Output file path (.png or .pdf)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory with training game logs")
    parser.add_argument("--cols", type=int, default=10,
                        help="Columns in the panel")
    parser.add_argument("--max-games", type=int, default=50,
                        help="Maximum boards to show")
    parser.add_argument("--title", type=str, default="quoridor")
    parser.add_argument("--subtitle", type=str, default="")
    args = parser.parse_args()

    if args.demo:
        print("Generating demo games...")
        games = generate_demo_games(n_iterations=10, games_per_iter=5)
    else:
        games = load_training_games(args.log_dir)

    if games:
        create_panel(
            games,
            output_path=args.output,
            title=args.title,
            subtitle=args.subtitle,
            cols=args.cols,
            max_games=args.max_games,
        )