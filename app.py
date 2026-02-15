"""
Quoridor Web App — play against the trained AI in your browser.

Usage:
    python app.py                          # default settings
    python app.py --simulations 200        # stronger AI
    python app.py --port 8080              # custom port

Opens at http://localhost:5000
"""

import argparse
import glob
import json
import os

from flask import Flask, jsonify, request, send_from_directory
import torch

from game import QuoridorGame, MoveEncoder
from network import QuoridorNet
from mcts import MCTS, select_action
from config import Config

app = Flask(__name__, static_folder="web", static_url_path="")

# Global state
game_state = None
ai_mcts = None
ai_network = None
ai_device = None
ai_player = 1  # AI plays as dark (P1) by default


def load_model(checkpoint_dir="checkpoints", device="auto"):
    """Load the latest trained model."""
    cfg = Config()
    if device == "auto":
        device = cfg.resolve_device()

    files = sorted(glob.glob(os.path.join(checkpoint_dir, "model_iter_*.pt")))
    if not files:
        raise FileNotFoundError("No checkpoints found. Run training first.")

    path = files[-1]
    print(f"Loading model: {path}")
    data = torch.load(path, map_location=device, weights_only=False)

    cfg_data = data.get("config", {})
    cfg = Config(**{k: v for k, v in cfg_data.items()
                    if k in Config.__dataclass_fields__})

    net = QuoridorNet(
        input_planes=cfg.input_planes,
        num_blocks=cfg.num_res_blocks,
        num_channels=cfg.num_channels,
    ).to(device)
    net.load_state_dict(data["model_state_dict"])
    net.eval()

    print(f"Loaded iteration {data.get('iteration', '?')}, device: {device}")
    return net, cfg, device


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/api/new", methods=["POST"])
def new_game():
    global game_state, ai_player
    data = request.json or {}
    ai_player = data.get("ai_player", 1)  # 0 = AI is light, 1 = AI is dark

    game_state = QuoridorGame(2)
    result = get_game_json()

    # If AI goes first, make its move
    if game_state.current_player == ai_player:
        ai_move()
        result = get_game_json()

    return jsonify(result)


@app.route("/api/move", methods=["POST"])
def human_move():
    global game_state
    if game_state is None or game_state.is_over:
        return jsonify({"error": "No active game"}), 400

    data = request.json
    action = data.get("action")
    move_data = data.get("data")

    try:
        if action == "move":
            move = ("move", tuple(move_data))
        elif action == "wall":
            move = ("wall", (move_data[0], move_data[1], move_data[2]))
        else:
            return jsonify({"error": "Invalid action"}), 400

        # Validate it's human's turn
        if game_state.current_player == ai_player:
            return jsonify({"error": "Not your turn"}), 400

        game_state.make_move(move)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = get_game_json()

    # AI responds if game isn't over
    if not game_state.is_over and game_state.current_player == ai_player:
        ai_move()
        result = get_game_json()

    return jsonify(result)


@app.route("/api/state", methods=["GET"])
def get_state():
    if game_state is None:
        return jsonify({"error": "No active game"}), 400
    return jsonify(get_game_json())


@app.route("/api/legal", methods=["GET"])
def get_legal():
    """Return legal moves for the current player (for UI highlighting)."""
    if game_state is None or game_state.is_over:
        return jsonify({"pawn_moves": [], "walls": []})

    pawn_moves = [list(m[1]) for m in game_state.get_legal_pawn_moves()]

    return jsonify({
        "pawn_moves": pawn_moves,
        "can_place_walls": game_state.walls_remaining[game_state.current_player] > 0,
    })


# ── Game logic helpers ──────────────────────────────────────────────

def ai_move():
    """Have the AI make a move."""
    global game_state
    action_probs = ai_mcts.search(game_state, add_noise=False)
    action_idx = select_action(action_probs, temperature=0.1)
    pawn_pos = game_state.pawns[game_state.current_player]
    move = MoveEncoder.decode(action_idx, pawn_pos)
    game_state.make_move(move)


def get_game_json():
    """Serialize game state for the frontend."""
    walls = []
    for w in game_state.walls_placed:
        walls.append({"r": w[0], "c": w[1], "o": w[2]})

    return {
        "pawns": [
            {"r": game_state.pawns[0][0], "c": game_state.pawns[0][1]},
            {"r": game_state.pawns[1][0], "c": game_state.pawns[1][1]},
        ],
        "walls": walls,
        "walls_remaining": list(game_state.walls_remaining),
        "current_player": game_state.current_player,
        "winner": game_state.winner,
        "ai_player": ai_player,
        "num_moves": len(game_state.move_history),
    }


# ── Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quoridor Web App")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    ai_network, cfg, ai_device = load_model(device=args.device)
    ai_mcts = MCTS(
        network=ai_network,
        device=ai_device,
        num_simulations=args.simulations,
        c_puct=1.5,
    )

    print(f"\nQuoridor AI ready ({args.simulations} simulations)")
    print(f"Open http://localhost:{args.port}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)
