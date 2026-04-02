"""
Flask web server for two-player local Quoridor.
Serves both the stateless API (matching Vercel) and the frontend.

Usage:
    python3 app.py                # Default: port 5000
    python3 app.py --port 8080
"""

import argparse
import os

from flask import Flask, request, jsonify, send_from_directory

from engine import QuoridorGame

app = Flask(__name__, static_folder="public")


def _game_from_history(history):
    game = QuoridorGame(2)
    for move in history:
        action = move[0]
        if action == "move":
            game.make_move(("move", tuple(move[1])))
        elif action == "wall":
            game.make_move(("wall", (move[1][0], move[1][1], move[1][2])))
    return game


def _game_to_json(game):
    return {
        "pawns": [list(p) for p in game.pawns],
        "walls": [[w[0], w[1], w[2]] for w in game.walls_placed],
        "walls_remaining": list(game.walls_remaining),
        "current_player": game.current_player,
        "winner": game.winner,
        "move_count": len(game.move_history),
    }


def _history_to_json(game):
    result = []
    for action, data in game.move_history:
        if action == "move":
            result.append(["move", list(data)])
        else:
            result.append(["wall", [data[0], data[1], data[2]]])
    return result


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.route("/api/game", methods=["POST", "OPTIONS"])
def game_api():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json() or {}
    action = data.get("action", "new")
    hist = data.get("history", [])

    try:
        if action == "new":
            game = QuoridorGame(2)

        elif action == "move":
            game = _game_from_history(hist)
            move_data = data.get("move", {})
            if move_data.get("type") == "move":
                game.make_move(("move", tuple(move_data["to"])))
            elif move_data.get("type") == "wall":
                game.make_move(("wall", (move_data["pos"][0], move_data["pos"][1], move_data["orient"])))
            else:
                return jsonify({"error": "Invalid move type"}), 400

        elif action == "undo":
            if not hist:
                return jsonify({"error": "No moves to undo"}), 400
            game = _game_from_history(hist[:-1])

        elif action == "legal":
            game = _game_from_history(hist)
            moves = game.get_legal_pawn_moves()
            return jsonify({"pawn_moves": [list(m[1]) for m in moves]})

        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400

        return jsonify({
            "state": _game_to_json(game),
            "history": _history_to_json(game),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    port = args.port or int(os.environ.get("PORT", 5000))
    print(f"\nStarting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
