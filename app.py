"""
Flask web server for local Quoridor (2 or 4 players).
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


def _game_from_history(history, num_players=2):
    game = QuoridorGame(num_players)
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
        "num_players": game.num_players,
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


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("public", filename)


@app.route("/api/game", methods=["POST", "OPTIONS"])
def game_api():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json() or {}
    action = data.get("action", "new")
    hist = data.get("history", [])
    np = data.get("num_players", 2)
    if np not in (2, 4):
        np = 2

    try:
        if action == "new":
            game = QuoridorGame(np)

        elif action == "move":
            game = _game_from_history(hist, np)
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
            game = _game_from_history(hist[:-1], np)

        elif action == "legal":
            game = _game_from_history(hist, np)
            pawn_moves = game.get_legal_pawn_moves()
            wall_moves = game.get_legal_walls()
            return jsonify({
                "pawn_moves": [list(m[1]) for m in pawn_moves],
                "legal_walls": [[m[1][0], m[1][1], m[1][2]] for m in wall_moves],
                "shortest_paths": [game.shortest_path_length(p) for p in range(np)],
            })

        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400

        return jsonify({
            "state": _game_to_json(game),
            "history": _history_to_json(game),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400


# --- Online rooms (in-memory for local dev, Redis on Vercel) ---

import time
import secrets

_rooms = {}  # In-memory room store for local dev


def _room_game_to_state(game):
    return _game_to_json(game)


def _room_legal(game):
    pawn_moves = game.get_legal_pawn_moves()
    wall_moves = game.get_legal_walls()
    return {
        "pawn_moves": [list(m[1]) for m in pawn_moves],
        "legal_walls": [[m[1][0], m[1][1], m[1][2]] for m in wall_moves],
        "shortest_paths": [game.shortest_path_length(p) for p in range(game.num_players)],
    }


@app.route("/api/room", methods=["POST", "OPTIONS"])
def room_api():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json() or {}
    action = data.get("action")
    token = data.get("player_token", "")
    room_id = data.get("room_id", "")

    try:
        if action == "create":
            np = data.get("num_players", 2)
            if np not in (2, 4):
                np = 2
            room_id = secrets.token_urlsafe(4).replace("-", "").replace("_", "")[:6].lower()
            game = QuoridorGame(np)
            players = {f"seat_{i}": (token if i == 0 else None) for i in range(np)}
            room = {
                "num_players": np, "players": players, "status": "waiting",
                "game": game, "state": _room_game_to_state(game),
                "history": _history_to_json(game), "legal": _room_legal(game),
                "move_count": 0,
            }
            _rooms[room_id] = room
            return jsonify({
                "room_id": room_id, "seat": 0, "state": room["state"],
                "history": room["history"], "legal": room["legal"],
                "status": "waiting", "num_players": np,
            })

        room = _rooms.get(room_id)
        if not room:
            return jsonify({"error": "room_not_found"}), 404

        def _get_seat():
            for i in range(room["num_players"]):
                if room["players"].get(f"seat_{i}") == token:
                    return i
            return None

        if action == "join":
            seat = _get_seat()
            if seat is None:
                for i in range(room["num_players"]):
                    if room["players"].get(f"seat_{i}") is None:
                        room["players"][f"seat_{i}"] = token
                        seat = i
                        break
                else:
                    return jsonify({"error": "room_full"}), 400
                if all(room["players"].get(f"seat_{i}") for i in range(room["num_players"])):
                    room["status"] = "playing"
            is_my_turn = room["state"]["current_player"] == seat
            legal = room["legal"] if is_my_turn else {"pawn_moves": [], "legal_walls": [], "shortest_paths": room["legal"].get("shortest_paths", [])}
            return jsonify({
                "room_id": room_id, "seat": seat, "state": room["state"],
                "history": room["history"], "legal": legal,
                "status": room["status"], "num_players": room["num_players"],
                "move_count": room["move_count"],
            })

        elif action == "move":
            if room["status"] != "playing":
                return jsonify({"error": "game_not_started"}), 400
            seat = _get_seat()
            if seat is None:
                return jsonify({"error": "not_in_room"}), 403
            game = room["game"]
            if game.current_player != seat:
                return jsonify({"error": "not_your_turn"}), 400
            move_data = data.get("move", {})
            if move_data.get("type") == "move":
                game.make_move(("move", tuple(move_data["to"])))
            elif move_data.get("type") == "wall":
                game.make_move(("wall", (move_data["pos"][0], move_data["pos"][1], move_data["orient"])))
            else:
                return jsonify({"error": "invalid_move_type"}), 400
            room["state"] = _room_game_to_state(game)
            room["history"] = _history_to_json(game)
            room["move_count"] = len(game.move_history)
            if game.winner is not None:
                room["status"] = "finished"
                room["legal"] = {"pawn_moves": [], "legal_walls": [], "shortest_paths": [game.shortest_path_length(p) for p in range(game.num_players)]}
            else:
                room["legal"] = _room_legal(game)
            return jsonify({
                "state": room["state"], "history": room["history"],
                "legal": room["legal"], "move_count": room["move_count"],
                "status": room["status"],
            })

        elif action == "poll":
            last = data.get("last_move_count", -1)
            seat = _get_seat()
            if room["move_count"] == last and room["status"] == "playing":
                return jsonify({"changed": False, "status": room["status"]})
            is_my_turn = seat is not None and room["state"]["current_player"] == seat
            legal = room["legal"] if is_my_turn else {"pawn_moves": [], "legal_walls": [], "shortest_paths": room["legal"].get("shortest_paths", [])}
            return jsonify({
                "changed": True, "state": room["state"], "history": room["history"],
                "legal": legal, "move_count": room["move_count"], "status": room["status"],
            })

        else:
            return jsonify({"error": f"Unknown action: {action}"}), 400

    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    port = args.port or int(os.environ.get("PORT", 5000))
    print(f"\nStarting server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
