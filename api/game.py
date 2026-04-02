"""
Stateless Vercel serverless function for Quoridor (2 or 4 players).

The client sends the full move history + num_players with each request.
The server replays it, validates the new move, and returns the updated state.
"""

import json
import sys
import os
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import QuoridorGame


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


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}

        action = body.get("action", "new")
        history = body.get("history", [])
        np = body.get("num_players", 2)
        if np not in (2, 4):
            np = 2

        try:
            if action == "new":
                game = QuoridorGame(np)

            elif action == "move":
                game = _game_from_history(history, np)
                move_data = body.get("move", {})
                if move_data.get("type") == "move":
                    game.make_move(("move", tuple(move_data["to"])))
                elif move_data.get("type") == "wall":
                    game.make_move(("wall", (move_data["pos"][0], move_data["pos"][1], move_data["orient"])))
                else:
                    self._json_response(400, {"error": "Invalid move type"})
                    return

            elif action == "undo":
                if not history:
                    self._json_response(400, {"error": "No moves to undo"})
                    return
                game = _game_from_history(history[:-1], np)

            elif action == "legal":
                game = _game_from_history(history, np)
                pawn_moves = game.get_legal_pawn_moves()
                wall_moves = game.get_legal_walls()
                self._json_response(200, {
                    "pawn_moves": [list(m[1]) for m in pawn_moves],
                    "legal_walls": [[m[1][0], m[1][1], m[1][2]] for m in wall_moves],
                    "shortest_paths": [game.shortest_path_length(p) for p in range(np)],
                })
                return

            else:
                self._json_response(400, {"error": f"Unknown action: {action}"})
                return

            self._json_response(200, {
                "state": _game_to_json(game),
                "history": _history_to_json(game),
            })

        except ValueError as e:
            self._json_response(400, {"error": str(e)})

    def _json_response(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
