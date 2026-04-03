"""
Online multiplayer room management for Quoridor.

Stores room state in Upstash Redis (Vercel KV) with 4-hour TTL.
Clients poll every ~1.5s for updates.
"""

import json
import sys
import os
import time
import secrets
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import QuoridorGame

# Import KV helper (lives in same api/ directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv

ROOM_TTL = 14400  # 4 hours


def _game_to_state(game):
    """Frontend-friendly state (same format as api/game.py)."""
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


def _legal_moves(game):
    pawn_moves = game.get_legal_pawn_moves()
    wall_moves = game.get_legal_walls()
    return {
        "pawn_moves": [list(m[1]) for m in pawn_moves],
        "legal_walls": [[m[1][0], m[1][1], m[1][2]] for m in wall_moves],
        "shortest_paths": [game.shortest_path_length(p) for p in range(game.num_players)],
    }


def _make_room_id():
    return secrets.token_urlsafe(4).replace("-", "").replace("_", "")[:6].lower()


def _get_seat(room, token):
    """Return seat number for a player token, or None."""
    for i in range(room["num_players"]):
        if room["players"].get(f"seat_{i}") == token:
            return i
    return None


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if not kv.available():
            self._json(503, {"error": "online_not_configured"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}

        action = body.get("action")
        token = body.get("player_token", "")
        room_id = body.get("room_id", "")

        try:
            if action == "create":
                self._handle_create(body, token)
            elif action == "join":
                self._handle_join(room_id, token)
            elif action == "move":
                self._handle_move(room_id, token, body.get("move", {}))
            elif action == "poll":
                self._handle_poll(room_id, token, body.get("last_move_count", -1))
            else:
                self._json(400, {"error": f"Unknown action: {action}"})
        except Exception as e:
            self._json(400, {"error": str(e)})

    def _handle_create(self, body, token):
        np = body.get("num_players", 2)
        if np not in (2, 4):
            np = 2

        room_id = _make_room_id()
        game = QuoridorGame(np)

        players = {}
        for i in range(np):
            players[f"seat_{i}"] = token if i == 0 else None

        room = {
            "room_id": room_id,
            "num_players": np,
            "players": players,
            "status": "waiting",
            "game_dict": game.to_dict(),
            "state": _game_to_state(game),
            "history": _history_to_json(game),
            "legal": _legal_moves(game),
            "move_count": 0,
            "created_at": int(time.time()),
        }

        kv.set(f"room:{room_id}", room, ex=ROOM_TTL)

        self._json(200, {
            "room_id": room_id,
            "seat": 0,
            "state": room["state"],
            "history": room["history"],
            "legal": room["legal"],
            "status": room["status"],
            "num_players": np,
        })

    def _handle_join(self, room_id, token):
        room = kv.get(f"room:{room_id}")
        if not room:
            self._json(404, {"error": "room_not_found"})
            return

        # Check if already seated (reconnection)
        seat = _get_seat(room, token)
        if seat is not None:
            legal = room["legal"] if room["state"]["current_player"] == seat else {"pawn_moves": [], "legal_walls": [], "shortest_paths": room["legal"].get("shortest_paths", [])}
            self._json(200, {
                "room_id": room_id,
                "seat": seat,
                "state": room["state"],
                "history": room["history"],
                "legal": legal,
                "status": room["status"],
                "num_players": room["num_players"],
                "move_count": room["move_count"],
            })
            return

        # Find first empty seat
        for i in range(room["num_players"]):
            if room["players"].get(f"seat_{i}") is None:
                room["players"][f"seat_{i}"] = token
                seat = i
                break
        else:
            self._json(400, {"error": "room_full"})
            return

        # Check if all seats filled
        all_seated = all(room["players"].get(f"seat_{i}") for i in range(room["num_players"]))
        if all_seated:
            room["status"] = "playing"

        kv.set(f"room:{room_id}", room, ex=ROOM_TTL)

        legal = room["legal"] if room["state"]["current_player"] == seat else {"pawn_moves": [], "legal_walls": [], "shortest_paths": room["legal"].get("shortest_paths", [])}

        self._json(200, {
            "room_id": room_id,
            "seat": seat,
            "state": room["state"],
            "history": room["history"],
            "legal": legal,
            "status": room["status"],
            "num_players": room["num_players"],
            "move_count": room["move_count"],
        })

    def _handle_move(self, room_id, token, move_data):
        room = kv.get(f"room:{room_id}")
        if not room:
            self._json(404, {"error": "room_not_found"})
            return

        if room["status"] != "playing":
            self._json(400, {"error": "game_not_started"})
            return

        seat = _get_seat(room, token)
        if seat is None:
            self._json(403, {"error": "not_in_room"})
            return

        game = QuoridorGame.from_dict(room["game_dict"])

        if game.current_player != seat:
            self._json(400, {"error": "not_your_turn"})
            return

        # Apply move
        if move_data.get("type") == "move":
            game.make_move(("move", tuple(move_data["to"])))
        elif move_data.get("type") == "wall":
            game.make_move(("wall", (move_data["pos"][0], move_data["pos"][1], move_data["orient"])))
        else:
            self._json(400, {"error": "invalid_move_type"})
            return

        # Update room
        room["game_dict"] = game.to_dict()
        room["state"] = _game_to_state(game)
        room["history"] = _history_to_json(game)
        room["move_count"] = len(game.move_history)

        if game.winner is not None:
            room["status"] = "finished"
            room["legal"] = {"pawn_moves": [], "legal_walls": [], "shortest_paths": [game.shortest_path_length(p) for p in range(game.num_players)]}
        else:
            room["legal"] = _legal_moves(game)

        kv.set(f"room:{room_id}", room, ex=ROOM_TTL)

        self._json(200, {
            "state": room["state"],
            "history": room["history"],
            "legal": room["legal"],
            "move_count": room["move_count"],
            "status": room["status"],
        })

    def _handle_poll(self, room_id, token, last_move_count):
        room = kv.get(f"room:{room_id}")
        if not room:
            self._json(404, {"error": "room_not_found"})
            return

        seat = _get_seat(room, token)

        if room["move_count"] == last_move_count and room["status"] == "playing":
            # Check if status changed (e.g. new player joined)
            self._json(200, {"changed": False, "status": room["status"]})
            return

        # Something changed — return full state
        is_my_turn = seat is not None and room["state"]["current_player"] == seat
        legal = room["legal"] if is_my_turn else {"pawn_moves": [], "legal_walls": [], "shortest_paths": room["legal"].get("shortest_paths", [])}

        self._json(200, {
            "changed": True,
            "state": room["state"],
            "history": room["history"],
            "legal": legal,
            "move_count": room["move_count"],
            "status": room["status"],
        })

    def _json(self, status, data):
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
