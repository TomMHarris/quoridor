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

# Sentinel token marking an AI-controlled seat. Any human client is allowed to
# play moves for a seat tagged with this value (in practice, only the host of
# the room runs the AI search and posts moves).
AI_TOKEN = "__AI__"


def _game_to_state(game):
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
    """Seat number for a human player token, or None. AI seats never match."""
    if not token or token == AI_TOKEN:
        return None
    for i in range(room["num_players"]):
        if room["players"].get(f"seat_{i}") == token:
            return i
    return None


def _seats_info(room):
    """Public seat info — no tokens leak to clients."""
    info = []
    for i in range(room["num_players"]):
        token = room["players"].get(f"seat_{i}")
        info.append({
            "seat": i,
            "name": room.get("names", {}).get(f"seat_{i}", ""),
            "is_ai": token == AI_TOKEN,
            "occupied": token is not None,
        })
    return info


def _is_host(room, token):
    """Only the seat-0 occupant can fill seats / start AI."""
    return token and room["players"].get("seat_0") == token


def _room_extras(room):
    return {
        "seats": _seats_info(room),
        "ai_depth": room.get("ai_depth"),
        "nudges": room.get("nudges", {}),
        "host_seat": 0,
    }


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
                self._handle_join(room_id, token, body.get("name", ""))
            elif action == "move":
                self._handle_move(room_id, token, body.get("move", {}))
            elif action == "poll":
                self._handle_poll(room_id, token, body.get("last_move_count", -1))
            elif action == "fill_ai":
                self._handle_fill_ai(room_id, token, body.get("ai_depth", 2))
            elif action == "nudge":
                self._handle_nudge(room_id, token, body.get("target_seat"))
            else:
                self._json(400, {"error": f"Unknown action: {action}"})
        except Exception as e:
            self._json(400, {"error": str(e)})

    def _handle_create(self, body, token):
        np = body.get("num_players", 2)
        if np not in (2, 4):
            np = 2
        name = (body.get("name") or "").strip()[:24]

        room_id = _make_room_id()
        game = QuoridorGame(np)

        players = {}
        names = {}
        for i in range(np):
            players[f"seat_{i}"] = token if i == 0 else None
            if i == 0 and name:
                names[f"seat_{i}"] = name

        room = {
            "room_id": room_id,
            "num_players": np,
            "players": players,
            "names": names,
            "status": "waiting",
            "game_dict": game.to_dict(),
            "state": _game_to_state(game),
            "history": _history_to_json(game),
            "legal": _legal_moves(game),
            "move_count": 0,
            "nudges": {},
            "created_at": int(time.time()),
        }

        kv.set(f"room:{room_id}", room, ex=ROOM_TTL)

        resp = {
            "room_id": room_id,
            "seat": 0,
            "state": room["state"],
            "history": room["history"],
            "legal": room["legal"],
            "status": room["status"],
            "num_players": np,
        }
        resp.update(_room_extras(room))
        self._json(200, resp)

    def _handle_join(self, room_id, token, name):
        room = kv.get(f"room:{room_id}")
        if not room:
            self._json(404, {"error": "room_not_found"})
            return

        name = (name or "").strip()[:24]

        # Check if already seated (reconnection)
        seat = _get_seat(room, token)
        if seat is None:
            # Find first empty seat (skip AI seats and other humans)
            for i in range(room["num_players"]):
                if room["players"].get(f"seat_{i}") is None:
                    room["players"][f"seat_{i}"] = token
                    seat = i
                    break
            else:
                self._json(400, {"error": "room_full"})
                return

            # Start the game if all seats now occupied
            all_seated = all(room["players"].get(f"seat_{i}") for i in range(room["num_players"]))
            if all_seated:
                room["status"] = "playing"

        # Always update the player's name on (re)join
        if name:
            room.setdefault("names", {})[f"seat_{seat}"] = name

        kv.set(f"room:{room_id}", room, ex=ROOM_TTL)

        is_my_turn = room["state"]["current_player"] == seat
        legal = room["legal"] if is_my_turn else {
            "pawn_moves": [], "legal_walls": [],
            "shortest_paths": room["legal"].get("shortest_paths", []),
        }

        resp = {
            "room_id": room_id,
            "seat": seat,
            "state": room["state"],
            "history": room["history"],
            "legal": legal,
            "status": room["status"],
            "num_players": room["num_players"],
            "move_count": room["move_count"],
        }
        resp.update(_room_extras(room))
        self._json(200, resp)

    def _handle_fill_ai(self, room_id, token, ai_depth):
        room = kv.get(f"room:{room_id}")
        if not room:
            self._json(404, {"error": "room_not_found"})
            return
        if not _is_host(room, token):
            self._json(403, {"error": "only_host_can_fill"})
            return
        if ai_depth not in (1, 2, 3):
            ai_depth = 2

        for i in range(room["num_players"]):
            if room["players"].get(f"seat_{i}") is None:
                room["players"][f"seat_{i}"] = AI_TOKEN
                difficulty = {1: "easy", 2: "medium", 3: "hard"}[ai_depth]
                room.setdefault("names", {})[f"seat_{i}"] = f"AI ({difficulty})"

        all_seated = all(room["players"].get(f"seat_{i}") for i in range(room["num_players"]))
        if all_seated:
            room["status"] = "playing"
        room["ai_depth"] = ai_depth

        kv.set(f"room:{room_id}", room, ex=ROOM_TTL)
        resp = {"status": room["status"]}
        resp.update(_room_extras(room))
        self._json(200, resp)

    def _handle_move(self, room_id, token, move_data):
        room = kv.get(f"room:{room_id}")
        if not room:
            self._json(404, {"error": "room_not_found"})
            return

        if room["status"] != "playing":
            self._json(400, {"error": "game_not_started"})
            return

        game = QuoridorGame.from_dict(room["game_dict"])
        current_seat_token = room["players"].get(f"seat_{game.current_player}")
        is_ai_seat = current_seat_token == AI_TOKEN

        # Authorization: human players can move only on their own seat. Any
        # client may post a move on behalf of an AI seat (in practice the host
        # does this after running the JS search locally).
        if not is_ai_seat:
            seat = _get_seat(room, token)
            if seat is None:
                self._json(403, {"error": "not_in_room"})
                return
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
            room["legal"] = {
                "pawn_moves": [], "legal_walls": [],
                "shortest_paths": [game.shortest_path_length(p) for p in range(game.num_players)],
            }
        else:
            room["legal"] = _legal_moves(game)

        kv.set(f"room:{room_id}", room, ex=ROOM_TTL)

        resp = {
            "state": room["state"],
            "history": room["history"],
            "legal": room["legal"],
            "move_count": room["move_count"],
            "status": room["status"],
        }
        resp.update(_room_extras(room))
        self._json(200, resp)

    def _handle_nudge(self, room_id, token, target_seat):
        room = kv.get(f"room:{room_id}")
        if not room:
            self._json(404, {"error": "room_not_found"})
            return
        sender_seat = _get_seat(room, token)
        if sender_seat is None:
            self._json(403, {"error": "not_in_room"})
            return
        try:
            target_seat = int(target_seat)
        except (TypeError, ValueError):
            self._json(400, {"error": "bad_target"})
            return
        if not (0 <= target_seat < room["num_players"]):
            self._json(400, {"error": "bad_target"})
            return
        room.setdefault("nudges", {})[f"seat_{target_seat}"] = int(time.time() * 1000)
        kv.set(f"room:{room_id}", room, ex=ROOM_TTL)
        self._json(200, {"ok": True})

    def _handle_poll(self, room_id, token, last_move_count):
        room = kv.get(f"room:{room_id}")
        if not room:
            self._json(404, {"error": "room_not_found"})
            return

        seat = _get_seat(room, token)

        if room["move_count"] == last_move_count and room["status"] == "playing":
            # Status hasn't changed but other clients might be interested in
            # nudges and seat name updates — include them lightweight.
            resp = {"changed": False, "status": room["status"]}
            resp.update(_room_extras(room))
            self._json(200, resp)
            return

        is_my_turn = seat is not None and room["state"]["current_player"] == seat
        legal = room["legal"] if is_my_turn else {
            "pawn_moves": [], "legal_walls": [],
            "shortest_paths": room["legal"].get("shortest_paths", []),
        }

        resp = {
            "changed": True,
            "state": room["state"],
            "history": room["history"],
            "legal": legal,
            "move_count": room["move_count"],
            "status": room["status"],
        }
        resp.update(_room_extras(room))
        self._json(200, resp)

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
