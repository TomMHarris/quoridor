"""
Online multiplayer room management for Quoridor.

Room state lives in the kv store (Upstash / Redis) with a 4-hour TTL; clients
poll every ~1.5s. Every write is a compare-and-set against the exact room the
request read, so concurrent requests — a nudge landing during a move, two people
taking the last seat — retry against fresh state instead of overwriting each
other.

`dispatch` holds all the logic. The Vercel handler below and the Flask dev
server (app.py) are both thin wrappers around it.
"""

import json
import sys
import os
import time
import secrets
import traceback
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import QuoridorGame

# Import KV helper (lives in same api/ directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kv

ROOM_TTL = 14400  # 4 hours
SWAP_RETRIES = 6  # compare-and-set attempts before answering "busy"

# Sentinel token marking an AI-controlled seat. Any seated human may play moves
# for a seat tagged with this value: the host's browser does, and the others
# step in if the host's tab goes quiet.
AI_TOKEN = "__AI__"


class _Reject(Exception):
    """End the request with an error response."""

    def __init__(self, status, error):
        super().__init__(error)
        self.status = status
        self.error = error


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


def _legal_for(room, seat):
    """The stored legal moves if it is `seat`'s turn; otherwise none."""
    if seat is not None and room["state"]["current_player"] == seat:
        return room["legal"]
    return {
        "pawn_moves": [], "legal_walls": [],
        "shortest_paths": room["legal"].get("shortest_paths", []),
    }


def _make_room_id():
    return secrets.token_urlsafe(4).replace("-", "").replace("_", "")[:6].lower()


def _key(room_id):
    return f"room:{room_id}"


def _get_seat(room, token):
    """Seat number for a human player token, or None. AI seats never match."""
    if not token or token == AI_TOKEN:
        return None
    for i in range(room["num_players"]):
        if room["players"].get(f"seat_{i}") == token:
            return i
    return None


def _all_seated(room):
    return all(room["players"].get(f"seat_{i}") for i in range(room["num_players"]))


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
    return bool(token) and room["players"].get("seat_0") == token


def _room_extras(room):
    return {
        "seats": _seats_info(room),
        "ai_depth": room.get("ai_depth"),
        "nudges": room.get("nudges", {}),
        "host_seat": 0,
        "next_room": room.get("next_room"),
    }


def _snapshot(room, seat):
    """Everything a client needs to draw the room from `seat`'s point of view."""
    resp = {
        "state": room["state"],
        "history": room["history"],
        "legal": _legal_for(room, seat),
        "status": room["status"],
        "num_players": room["num_players"],
        "move_count": room["move_count"],
    }
    resp.update(_room_extras(room))
    return resp


def _new_room(num_players, players, names, ai_depth=None):
    game = QuoridorGame(num_players)
    return {
        "num_players": num_players,
        "players": players,
        "names": names,
        "status": "playing" if all(players.values()) else "waiting",
        "game_dict": game.to_dict(),
        "state": _game_to_state(game),
        "history": _history_to_json(game),
        "legal": _legal_moves(game),
        "move_count": 0,
        "nudges": {},
        "ai_depth": ai_depth,
        "created_at": int(time.time()),
    }


def _store_new_room(room):
    """Save a new room under a fresh id (never overwriting a live room)."""
    for _ in range(8):
        room_id = _make_room_id()
        room["room_id"] = room_id
        if kv.create(_key(room_id), room, ROOM_TTL):
            return room_id
    raise _Reject(503, "busy")


def _update(room_id, change):
    """
    Read a room, apply change(room) and write it back atomically.

    change mutates the room and returns the response payload, or raises _Reject.
    If another request wrote the room in between, the swap fails and change runs
    again on the fresh room, so every check it makes is against current state.
    """
    for _ in range(SWAP_RETRIES):
        room, raw = kv.load(_key(room_id))
        if room is None:
            raise _Reject(404, "room_not_found")
        payload = change(room)
        if json.dumps(room) == raw:  # nothing changed, e.g. a plain reconnect
            return payload
        if kv.swap(_key(room_id), raw, room, ROOM_TTL):
            return payload
    raise _Reject(503, "busy")


def _parse_move(move):
    """The engine's move tuple for a client move payload, or None if malformed."""
    if not isinstance(move, dict):
        return None

    def cell(value, top):
        return (isinstance(value, list) and len(value) == 2
                and all(type(x) is int and 0 <= x <= top for x in value))

    size = QuoridorGame.BOARD_SIZE
    if move.get("type") == "move" and cell(move.get("to"), size - 1):
        return ("move", tuple(move["to"]))
    if (move.get("type") == "wall" and cell(move.get("pos"), size - 2)
            and move.get("orient") in ("H", "V")):
        r, c = move["pos"]
        return ("wall", (r, c, move["orient"]))
    return None


# --- Actions ---------------------------------------------------------------


def _handle_create(body, token, room_id):
    if not token:
        raise _Reject(400, "bad_request")
    np = body.get("num_players", 2)
    if np not in (2, 4) or isinstance(np, bool):
        np = 2
    name = body.get("name")
    name = name.strip()[:24] if isinstance(name, str) else ""

    players = {f"seat_{i}": (token if i == 0 else None) for i in range(np)}
    names = {"seat_0": name} if name else {}
    room = _new_room(np, players, names)
    room_id = _store_new_room(room)

    resp = {"room_id": room_id, "seat": 0}
    resp.update(_snapshot(room, 0))
    return resp


def _handle_join(body, token, room_id):
    name = body.get("name")
    name = name.strip()[:24] if isinstance(name, str) else ""

    def change(room):
        seat = _get_seat(room, token)
        if seat is None:
            # Take the first empty seat (AI seats and other humans are taken)
            for i in range(room["num_players"]):
                if room["players"].get(f"seat_{i}") is None:
                    room["players"][f"seat_{i}"] = token
                    seat = i
                    break
            else:
                raise _Reject(400, "room_full")
            if _all_seated(room):
                room["status"] = "playing"

        # Always update the player's name on (re)join
        if name:
            room.setdefault("names", {})[f"seat_{seat}"] = name

        resp = {"room_id": room_id, "seat": seat}
        resp.update(_snapshot(room, seat))
        return resp

    return _update(room_id, change)


def _handle_fill_ai(body, token, room_id):
    ai_depth = body.get("ai_depth", 2)
    if ai_depth not in (1, 2, 3) or isinstance(ai_depth, bool):
        ai_depth = 2
    difficulty = {1: "easy", 2: "medium", 3: "hard"}[ai_depth]

    def change(room):
        if not _is_host(room, token):
            raise _Reject(403, "only_host_can_fill")
        for i in range(room["num_players"]):
            if room["players"].get(f"seat_{i}") is None:
                room["players"][f"seat_{i}"] = AI_TOKEN
                room.setdefault("names", {})[f"seat_{i}"] = f"AI ({difficulty})"
        if _all_seated(room):
            room["status"] = "playing"
        room["ai_depth"] = ai_depth

        resp = {"status": room["status"]}
        resp.update(_room_extras(room))
        return resp

    return _update(room_id, change)


def _handle_move(body, token, room_id):
    move = _parse_move(body.get("move"))
    if move is None:
        raise _Reject(400, "invalid_move_type")
    # Which position the move was chosen for. Without it a duplicate or late
    # post — a second tab of the host, a retried request — would be applied to
    # whichever seat is to move now. Optional so pages cached by the service
    # worker before this field existed keep working.
    expected = body.get("expected_move_count")

    def change(room):
        if room["status"] != "playing":
            raise _Reject(400, "game_not_started")
        if expected is not None and expected != room["move_count"]:
            raise _Reject(409, "stale_move")

        seat = _get_seat(room, token)
        if seat is None:
            raise _Reject(403, "not_in_room")
        game = QuoridorGame.from_dict(room["game_dict"])
        is_ai_seat = room["players"].get(f"seat_{game.current_player}") == AI_TOKEN
        if not is_ai_seat and game.current_player != seat:
            raise _Reject(400, "not_your_turn")

        game.make_move(move)  # ValueError on an illegal move

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

        # Legal moves are the *next* player's: only they get to see them.
        return _snapshot(room, seat)

    return _update(room_id, change)


def _handle_nudge(body, token, room_id):
    try:
        target_seat = int(body.get("target_seat"))
    except (TypeError, ValueError):
        raise _Reject(400, "bad_target")

    def change(room):
        if _get_seat(room, token) is None:
            raise _Reject(403, "not_in_room")
        if not (0 <= target_seat < room["num_players"]):
            raise _Reject(400, "bad_target")
        room.setdefault("nudges", {})[f"seat_{target_seat}"] = int(time.time() * 1000)
        return {"ok": True}

    return _update(room_id, change)


def _handle_rematch(body, token, room_id):
    """
    Start a new game with the same seats. The first player to ask creates it
    and records it in the finished room; everyone else — including other
    players asking at the same moment — gets that same room, and clients still
    polling the finished room follow `next_room` into it.
    """
    room = kv.get(_key(room_id))
    if not room:
        raise _Reject(404, "room_not_found")
    if _get_seat(room, token) is None:
        raise _Reject(403, "not_in_room")
    if room["status"] != "finished":
        raise _Reject(400, "game_not_finished")
    if room.get("next_room"):
        return {"room_id": room["next_room"]}

    new_room = _new_room(room["num_players"], dict(room["players"]),
                         dict(room.get("names", {})), room.get("ai_depth"))
    new_id = _store_new_room(new_room)

    def change(room):
        # A concurrent rematch may have won the race: use its room, and leave
        # ours to expire unvisited.
        room.setdefault("next_room", new_id)
        return {"room_id": room["next_room"]}

    return _update(room_id, change)


def _handle_poll(body, token, room_id):
    room = kv.get(_key(room_id))
    if not room:
        raise _Reject(404, "room_not_found")

    if room["move_count"] == body.get("last_move_count", -1) and room["status"] == "playing":
        # Status hasn't changed but other clients might be interested in
        # nudges and seat name updates — include them lightweight.
        resp = {"changed": False, "status": room["status"]}
        resp.update(_room_extras(room))
        return resp

    resp = {"changed": True}
    resp.update(_snapshot(room, _get_seat(room, token)))
    return resp


_ACTIONS = {
    "create": _handle_create,
    "join": _handle_join,
    "move": _handle_move,
    "poll": _handle_poll,
    "fill_ai": _handle_fill_ai,
    "nudge": _handle_nudge,
    "rematch": _handle_rematch,
}


def dispatch(body):
    """Run one room API request. Returns (http_status, json_payload)."""
    if not kv.available():
        return 503, {"error": "online_not_configured"}
    if not isinstance(body, dict):
        return 400, {"error": "bad_request"}

    action = body.get("action")
    token = body.get("player_token") or ""
    room_id = body.get("room_id") or ""
    if not isinstance(token, str) or not isinstance(room_id, str) or len(room_id) > 32:
        return 400, {"error": "bad_request"}
    handler_fn = _ACTIONS.get(action) if isinstance(action, str) else None
    if handler_fn is None:
        return 400, {"error": f"Unknown action: {str(action)[:40]}"}

    try:
        return 200, handler_fn(body, token, room_id)
    except _Reject as e:
        return e.status, {"error": e.error}
    except kv.StoreUnavailable as e:
        # The store is configured but not answering — a deleted database, a
        # rotated password, a network blip. One stable code for the client;
        # the specifics go in `detail` for whoever is debugging.
        return 503, {"error": "online_unavailable", "detail": str(e)[:200]}
    except ValueError as e:
        # The engine rejecting an illegal move; its message is meant for people.
        return 400, {"error": str(e)[:200]}
    except Exception:
        traceback.print_exc()
        return 500, {"error": "server_error"}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length)) if length else {}
        except (ValueError, OSError):  # bad length, bad JSON, bad UTF-8
            self._json(400, {"error": "bad_request"})
            return
        status, payload = dispatch(body)
        self._json(status, payload)

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
