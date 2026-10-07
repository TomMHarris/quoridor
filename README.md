# Quoridor

A minimalist Quoridor board game — play locally, online with friends, or against an AI.

**[Play now](https://quoridor-delta.vercel.app/)**

## Features

- **Local play** — 2 or 4 players on the same device, **fully offline**; a game
  in progress survives reloads and the app being closed
- **Online multiplayer** — create a room, share a link, play with friends;
  fill empty seats with the computer; one-tap rematch
- **VS Computer** — minimax AI with 3 difficulty levels (easy/medium/hard)
- **Mobile-first** — optimised for phones and tablets, installable as a PWA
- **Board rotation** — optional rotation between turns for face-to-face play

## Offline

Everything except online rooms runs in the browser: pass-and-play, vs computer,
undo, and the best-move hint. Install the PWA once and it needs no network at
all — no server, no wifi. Only online rooms talk to the API.

## Stack

- **Frontend**: Single HTML page with vanilla JS
- **Local rules**: `public/local-game.js` on top of the rules in `public/ai.js` —
  one JS implementation shared by the validator and the search
- **AI**: Client-side JS minimax with alpha-beta pruning and shortest-path heuristic
- **Server rules**: Python `engine.py` — authoritative for online rooms
- **Online**: Redis / Upstash for room state, 1.5s polling
- **Hosting**: Vercel (serverless Python functions + static files)

## Running locally

```bash
pip install flask
python3 app.py
# Open http://localhost:5000
```

Local and vs-computer games don't need the server at all — opening
`public/index.html` through any static file server is enough. The Flask app adds
the online room API by running the same handler as the Vercel function
(`api/room.py`), with rooms kept in memory (lost on restart) unless a store is
configured in the environment.

## Deploying to Vercel

Push to GitHub and import the repo into Vercel — static files in `public/` and Python serverless functions in `api/` are auto-detected via `vercel.json`.

Online multiplayer needs a room store. Attach any Redis-compatible database
(Upstash, Redis Cloud, …) and set **one** of:

| Variable(s) | Transport |
| --- | --- |
| `KV_REST_API_URL` + `KV_REST_API_TOKEN` | HTTP REST (preferred on Vercel) |
| `UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN` | HTTP REST |
| `REDIS_URL` (or `KV_URL` / `STORAGE_REDIS_URL` / `REDIS_TLS_URL`) | Redis over TCP |

Most Vercel storage integrations set these automatically. With none of them, the
room API answers `online_not_configured`; if the store is set but unreachable —
a deleted database, a rotated password — it answers `online_unavailable` with a
`detail` naming the variable it used and why the connection failed, and logs the
full error to the function logs. Local play is unaffected either way.

To check the store from anywhere:

```bash
curl -s -X POST https://quoridor-delta.vercel.app/api/room \
  -H 'Content-Type: application/json' -d '{"action":"poll","room_id":"none"}'
# healthy:  {"error": "room_not_found"}
# broken:   {"error": "online_unavailable",
#            "detail": "GET via REDIS_URL (redis://): host_not_found (...)"}
```

## Architecture

```
public/              Frontend (HTML, JS, PWA assets)
  index.html         Main app
  ai.js              Client-side minimax AI (+ the shared rules primitives)
  local-game.js      Offline rules/history layer for local play
  ai-worker.js       Runs the best-move search off the main thread
  sw.js              Service worker — caches everything a local game needs
  how-it-thinks.html Explainer page for the AI
api/                 Vercel serverless functions
  room.py            Online room management (store-backed); `dispatch()` is
                     shared with app.py
  kv.py              Room store (Upstash REST, Redis over TCP, or in-memory);
                     updates are compare-and-set, so concurrent requests can't
                     overwrite each other
  game.py            Stateless local-game API — kept as the Python reference;
                     the browser no longer calls it
engine.py            Quoridor game engine (rules, BFS pathfinding)
app.py               Flask dev server (same API as Vercel functions)
```

## Rules notes

Standard Quoridor. One case the official rules leave open: in 4-player a pawn
can be boxed in by other pawns and walls with no walls left to place. That
player passes until they can move again (`engine.py` and `public/local-game.js`
agree on this).

Online, computer seats are played by the room creator's browser. If that
browser goes quiet for 8 seconds (tab closed, phone locked), another player's
browser takes the computer's turns instead.

## How the AI works

See [the live explainer page](https://quoridor-delta.vercel.app/how-it-thinks.html) for a visual walkthrough of the minimax search, complexity growth, and evaluation function.

## License

MIT
