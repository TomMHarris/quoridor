# Quoridor

A minimalist Quoridor board game — play locally, online with friends, or against an AI.

**[Play now](https://quoridor-delta.vercel.app/)**

## Features

- **Local play** — 2 or 4 players on the same device, **fully offline**
- **Online multiplayer** — create a room, share a link, play with friends
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
pip install flask numpy
python3 app.py
# Open http://localhost:5000
```

Local and vs-computer games don't need the server at all — opening
`public/index.html` through any static file server is enough. The Flask app adds
the online room API, which uses an in-memory store locally (state is lost on
restart).

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
a deleted database, a rotated password — it answers `online_unavailable` and logs
the reason. Local play is unaffected either way.

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
  room.py            Online room management (store-backed)
  kv.py              Room store (Upstash REST or Redis over TCP)
  game.py            Stateless local-game API — kept as the Python reference;
                     the browser no longer calls it
engine.py            Quoridor game engine (rules, BFS pathfinding)
app.py               Flask dev server (same API as Vercel functions)
```

## How the AI works

See [the live explainer page](https://quoridor-delta.vercel.app/how-it-thinks.html) for a visual walkthrough of the minimax search, complexity growth, and evaluation function.

## License

MIT
