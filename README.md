# Quoridor

A minimalist Quoridor board game — play locally, online with friends, or against an AI.

**[Play now](https://quoridor-delta.vercel.app/)**

## Features

- **Local play** — 2 or 4 players on the same device
- **Online multiplayer** — create a room, share a link, play with friends
- **VS Computer** — minimax AI with 3 difficulty levels (easy/medium/hard)
- **Mobile-first** — optimised for phones and tablets, installable as a PWA
- **Board rotation** — optional rotation between turns for face-to-face play

## Stack

- **Frontend**: Single HTML page with vanilla JS
- **Game engine**: Python (validates moves, BFS pathfinding, wall legality)
- **AI**: Client-side JS minimax with alpha-beta pruning and shortest-path heuristic
- **Online**: Redis (Upstash) for room state, 1.5s polling
- **Hosting**: Vercel (serverless Python functions + static files)

## Running locally

```bash
pip install flask numpy
python3 app.py
# Open http://localhost:5000
```

Online rooms use an in-memory store locally (state is lost on server restart). For persistent rooms, set `REDIS_URL`.

## Deploying to Vercel

Push to GitHub and import the repo into Vercel — static files in `public/` and Python serverless functions in `api/` are auto-detected via `vercel.json`.

For online multiplayer, add a Redis database (any Redis-compatible provider works — Upstash, Redis Cloud, etc.) and set the `REDIS_URL` environment variable in your Vercel project settings.

## Architecture

```
public/              Frontend (HTML, JS, PWA assets)
  index.html         Main app
  ai.js              Client-side minimax AI
  how-it-thinks.html Explainer page for the AI
api/                 Vercel serverless functions
  game.py            Local game API (stateless — sends history per call)
  room.py            Online room management (Redis-backed)
  kv.py              Redis wrapper
engine.py            Quoridor game engine (rules, BFS pathfinding)
app.py               Flask dev server (same API as Vercel functions)
```

## How the AI works

See [the live explainer page](https://quoridor-delta.vercel.app/how-it-thinks.html) for a visual walkthrough of the minimax search, complexity growth, and evaluation function.

## License

MIT
