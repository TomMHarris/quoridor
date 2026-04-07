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

## Architecture

```
public/          Frontend (HTML, JS, PWA assets)
  index.html     Main app
  ai.js          Client-side minimax AI
api/             Vercel serverless functions
  game.py        Local game API (stateless)
  room.py        Online room management
  kv.py          Redis wrapper
engine.py        Quoridor game engine
app.py           Flask dev server
```
