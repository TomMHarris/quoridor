/**
 * Offline rules engine for local play (pass-and-play and vs computer).
 *
 * This is the client-side twin of api/game.py: same actions, same request and
 * response shapes, same history format — so the page can call it instead of
 * POSTing to the server and nothing downstream notices. That's what makes the
 * installed app playable with no internet at all: two people can share a phone
 * on a plane, and "best move" still works because the AI was always local too.
 *
 * The rules themselves are not reimplemented here. They come from AI.rules —
 * the very code the search runs on — so the validator and the AI can never
 * disagree about what is legal. This file only adds what the search doesn't
 * need: strict validation of a submitted move, full (unpruned) legal-wall
 * enumeration for the UI, and the move history that drives undo.
 *
 * History format (identical to the server's):
 *   ["move", [r, c]]  ·  ["wall", [r, c, "H" | "V"]]
 */
const LocalGame = (() => {
  const WALL_GRID = 8;

  // Mirrors QuoridorGame.SETUP in engine.py. 4-player seats run clockwise from
  // the bottom: bottom, left, top, right.
  const SETUP = {
    2: {
      starts: [[8, 4], [0, 4]],
      goals: [{ row: 0 }, { row: 8 }],
      walls: 10,
    },
    4: {
      starts: [[8, 4], [4, 0], [0, 4], [4, 8]],
      goals: [{ row: 0 }, { col: 8 }, { row: 8 }, { col: 0 }],
      walls: 5,
    },
  };

  const R = () => AI.rules;

  function start(np) {
    const cfg = SETUP[np];
    const s = R().create(np, cfg.starts, new Array(np).fill(cfg.walls), [], 0, cfg.goals);
    s.winner = null;
    s.wallList = [];   // walls in placement order — the UI renders this list
    return s;
  }

  // Validate `move` against the position and apply it. Throws on anything
  // illegal, exactly where the Python engine would raise ValueError.
  function play(s, move) {
    if (s.winner !== null) throw new Error("game is already over");

    if (move.type === "move") {
      const to = move.to;
      if (!Array.isArray(to)) throw new Error("invalid move");
      if (!R().getPawnMoves(s, s.cp).some(m => m[0] === to[0] && m[1] === to[1])) {
        throw new Error("illegal move");
      }
      R().applyMove(s, { type: "move", to: [to[0], to[1]] });
      return;
    }

    if (move.type === "wall") {
      const pos = move.pos, o = move.orient;
      if (!Array.isArray(pos) || (o !== "H" && o !== "V")) throw new Error("invalid wall");
      const r = pos[0], c = pos[1];
      if (s.wr[s.cp] <= 0) throw new Error("no walls remaining");
      if (!R().isValidWall(s, r, c, o)) throw new Error("illegal wall");
      s.wallList.push([r, c, o]);
      R().applyMove(s, { type: "wall", pos: [r, c], orient: o });
      return;
    }

    throw new Error("invalid move type");
  }

  function historyEntry(move) {
    return move.type === "move"
      ? ["move", [move.to[0], move.to[1]]]
      : ["wall", [move.pos[0], move.pos[1], move.orient]];
  }

  // Replay from the opening position, like the stateless server does. Every
  // move is re-validated on the way, so a corrupted history fails loudly
  // instead of producing an impossible board.
  function replay(history, np) {
    const s = start(np);
    for (const entry of history) {
      const [action, data] = entry;
      if (action === "move") play(s, { type: "move", to: [data[0], data[1]] });
      else if (action === "wall") play(s, { type: "wall", pos: [data[0], data[1]], orient: data[2] });
      else throw new Error(`unknown action: ${action}`);
    }
    return s;
  }

  function toState(s, moveCount) {
    return {
      pawns: s.pawns.map(p => [p[0], p[1]]),
      walls: s.wallList.map(w => [w[0], w[1], w[2]]),
      walls_remaining: s.wr.slice(),
      current_player: s.cp,
      winner: s.winner == null ? null : s.winner,
      move_count: moveCount,
      num_players: s.np,
    };
  }

  function shortestPaths(s) {
    const out = [];
    for (let p = 0; p < s.np; p++) out.push(R().shortestPath(s, p));
    return out;
  }

  // Every wall the current player may legally place — all 8×8×2 slots, not the
  // search's pruned candidate set. The UI needs the complete list to decide
  // which grooves are tappable.
  function legal(s) {
    const paths = shortestPaths(s);
    if (s.winner !== null) return { pawn_moves: [], legal_walls: [], shortest_paths: paths };

    const pawnMoves = R().getPawnMoves(s, s.cp).map(m => [m[0], m[1]]);
    const walls = [];
    if (s.wr[s.cp] > 0) {
      for (let r = 0; r < WALL_GRID; r++) {
        for (let c = 0; c < WALL_GRID; c++) {
          if (R().isValidWall(s, r, c, "H")) walls.push([r, c, "H"]);
          if (R().isValidWall(s, r, c, "V")) walls.push([r, c, "V"]);
        }
      }
    }
    return { pawn_moves: pawnMoves, legal_walls: walls, shortest_paths: paths };
  }

  // Same contract as POST /api/game: an action plus {history, num_players,
  // move}, answered with {state, history} / {pawn_moves, ...} / {error}.
  function handle(action, body) {
    const b = body || {};
    const np = b.num_players === 4 ? 4 : 2;
    const history = b.history || [];

    try {
      if (action === "new") {
        return { state: toState(start(np), 0), history: [] };
      }

      if (action === "move") {
        const s = replay(history, np);
        const move = b.move || {};
        play(s, move);
        const h = history.concat([historyEntry(move)]);
        return { state: toState(s, h.length), history: h };
      }

      if (action === "undo") {
        if (!history.length) return { error: "no moves to undo" };
        const h = history.slice(0, -1);
        return { state: toState(replay(h, np), h.length), history: h };
      }

      if (action === "legal") {
        return legal(replay(history, np));
      }

      return { error: `unknown action: ${action}` };
    } catch (e) {
      return { error: String((e && e.message) || e) };
    }
  }

  return { handle };
})();

// Node (test harness) — browsers get the global from the const above.
if (typeof module !== "undefined" && module.exports) module.exports = LocalGame;
