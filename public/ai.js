/**
 * Quoridor AI — Minimax with alpha-beta pruning.
 *
 * Runs entirely client-side. Uses shortest-path-length as the evaluation
 * heuristic: good positions have a short path for us and a long path for
 * the opponent.
 *
 * The game state is represented as a compact object that can be cheaply
 * cloned and mutated during the search.
 */

const AI = (() => {
  const B = 9;          // board size
  const W = 8;          // wall grid size
  const DIRS = [[-1,0],[1,0],[0,-1],[0,1]];

  // ── Compact game state ──

  function create(numPlayers, pawns, wallsRemaining, wallsPlaced, currentPlayer, goals) {
    // Sets stored as objects with string keys for fast lookup
    const hBlocked = {};  // "r,c" → true  (horizontal wall blocks step down from (r,c))
    const vBlocked = {};  // "r,c" → true  (vertical wall blocks step right from (r,c))
    const wallCenters = {};
    const wallSet = {};   // "r,c,o" → true

    for (const [r, c, o] of wallsPlaced) {
      wallSet[`${r},${c},${o}`] = true;
      wallCenters[`${r},${c}`] = true;
      if (o === "H") {
        hBlocked[`${r},${c}`] = true;
        hBlocked[`${r},${c+1}`] = true;
      } else {
        vBlocked[`${r},${c}`] = true;
        vBlocked[`${r+1},${c}`] = true;
      }
    }

    return {
      np: numPlayers,
      pawns: pawns.map(p => [p[0], p[1]]),
      wr: wallsRemaining.slice(),
      walls: wallSet,
      centers: wallCenters,
      hb: hBlocked,
      vb: vBlocked,
      cp: currentPlayer,
      goals: goals,
    };
  }

  function fromServerState(state) {
    // Map server state to our compact format
    const np = state.num_players || 2;
    const goals = np === 2
      ? [{row: 0}, {row: 8}]
      : [{row: 0}, {col: 8}, {row: 8}, {col: 0}];
    return create(np, state.pawns, state.walls_remaining, state.walls, state.current_player, goals);
  }

  function clone(s) {
    return {
      np: s.np,
      pawns: s.pawns.map(p => [p[0], p[1]]),
      wr: s.wr.slice(),
      walls: Object.assign({}, s.walls),
      centers: Object.assign({}, s.centers),
      hb: Object.assign({}, s.hb),
      vb: Object.assign({}, s.vb),
      cp: s.cp,
      goals: s.goals,
    };
  }

  // ── Movement ──

  function canStep(s, r1, c1, r2, c2) {
    if (r2 < 0 || r2 >= B || c2 < 0 || c2 >= B) return false;
    const dr = r2 - r1, dc = c2 - c1;
    if (dr === 1)  return !s.hb[`${r1},${c1}`];
    if (dr === -1) return !s.hb[`${r2},${c2}`];
    if (dc === 1)  return !s.vb[`${r1},${c1}`];
    if (dc === -1) return !s.vb[`${r2},${c2}`];
    return false;
  }

  function reachedGoal(s, p) {
    const [r, c] = s.pawns[p];
    const g = s.goals[p];
    return (g.row !== undefined && r === g.row) || (g.col !== undefined && c === g.col);
  }

  function getPawnMoves(s, player) {
    const [r, c] = s.pawns[player];
    const occupied = new Set(s.pawns.map(p => `${p[0]},${p[1]}`));
    const moves = [];

    for (const [dr, dc] of DIRS) {
      const nr = r + dr, nc = c + dc;
      if (!canStep(s, r, c, nr, nc)) continue;

      if (!occupied.has(`${nr},${nc}`)) {
        moves.push([nr, nc]);
      } else {
        // Jump
        const jr = nr + dr, jc = nc + dc;
        if (jr >= 0 && jr < B && jc >= 0 && jc < B &&
            canStep(s, nr, nc, jr, jc) && !occupied.has(`${jr},${jc}`)) {
          moves.push([jr, jc]);
        } else {
          // Diagonal jumps
          for (const [ddr, ddc] of DIRS) {
            if ((ddr === dr && ddc === dc) || (ddr === -dr && ddc === -dc)) continue;
            const diag_r = nr + ddr, diag_c = nc + ddc;
            if (diag_r >= 0 && diag_r < B && diag_c >= 0 && diag_c < B &&
                canStep(s, nr, nc, diag_r, diag_c) && !occupied.has(`${diag_r},${diag_c}`)) {
              moves.push([diag_r, diag_c]);
            }
          }
        }
      }
    }
    return moves;
  }

  // ── Walls ──

  function wallHasConflict(s, r, c, o) {
    if (r < 0 || r >= W || c < 0 || c >= W) return true;
    if (s.centers[`${r},${c}`]) return true;
    if (o === "H") {
      return !!s.walls[`${r},${c-1},H`] || !!s.walls[`${r},${c+1},H`];
    } else {
      return !!s.walls[`${r-1},${c},V`] || !!s.walls[`${r+1},${c},V`];
    }
  }

  function addWallEdges(s, r, c, o) {
    if (o === "H") {
      s.hb[`${r},${c}`] = true;
      s.hb[`${r},${c+1}`] = true;
    } else {
      s.vb[`${r},${c}`] = true;
      s.vb[`${r+1},${c}`] = true;
    }
  }

  function removeWallEdges(s, r, c, o) {
    if (o === "H") {
      delete s.hb[`${r},${c}`];
      delete s.hb[`${r},${c+1}`];
    } else {
      delete s.vb[`${r},${c}`];
      delete s.vb[`${r+1},${c}`];
    }
  }

  function hasPath(s, player) {
    const [sr, sc] = s.pawns[player];
    const g = s.goals[player];
    const visited = new Set([`${sr},${sc}`]);
    const queue = [[sr, sc]];
    let qi = 0;
    while (qi < queue.length) {
      const [r, c] = queue[qi++];
      if ((g.row !== undefined && r === g.row) || (g.col !== undefined && c === g.col)) return true;
      for (const [dr, dc] of DIRS) {
        const nr = r + dr, nc = c + dc;
        const key = `${nr},${nc}`;
        if (!visited.has(key) && canStep(s, r, c, nr, nc)) {
          visited.add(key);
          queue.push([nr, nc]);
        }
      }
    }
    return false;
  }

  function isValidWall(s, r, c, o) {
    if (wallHasConflict(s, r, c, o)) return false;
    addWallEdges(s, r, c, o);
    let ok = true;
    for (let p = 0; p < s.np; p++) {
      if (!reachedGoal(s, p) && !hasPath(s, p)) { ok = false; break; }
    }
    removeWallEdges(s, r, c, o);
    return ok;
  }

  // ── BFS shortest path length ──

  function shortestPath(s, player) {
    const [sr, sc] = s.pawns[player];
    const g = s.goals[player];
    const dist = {};
    dist[`${sr},${sc}`] = 0;
    const queue = [[sr, sc]];
    let qi = 0;
    while (qi < queue.length) {
      const [r, c] = queue[qi++];
      const d = dist[`${r},${c}`];
      if ((g.row !== undefined && r === g.row) || (g.col !== undefined && c === g.col)) return d;
      for (const [dr, dc] of DIRS) {
        const nr = r + dr, nc = c + dc;
        const key = `${nr},${nc}`;
        if (dist[key] === undefined && canStep(s, r, c, nr, nc)) {
          dist[key] = d + 1;
          queue.push([nr, nc]);
        }
      }
    }
    return 99;
  }

  // ── Move application ──

  function applyMove(s, move) {
    if (move.type === "move") {
      s.pawns[s.cp] = [move.to[0], move.to[1]];
    } else {
      const [r, c, o] = [move.pos[0], move.pos[1], move.orient];
      s.walls[`${r},${c},${o}`] = true;
      s.centers[`${r},${c}`] = true;
      addWallEdges(s, r, c, o);
      s.wr[s.cp]--;
    }
    // Check winner
    if (reachedGoal(s, s.cp)) {
      s.winner = s.cp;
    } else {
      s.cp = (s.cp + 1) % s.np;
    }
  }

  // ── Candidate wall generation (pruned) ──

  function getCandidateWalls(s) {
    if (s.wr[s.cp] <= 0) return [];

    const candidates = new Set();

    // Near all pawns
    for (const [pr, pc] of s.pawns) {
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const wr = pr + dr, wc = pc + dc;
          if (wr >= 0 && wr < W && wc >= 0 && wc < W) candidates.add(`${wr},${wc}`);
        }
      }
    }

    // Along opponent shortest paths
    for (let p = 0; p < s.np; p++) {
      if (p === s.cp) continue;
      // Quick BFS to get path cells
      const [sr, sc] = s.pawns[p];
      const g = s.goals[p];
      const prev = {};
      prev[`${sr},${sc}`] = null;
      const queue = [[sr, sc]];
      let qi = 0, goalCell = null;
      while (qi < queue.length && !goalCell) {
        const [r, c] = queue[qi++];
        if ((g.row !== undefined && r === g.row) || (g.col !== undefined && c === g.col)) {
          goalCell = [r, c];
          break;
        }
        for (const [dr, dc] of DIRS) {
          const nr = r + dr, nc = c + dc;
          const key = `${nr},${nc}`;
          if (prev[key] === undefined && canStep(s, r, c, nr, nc)) {
            prev[key] = `${r},${c}`;
            queue.push([nr, nc]);
          }
        }
      }
      if (goalCell) {
        let key = `${goalCell[0]},${goalCell[1]}`;
        while (key) {
          const [r, c] = key.split(",").map(Number);
          for (let dr = -1; dr <= 0; dr++) {
            for (let dc = -1; dc <= 0; dc++) {
              const wr = r + dr, wc = c + dc;
              if (wr >= 0 && wr < W && wc >= 0 && wc < W) candidates.add(`${wr},${wc}`);
            }
          }
          key = prev[key];
        }
      }
    }

    const walls = [];
    for (const cand of candidates) {
      const [r, c] = cand.split(",").map(Number);
      if (isValidWall(s, r, c, "H")) walls.push({type: "wall", pos: [r, c], orient: "H"});
      if (isValidWall(s, r, c, "V")) walls.push({type: "wall", pos: [r, c], orient: "V"});
    }
    return walls;
  }

  // ── Evaluation ──
  //
  // Two signals:
  //   - Path difference (opp shortest path − my shortest path). Dominant signal.
  //   - Wall resource (net walls in hand). Secondary — walls are valuable
  //     because they can lengthen the opponent's future path.
  //
  // No hand-tuned positional heuristics. Strength comes from search depth.

  function evaluate(s, aiPlayer) {
    if (s.winner !== undefined && s.winner !== null) {
      return s.winner === aiPlayer ? 10000 : -10000;
    }

    const myDist = shortestPath(s, aiPlayer);
    let score = 0;
    for (let p = 0; p < s.np; p++) {
      if (p === aiPlayer) continue;
      const theirDist = shortestPath(s, p);
      score += (theirDist - myDist) * 10;
      score += (s.wr[aiPlayer] - s.wr[p]) * 2;
    }
    return score;
  }

  // ── Minimax with alpha-beta ──
  //
  // Takes an optional `orderedMoves` for the root — a pre-sorted list from a
  // previous iteration's result. Good move ordering dramatically improves
  // alpha-beta pruning (best-first ordering gives O(b^(d/2)) instead of O(b^d)).

  function minimax(s, depth, alpha, beta, aiPlayer, maximizing, orderedMoves = null) {
    if (depth === 0 || (s.winner !== undefined && s.winner !== null)) {
      return { score: evaluate(s, aiPlayer), move: null };
    }

    let allMoves;
    if (orderedMoves) {
      allMoves = orderedMoves;
    } else {
      const pawnMoves = getPawnMoves(s, s.cp).map(to => ({ type: "move", to }));
      const wallMoves = getCandidateWalls(s);
      allMoves = [...pawnMoves, ...wallMoves];
    }

    if (allMoves.length === 0) return { score: evaluate(s, aiPlayer), move: null };

    let bestMove = allMoves[0];

    if (maximizing) {
      let maxScore = -Infinity;
      for (const move of allMoves) {
        const child = clone(s);
        applyMove(child, move);
        const { score } = minimax(child, depth - 1, alpha, beta, aiPlayer, child.cp === aiPlayer);
        if (score > maxScore) { maxScore = score; bestMove = move; }
        alpha = Math.max(alpha, score);
        if (beta <= alpha) break;
      }
      return { score: maxScore, move: bestMove };
    } else {
      let minScore = Infinity;
      for (const move of allMoves) {
        const child = clone(s);
        applyMove(child, move);
        const { score } = minimax(child, depth - 1, alpha, beta, aiPlayer, child.cp === aiPlayer);
        if (score < minScore) { minScore = score; bestMove = move; }
        beta = Math.min(beta, score);
        if (beta <= alpha) break;
      }
      return { score: minScore, move: bestMove };
    }
  }

  // ── Move ordering helpers ──

  function orderMovesAtRoot(s, aiPlayer) {
    // Pawn moves first, sorted by how much they reduce our distance to goal.
    // Then walls, sorted by how much they increase opponent's distance.
    // This heuristic ordering makes alpha-beta much more effective.
    const myDistBefore = shortestPath(s, aiPlayer);
    const opp = (aiPlayer + 1) % s.np;
    const oppDistBefore = shortestPath(s, opp);

    const pawnMoves = getPawnMoves(s, aiPlayer).map(to => {
      const child = clone(s);
      child.pawns[aiPlayer] = to;
      const reduction = myDistBefore - shortestPath(child, aiPlayer);
      return { move: { type: "move", to }, score: reduction };
    });
    pawnMoves.sort((a, b) => b.score - a.score);

    const wallCandidates = getCandidateWalls(s);
    const scoredWalls = wallCandidates.map(move => {
      const child = clone(s);
      applyMove(child, move);
      const impact = shortestPath(child, opp) - oppDistBefore;
      return { move, score: impact };
    });
    scoredWalls.sort((a, b) => b.score - a.score);

    return [...pawnMoves.map(x => x.move), ...scoredWalls.map(x => x.move)];
  }

  // ── Public API ──

  /**
   * Pick the best move using alpha-beta minimax with root move ordering.
   *
   * Move ordering: at the root we sort pawn moves by how much they shorten
   * our path, and walls by how much they lengthen the opponent's. This makes
   * alpha-beta cut off weaker branches much faster.
   *
   * Safeguards:
   *   - Anti-oscillation: among near-tied moves, prefer those that reduce our
   *     shortest path. Stops back-and-forth behavior when minimax sees "move
   *     forward then back" as equal to "move back then forward".
   *   - Light randomness: among moves within 1 point of the best, pick one at
   *     random. Different games play out differently.
   *
   * @param {Object} serverState - State object from the server API
   * @param {number} depth - Search depth (1=easy, 2=medium, 3=hard)
   */
  function searchAtDepth(s, depth, aiPlayer, moves, myDistBefore) {
    let alpha = -Infinity;
    const scored = [];
    for (const move of moves) {
      const child = clone(s);
      applyMove(child, move);
      const { score } = minimax(child, depth - 1, alpha, Infinity, aiPlayer, child.cp === aiPlayer);
      alpha = Math.max(alpha, score);
      let progress = 0;
      if (move.type === "move") {
        progress = myDistBefore - shortestPath(child, aiPlayer);
      }
      scored.push({ move, score, progress });
    }
    return scored;
  }

  function bestMove(serverState, maxDepth = 2, timeLimitMs = 1500) {
    const s = fromServerState(serverState);
    const aiPlayer = s.cp;
    const myDistBefore = shortestPath(s, aiPlayer);

    let moves = orderMovesAtRoot(s, aiPlayer);
    if (moves.length === 0) return null;

    // Iterative deepening: start shallow, go deeper while within time budget.
    // Each completed iteration gives us better root move ordering for the
    // next (best moves first → more alpha-beta cuts at deeper levels).
    const startTime = Date.now();
    let bestScored = null;
    for (let depth = 1; depth <= maxDepth; depth++) {
      const iterStart = Date.now();
      const scored = searchAtDepth(s, depth, aiPlayer, moves, myDistBefore);

      // Reorder moves by score for the next iteration
      scored.sort((a, b) => b.score - a.score);
      moves = scored.map(x => x.move);
      bestScored = scored;

      // Stop if a forced win/loss is found
      if (scored[0].score >= 10000 || scored[0].score <= -10000) break;

      // Stop if estimated next iteration would exceed the time budget.
      // Each deeper level is roughly 6-10x slower; use 8x as a conservative estimate.
      const iterTime = Date.now() - iterStart;
      const elapsed = Date.now() - startTime;
      if (elapsed + iterTime * 8 > timeLimitMs) break;
    }

    // Pick best with anti-oscillation + light randomness
    const maxScore = Math.max(...bestScored.map(x => x.score));
    const margin = 1;
    const nearBest = bestScored.filter(x => x.score >= maxScore - margin);
    const maxProgress = Math.max(...nearBest.map(x => x.progress));
    const finalists = nearBest.filter(x => x.progress === maxProgress);
    return finalists[Math.floor(Math.random() * finalists.length)].move;
  }

  return { bestMove };
})();
