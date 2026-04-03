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

  function evaluate(s, aiPlayer) {
    // Check for winner
    if (s.winner !== undefined && s.winner !== null) {
      return s.winner === aiPlayer ? 10000 : -10000;
    }

    let score = 0;
    const myDist = shortestPath(s, aiPlayer);

    for (let p = 0; p < s.np; p++) {
      if (p === aiPlayer) continue;
      const theirDist = shortestPath(s, p);
      score += (theirDist - myDist) * 10;
      // Bonus for having more walls
      score += (s.wr[aiPlayer] - s.wr[p]) * 2;
    }

    return score;
  }

  // ── Minimax with alpha-beta ──

  function minimax(s, depth, alpha, beta, aiPlayer, maximizing) {
    if (depth === 0 || (s.winner !== undefined && s.winner !== null)) {
      return { score: evaluate(s, aiPlayer), move: null };
    }

    const pawnMoves = getPawnMoves(s, s.cp).map(to => ({type: "move", to}));
    const wallMoves = depth >= 1 ? getCandidateWalls(s) : [];  // Only consider walls at higher depths
    const allMoves = [...pawnMoves, ...wallMoves];

    if (allMoves.length === 0) return { score: evaluate(s, aiPlayer), move: null };

    // Move ordering: pawn moves first (they're usually more impactful per search node)
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

  // ── Public API ──

  /**
   * Pick the best move for the current player.
   * @param {Object} serverState - The state object from the server API
   * @param {number} depth - Search depth (1=easy, 2=medium, 3=hard)
   * @returns {Object} move - {type: "move", to: [r,c]} or {type: "wall", pos: [r,c], orient: "H"|"V"}
   */
  function bestMove(serverState, depth = 2) {
    const s = fromServerState(serverState);
    const aiPlayer = s.cp;
    const { move } = minimax(s, depth, -Infinity, Infinity, aiPlayer, true);
    return move;
  }

  return { bestMove };
})();
