/**
 * Web Worker wrapper around the AI search.
 *
 * The "best move" hint searches to depth 6, which can run for tens of seconds.
 * On the main thread that froze the page — long enough that the browser put up
 * its "page unresponsive" dialog. Running it here keeps the UI live, so the
 * page can show elapsed time, the depth reached so far, and a cancel button.
 *
 * Protocol — main thread sends {id, state, depth, budget}; we reply with:
 *   {id, type: "progress", depth, score, move}   after each completed iteration
 *                                          (move = best so far, if stopped now)
 *   {id, type: "done", move, depth, score} when the search finishes
 *   {id, type: "error", message}           if it threw
 * `id` lets the page ignore replies from a search it has already cancelled.
 */

importScripts("/ai.js");

self.onmessage = function (e) {
  const { id, state, depth, budget } = e.data || {};
  try {
    const move = AI.bestMove(state, depth, budget, function (d, score, best) {
      self.postMessage({ id: id, type: "progress", depth: d, score: score, move: best });
    });
    self.postMessage({
      id: id,
      type: "done",
      move: move,
      depth: AI._lastDepth,
      score: AI._lastScore,
    });
  } catch (err) {
    self.postMessage({
      id: id,
      type: "error",
      message: String((err && err.message) || err),
    });
  }
};
