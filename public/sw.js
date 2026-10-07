// Bump on every asset change — activate() deletes older caches.
const CACHE_NAME = "quoridor-v5";

// Everything a local game needs. Local play (pass-and-play and vs computer)
// runs entirely client-side, so with these cached the installed app is fully
// playable offline — including "best move", which is why the AI files are here.
const ASSETS = [
  "/",
  "/index.html",
  "/ai.js",
  "/ai-worker.js",
  "/local-game.js",
  "/how-it-thinks.html",
  "/manifest.json",
  "/icon-192.png",
  "/icon-512.png",
];

const NET_TIMEOUT = 2500;  // ms before falling back to a cached copy

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then((c) => c.addAll(ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);

  // Never cache API calls — online rooms must always see live state.
  if (url.pathname.startsWith("/api/")) return;
  if (e.request.method !== "GET") return;

  // Every load of the app — "/", "/?room=ab12", "/?utm_source=…" — is the same
  // index.html, so it is stored and looked up under "/". Matching the full URL
  // meant a room link never opened before failed offline with the browser's
  // error page (instead of the app's local-game fallback), and the cache kept
  // a copy of the page for every room.
  const appPage = e.request.mode === "navigate" && url.origin === self.location.origin &&
    (url.pathname === "/" || url.pathname === "/index.html");
  const key = appPage ? "/" : e.request;

  e.respondWith(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(key);

      const network = fetch(e.request).then((res) => {
        // Keep the cache warm for next launch. Opaque cross-origin responses
        // (the web font) can't always be stored, so this is best-effort.
        if (res && (res.ok || res.type === "opaque")) {
          cache.put(key, res.clone()).catch(() => {});
        }
        return res;
      });
      e.waitUntil(network.catch(() => {}));

      if (!cached) return network;

      // With a cached copy in hand, don't let a dead or crawling connection
      // hold up the launch — captive-portal wifi that accepts the connection
      // and then never answers used to hang the page. The fetch keeps running,
      // so the next launch picks up the fresh file.
      // A server error counts as no answer too: the cached copy beats an
      // error page.
      return Promise.race([
        network.then((res) => (res.ok || res.type === "opaque" ? res : cached), () => cached),
        new Promise((resolve) => setTimeout(() => resolve(cached), NET_TIMEOUT)),
      ]);
    })()
  );
});
