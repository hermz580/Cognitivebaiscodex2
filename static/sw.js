// Service Worker — Cognitive Bias Codex
// Strategy: Cache-first for app shell; network-first for API calls.

const CACHE_NAME = "bias-codex-v1";

// Static assets to pre-cache (app shell)
const SHELL = [
  "/",
  "/static/manifest.json",
];

// ── Install: pre-cache app shell ──────────────────────────────────────────────
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL))
  );
  self.skipWaiting();
});

// ── Activate: remove old caches ───────────────────────────────────────────────
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

// ── Fetch: routing strategy ───────────────────────────────────────────────────
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Always go to network for API calls and camera/media
  if (
    url.pathname.startsWith("/search") ||
    url.pathname.startsWith("/analyze") ||
    url.pathname.startsWith("/result") ||
    url.pathname.startsWith("/auth") ||
    url.pathname.startsWith("/stats") ||
    url.pathname.startsWith("/health") ||
    url.pathname.startsWith("/metrics")
  ) {
    event.respondWith(fetch(request));
    return;
  }

  // Cache-first for everything else (HTML, CSS, fonts, manifest)
  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request).then((response) => {
        // Only cache successful, same-origin GET responses
        if (
          response.ok &&
          request.method === "GET" &&
          url.origin === self.location.origin
        ) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((c) => c.put(request, clone));
        }
        return response;
      });
    })
  );
});
