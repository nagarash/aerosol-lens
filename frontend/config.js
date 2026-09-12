/* Aerosol Lens frontend config.
 *
 * BACKEND_URL: where the FastAPI backend lives.
 * - Local dev:        "http://localhost:8000" (backend on your machine)
 * - Docker Compose:   "" (same origin; nginx serves both) -- leave as "" 
 * - Cloudflare Pages: "https://<your-backend>.fly.dev" (your Fly.io URL)
 *
 * On Cloudflare Pages, set this to your deployed backend URL before
 * publishing (or inject it at build time).
 */
window.AEROSOL_LENS_CONFIG = {
  BACKEND_URL: "http://localhost:8000",
};
