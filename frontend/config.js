/* Aerosol Lens frontend config.
 *
 * BACKEND_URL: where the FastAPI backend lives.
 * - Local dev:        "http://localhost:8000" (backend on your machine)
 * - Docker Compose:   "" (same origin; nginx serves both) -- leave as ""
 * - Fly.io / Cloudflare Pages / any static host: the deployed backend's
 *   URL. Cross-origin is fine -- the backend's CORS allows any origin.
 *
 * Set this to your deployed backend URL before publishing (or inject it
 * at build time) if it differs from the default below.
 */
window.AEROSOL_LENS_CONFIG = {
  BACKEND_URL: "https://aerosol-lens-api.fly.dev",
};
