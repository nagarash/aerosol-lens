# Aerosol Lens

Ask a question about atmospheric aerosols and particulates in plain
language; an AI agent translates it into a data query and renders the
answer as a concentration map overlaid on a world map.

One view: **column aerosol optical depth**, speciated (dust, black
carbon, organic carbon, sulfate, sea salt, total) from the MERRA-2
reanalysis archive — *where is the plume going?* Multi-day questions
play as an animated video by default.

Health / surface air-quality questions are **not** servable in this
build (the Google Air Quality path was removed; CAMS is not wired up)
and return an honest `422`, never column data dressed up as an answer.

## Architecture

```
 ┌──────────┐   question    ┌──────────────────────────────────┐
 │ frontend │ ────────────▶ │ backend  POST /ask                │
 │ MapLibre │               │  1. PlanCache (in-proc, per-day)  │
 │  (Pages) │ ◀──────────── │  2. Jev fast path (OpenRouter)   │
 └────┬─────┘  plan+urls     │  3. LLM fallback (LiteLLM)       │
      │                     │  4. validate_plan()  ← GUARDRAILS │
      │                     │  5. dateless → newest data clamp  │
      │                     └──────────────────────────────────┘
      │ daily frames (/grid × N)
      ▼
 ┌────────────────────────────────────────────────────────────┐
 │ GET /grid                                                  │
 │  1. disk cache (500 MB LRU; repeats in ms)                 │
 │  2. Zarr daily-mean field store (/data/fields, if covered) │
 │  3. kerchunk byte-range reads of MERRA-2 NetCDF (fallback) │
 │  parallel per-day downloads; 429-aware per-IP rate limit   │
 └────────────────────────────────────────────────────────────┘
```

**`/ask` pipeline.** The TypeSafe `jev-1.13` decision model (via
OpenRouter, `JEV_ENABLED=1`, `JEV_MIN_CONFIDENCE=0.7` default) classifies
intent/variable/aggregation in one request (~100–500 ms); place and time
are extracted deterministically (local gazetteer, rule-based dates).
Low confidence, unresolvable place, or any Jev failure falls back to the
existing LiteLLM planner — never a 500. Repeated questions hit the
in-process plan cache (scoped by day + location).

**Dateless questions mean "latest available".** MERRA-2 runs several
weeks behind real time (`MERRA2_LATENCY_DAYS`, default 45), so a
default-to-today window would 422. When the question carries no time
expression, `/ask` clamps the plan to the newest plausible granule date
(`backend/grid.py::newest_plausible_date`, shared with the `/grid`
latency gate). Explicit dates — including "today" — are never touched:
asking for a day with no data still fails honestly.

**Hard guardrails** (in `agent/validator.py` + `backend/app.py`, not in
prompts):

- `intent="health"` → `422` ("not available in this build"). Column AOD
  can never answer a breathing-safety question.
- The Google Air Quality source no longer exists in the schema or
  routing.

## MERRA-2 column-AOD grids (`/grid`)

`/grid?source=merra2&variable=DUEXTTAU&bbox=w,s,e,n&t0=..&t1=..&agg=daily`
serves one aggregated 2D field as compact JSON (`lats[]`, `lons[]`,
`values[][]`, rounded to 4 decimals, capped at 360×180 cells). The
response reports `data_source` (`"fields"` or `"kerchunk"`) and
`cache_hit`.

Source facts (collection `tavg1_2d_aer_Nx`, short name `M2T1NXAER`):

- Reads go to the **GES DISC HTTPS archive** with the Earthdata bearer
  token (`EARTHDATA_TOKEN`, server-side only — Fly.io secret in
  production, never the repo or frontend). S3 is auto-selected from the
  index when the target is same-region.
- Hourly, single-level NetCDF-4; one file per day, 24 steps at
  `00:30`–`23:30 UTC`; grid 576 lon × 361 lat.
- Servable variables (550 nm aerosol extinction AOT, dimensionless):
  `TOTEXTTAU DUEXTTAU BCEXTTAU OCEXTTAU SUEXTTAU SSEXTTAU`.
- Legacy shorthand aliases (`DUAOD`, `TOTAOD`, …) are accepted and
  canonicalized — see `agent/mappings.py`.

**Latency layers** (each is a real measured win, in order):

1. **Disk cache** (`backend/grid_cache.py`): identical queries served
   from the Fly volume in milliseconds. 500 MB cap
   (`GRID_CACHE_MAX_MB`), oldest-first eviction; historical entries
   never expire, recent-edge windows refresh after 24 h.
2. **Zarr daily-mean field store** (`backend/fields.py`): one store per
   variable under `FIELDS_DIR` (default `/data/fields`), shape
   `(time, lat, lon)` float32, daily chunks with Blosc/Zstd. `/grid`
   serves from it only when *every* requested date is covered locally;
   otherwise the whole query falls back to kerchunk.
3. **Parallel per-day downloads** on the kerchunk path.
4. **Rate limiting**: per-IP sliding window (`GRID_RATE_LIMIT_PER_MIN`,
   default 30/min) with `429` + `Retry-After` — every request burns the
   deployer's Earthdata quota.

**Backfill** (admin only): `POST /admin/backfill` with JSON body `{}`
and `Authorization: Bearer $ADMIN_TOKEN` starts a background thread that
fills the field store day by day (default: last full year of archive,
~1.7 GB); safe to rerun, skips completed days, survives restarts by
re-running. `GET /admin/backfill/status` reports progress. Without
`ADMIN_TOKEN` set, both endpoints 404.

**Kerchunk index** (legacy path, still the fallback): build once with
`python backend/kerchunk_index.py --out data/kerchunk` (needs Earthdata
credentials in the environment); `/grid` reads
`KERCHUNK_INDEX_PATH` (default `/data/kerchunk/index.json`).

Other notes:

- The time window collapses to ONE 2D field: `agg=hourly` averages the
  hourly steps; `daily` averages daily means; `monthly_mean` averages
  monthly means.
- Windows entirely newer than the archive edge return 422 saying so
  plainly; tune with `MERRA2_LATENCY_DAYS` (default 45).
- Antimeridian-crossing bboxes (`w > e`) are served as two slices
  concatenated along lon; `lons[]` wraps accordingly.

## Frontend

Static single-page app (MapLibre GL JS via CDN — no build step, no npm).
Ask a question in the chat box; the app POSTs to `/ask`, then renders
the returned plan:

- **Plume rendering**: the plan's `/grid` URL is fetched, mapped through
  a colormap onto an offscreen canvas (NaN/missing → transparent,
  feathered bbox edges, low values fade out), then upscaled 2× with
  smoothing + slight blur so the plume reads as an organic cloud rather
  than a blocky grid. Added as a MapLibre `image` source fit to the grid
  bounds. The colorbar legend is **data-driven**: gradient + min/mid/max
  from the fetched values (p2/p98 normalization so hot-pixel outliers
  don't crush the ramp), with units, variable name, and time range.
  Colormaps: yellow-orange-red `dust` ramp for `DU*` (dust AOD),
  viridis-like default otherwise.
- **Video by default**: multi-day plans fetch one frame per day (capped
  at 31, 3-at-a-time, honoring `429`/`Retry-After`) and autoplay as a
  looping animation. All frames share one p2/p98 color scale so colors
  stay comparable over time. The timeline scrubs frames (scrubbing
  pauses); the play button toggles. Single-day plans render statically.
- **Basemaps** (all keyless): Esri Satellite (default), Streets, NatGeo,
  Ocean (+ reference labels), Terrain, Light, Dark; OpenFreeMap Positron
  and Bright. Vector styles switch via `setStyle()` with the plume layer
  re-added on style load.
- Backend errors surface plainly in the error banner (422/501/502
  messages, 429 with `Retry-After`), never as stack traces.

Pure rendering logic (colormaps, normalization, grid→pixel buffer) lives
in `frontend/js/grid-render.js`, written DOM-free and covered by
`node frontend/js/grid-render.test.js`.

### Running locally

```bash
cd frontend && python3 -m http.server 8080
# then open http://localhost:8080
```

Point the app at your backend in `frontend/config.js`:

```js
window.AEROSOL_LENS_CONFIG = { BACKEND_URL: "http://localhost:8000" };
```

### Deploying to Cloudflare Pages

1. Set `BACKEND_URL` in `frontend/config.js` to your deployed backend
   (e.g. `https://<your-app>.fly.dev`).
2. Deploy with `wrangler pages deploy <dir> --project-name aerosol-lens`.
   Wrangler 4.x infers preview vs production from the git branch — add
   `--branch main` to promote a direct upload to production.
3. No environment variables or build command needed on Pages. After a
   visible frontend change, ship the bundle under a **new filename**
   (aggressive client caching) and update `index.html` accordingly.

## Backend deployment (Fly.io)

```bash
fly deploy -a aerosol-lens-api   # from the repo root; fly.toml is there
```

Single `shared-cpu-1x` machine, 1 GB RAM, one attached volume mounted at
`/data` (holds the Zarr field store, kerchunk index, and grid cache).
Secrets: `EARTHDATA_TOKEN` (MERRA-2 archive access), `ADMIN_TOKEN`
(backfill endpoints). Note: Fly blocks the sandbox's egress IPs, so
`flyctl` must run from a real machine.

## Repo layout

| Path | What |
|---|---|
| `agent/` | QueryPlan schema, deterministic validator, prompts, mapping tables |
| `backend/` | FastAPI app (`app.py`), Jev fast path (`jev.py`), `/grid` slicer (`grid.py`), Zarr field store (`fields.py`), backfill (`backfill.py`), disk cache (`grid_cache.py`), rate limiter (`rate_limit.py`), kerchunk builder (`kerchunk_index.py`) |
| `frontend/` | Single-page MapLibre app (CDN), chat box, legend, timeline |
| `data/` | Gazetteer + event catalog + colormap samples |

## Configuration

| Variable | Default | What |
|---|---|---|
| `EARTHDATA_TOKEN` | — | Earthdata Login bearer token (server-side only) |
| `ADMIN_TOKEN` | — | Bearer token for `/admin/*` (unset → 404) |
| `LITELLM_MODEL` | — | LLM for the `/ask` fallback path |
| `OPENROUTER_API_KEY` | — | Jev classifier via OpenRouter |
| `JEV_ENABLED` | `1` | Kill switch for the Jev fast path |
| `JEV_MIN_CONFIDENCE` | `0.7` | Below this → LLM fallback |
| `MERRA2_LATENCY_DAYS` | `45` | Archive lag; drives the dateless clamp + `/grid` gate |
| `GRID_RATE_LIMIT_PER_MIN` | `30` | Per-IP `/grid` rate limit |
| `GRID_CACHE_MAX_MB` | `500` | Disk-cache cap on the Fly volume |
| `FIELDS_DIR` | `/data/fields` | Zarr daily-mean store root |
| `KERCHUNK_INDEX_PATH` | `/data/kerchunk/index.json` | Kerchunk manifest (fallback path) |

## Status

Live: Cloudflare Pages frontend + Fly.io backend (`aerosol-lens-api`).
Implemented: Jev `/ask` fast path with LLM fallback, dateless→latest
clamp, `/grid` disk cache + Zarr field store + admin backfill, parallel
daily downloads, per-IP rate limiting, video autoplay for multi-day
plans, 9 keyless basemaps, data-driven colorbar legend.
Still stubbed: the CAMS fetcher, OpenAQ overlay. Health/surface
questions intentionally return 422.

## License

MIT — see `LICENSE`.
