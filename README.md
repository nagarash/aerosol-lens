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


## Hourly plume pipeline (opt-in)

See [the implementation spec](docs/hourly-plume-plan.md). The existing daily
pipeline remains the default until hourly coverage has been ingested. Set
`HOURLY_PLUMES_ENABLED=1` to route `/ask` through the new local-first parser.
`JEV_ENABLED` applies only to the legacy route; the hourly route never calls Jev.

- Straightforward queries resolve locally. An unresolved query uses at most one
  general LLM call through `LITELLM_MODEL` (10-second timeout, no repair retry).
  Validated interpretations have a bounded in-process cache scoped by model,
  question, reference date, location, and prompt; availability is resolved anew.
  The model interprets names/events/dates; local code supplies coordinates and
  data availability. Without a configured model, clear queries still work.
- Recent wording uses the latest contiguous published hours for the selected
  variable, defaulting to 48 frames. Actual dates are always displayed. Explicit
  historical dates are preserved. Ranges beyond seven days initially show their
  first 48 hours and say so; comparisons and forecasts request clarification.
- The event catalog is now loaded for hourly routing. Initial event playback is
  the first 48 catalog hours, **not** a claimed peak or representative period.
- `/frames` provides timestamps, native coordinates, and six-hour batch URLs.
  `/frames/batch` returns gzip-compressed little-endian float32 arrays in
  `(time, lat, lon)` order; NaN means missing. The manifest fixes the display
  scale at AOD 0–1, with higher values saturated. Numeric data are not clipped.
- `frontend/app16.js` displays the first batch immediately and loads the rest
  sequentially while playing. It reuses a MapLibre canvas source, supports
  pause/scrub/basemap changes, and aborts obsolete requests.

### Ingestion and retention

The hourly store is separate from the existing **daily-mean** field store.
No all-history download is needed. First prepare a kerchunk manifest using the
existing index builder, then ingest its latest seven indexed days:

```bash
export HOURLY_FIELDS_DIR=/data/hourly
python -m backend.hourly
# Or a specific date range / one variable:
python -m backend.hourly --start 2020-06-14 --end 2020-06-15 --variables DUEXTTAU
# Preload and pin an event's first 48 hours:
python -m backend.hourly --events sahara-dust-godzilla-2020
```

Run incremental indexing followed by this idempotent command from your existing
job scheduler as new granules arrive. No scheduler or production backfill is
installed by this change. Existing Earthdata authentication is reused.

Each variable-day is a complete 24-hour Zarr array with `(6, 91, 144)` chunks,
validated and atomically published. The newest seven stored days per variable
are retained; other historical days use an LRU budget (default 2 GB), excluding
pinned dates. A rolling seven-day window across six variables is approximately
840 MB **uncompressed**, plus historical cache and pins. Reserve volume space
for both and monitor pins separately. Environment controls:

| Variable | Default | Purpose |
|---|---|---|
| `HOURLY_PLUMES_ENABLED` | `0` | Opt into hourly `/ask` routing |
| `HOURLY_FIELDS_DIR` | `/data/hourly` | Hourly Zarr store |
| `HOURLY_RETAIN_DAYS` | `7` | Protected recent days during cold-request eviction |
| `HOURLY_CACHE_MAX_MB` | `2048` | Historical cache budget, excluding recent/pinned data |

CLI retention flags control eviction after ingestion. Cold historical requests
fetch and cache a complete variable-day, then crop it for playback. This trades
higher first-request transfer for reuse across hours and viewports. Archive
fetches are serialized across processes to bound memory on the 1 GB server;
warm reads do not wait on the archive lock. This first implementation does not
claim optimized cold-download latency or a benchmark-selected chunk shape.
Missing indexed days fail explicitly rather than skipping gaps unnoticed.

### Verification and evals

```bash
pip install -r backend/requirements-test.txt
python -m pytest -q backend/test_hourly_plumes.py
python -m backend.eval_hourly                  # offline stubs, no NASA/model calls
python -m backend.eval_hourly --live --model openrouter/YOUR_MODEL
node frontend/js/hourly-frames.test.js
node frontend/js/grid-render.test.js
```

The live eval requires provider credentials and incurs provider charges. It
uses fixed coverage to isolate routing quality, reports each call's usage and
latency, and returns nonzero on failures. Offline results test routing and
validation; they are **not** evidence of any real model's accuracy. The older
`eval_llm_fallback.py` remains a legacy-parser evaluation only.

The response includes routing milliseconds; batch responses expose a
`Server-Timing` header. Browser Performance entries `plume-first-frame-ms` and
`plume-buffered-ms` measure query-to-display and query-to-full-buffer latency.
Use these to compare warm/cold paths; they are not production benchmarks.

Browser implementation follows the [MapLibre canvas-source API](https://maplibre.org/maplibre-gl-js/docs/API/type-aliases/CanvasSourceSpecification/).

## Legacy daily architecture

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
| `backend/` | FastAPI app (`app.py`), Jev fast path (`jev.py`), `/grid` slicer (`grid.py`), Zarr field store (`fields.py`), backfill (`backfill.py`), disk cache (`grid_cache.py`), rate limiter (`rate_limit.py`), kerchunk builder (`kerchunk_index.py`), gazetteer builders (`gazetteer_build.py`, `us_states_build.py`), LLM fallback eval harness (`eval_llm_fallback.py`) |
| `frontend/` | Single-page MapLibre app (CDN), chat box, legend, timeline |
| `data/` | Gazetteer + event catalog + colormap samples |

### Place gazetteer (`backend/geocode.py`)

Place names never reach the model as coordinates: the LLM/Jev returns a
name, `backend/geocode.py` resolves it to a bbox from a **local**
gazetteer (no geocoding API call), and unresolvable names 422 instead of
being guessed at. Two files back this, with different roles:

- `data/gazetteer.sample.json` — hand-curated, ~68 places: the original
  ~16 regions/cities plus all 50 US states + DC + Puerto Rico (+ Tonga)
  from `python backend/us_states_build.py` (needs `pip install pyshp`;
  pulls real boundaries from the US Census Bureau + Natural Earth, not
  hand-typed bboxes — see that file's docstring for why, including the
  Alaska antimeridian handling). The *only* source for
  `known_places()`/`known_places_hint()`, which is appended to every LLM
  fallback prompt (`agent/prompts.py`) — keep this small and reviewed;
  its size is a direct token-cost/latency knob. Curated names always win
  a collision with the extended (GeoNames) layer below, which is how
  common names like "Georgia" resolve to the expected US state instead
  of the country.
- `data/gazetteer_extended.json` (generated, committed) — ~2,050
  well-known GeoNames cities (population ≥ 500,000 by default) + named
  physical regions (deserts, basins, mountain ranges — a much lower
  notability floor for deserts specifically, since there are few enough
  of them worldwide that even regional ones are dust-relevant, vs.
  thousands of obscure mountain ridges), built by
  `python backend/gazetteer_build.py`. Deliberately NOT a general
  GeoNames dump — see that file's docstring for why and for the
  notability-floor tuning. Merged into local name matching only
  (`jev.py::extract_place`, `geocode.resolve_place`), never into the LLM
  prompt. A curated name always wins a collision with an extended one.
  Regenerate with:

  ```bash
  python backend/gazetteer_build.py                       # well-known cities + regions (~420MB GeoNames pull)
  python backend/gazetteer_build.py --cities-only          # skip the big pull, cities only
  python backend/gazetteer_build.py --min-city-population 100000  # broader coverage (~4,300 cities)
  ```

  Install `pyahocorasick` (in `backend/requirements.txt`) — `extract_place()`
  falls back to a per-name regex scan without it, which is meaningfully
  slower once GeoNames names are merged in on top of the curated set.

### LLM fallback eval (`backend/eval_llm_fallback.py`)

A fixed set of hard hand-picked questions (multi-place disambiguation,
named events, figurative phrasing, health refusals, places only the
extended gazetteer knows) run through the real `_parse_with_agent` code
path — same prompt, same retries, same guardrails — against whichever
candidate models you list, to compare LLM choices on cost/quality instead
of guessing:

```bash
export OPENROUTER_API_KEY=...
python backend/eval_llm_fallback.py
```

Costs real API calls (up to 2 per case per model); see the file's
docstring for the two real gaps it surfaced (event catalog never loaded
into the prompt; US state name collisions, now fixed).

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
