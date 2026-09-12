# Aerosol Lens

Ask a question about air quality or atmospheric aerosols in plain language;
an AI agent translates it into a data query and renders the answer as a
concentration map overlaid on a world map.

Two views, never blended:

- **Air quality** — surface-level concentrations (PM2.5/PM10): *is it safe
  to breathe here?*
- **Plume view** — column aerosol optical depth, speciated (dust, black
  carbon, sulfate…): *where is the plume going?*

## Architecture

```
 ┌──────────┐   question    ┌─────────────────────────────────┐
 │ frontend │ ────────────▶ │ backend  POST /ask               │
 │ MapLibre │               │  1. PlanCache (in-proc LRU)       │
 │  (CDN)   │ ◀──────────── │  2. agent parse (LiteLLM, JSON)   │
 └────┬─────┘  plan+urls     │  3. validate_plan()  ← GUARDRAILS │
      │                     │  4. data URLs + legend + caption  │
      │                     └─────────────────────────────────┘
      │ tiles/grid
      ▼
 ┌────────────────────────────────────────────────────────────┐
 │ data paths                                                  │
 │  live (google)  ── browser calls Google Air Quality API     │
 │                   DIRECTLY with the user's key (BYOK;       │
 │                   backend never proxies it)                │
 │  archive (merra2) ── backend /grid via kerchunk byte-range │
 │                   reads of MERRA-2 NetCDF on AWS Open Data │
 │  archive (cams)   ── historical surface PM2.5 (stub)       │
 └────────────────────────────────────────────────────────────┘
```

**Latency design:** the LLM only emits a typed `QueryPlan` (~1s, cached by
normalized question). Live tiles come straight from Google (backend does
zero work). Archive reads are HTTP range requests of a few MB via a
one-time kerchunk index — no ETL, no data duplication. Preprocessed
COG + TiTiler is the documented upgrade path if tile traffic demands it.

**Hard guardrails** (in `agent/validator.py`, not in prompts):

- `intent="health"` **requires** `level="surface"` — column AOD can never
  answer a breathing-safety question.
- Surface and column are never mixed in one view (UI toggle, one plan).

## Quickstart

```bash
cp .env.example .env        # set LITELLM_MODEL and (optionally) keys
docker compose up --build
# frontend: http://localhost:8080   backend: http://localhost:8000
```

Without Docker:

```bash
pip install -r backend/requirements.txt
PYTHONPATH=. uvicorn backend.app:app --reload
# serve frontend/ with any static server, e.g. python -m http.server
```

## MERRA-2 column-AOD grids (`/grid`)

`/grid?source=merra2&variable=DUEXTTAU&bbox=w,s,e,n&t0=..&t1=..&agg=daily`
serves one aggregated 2D field as compact JSON (`lats[]`, `lons[]`,
`values[][]`, rounded to 4 decimals, capped at 360×180 cells).

Source facts (collection `tavg1_2d_aer_Nx`, short name `M2T1NXAER`,
verified 2026-09-12 via `https://registry.opendata.aws/nasa-merra-2/`):

- Bucket: `s3://gesdisc-cumulus-prod-protected/MERRA2` (region us-west-2),
  layout `M2T1NXAER.5.12.4/YYYY/MM/MERRA2_400.tavg1_2d_aer_Nx.YYYYMMDD.nc4`.
- **Authentication required**: the bucket is protected, so anonymous S3
  reads are rejected. Set `EARTHDATA_TOKEN` (a long-lived Earthdata Login
  bearer token — generate one in the Earthdata Login profile; **server-side
  only**: Fly.io secrets in production, never the repo or frontend) and
  the backend exchanges it for temporary S3 credentials automatically
  (cached in-process, refreshed before expiry). Alternatively export
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`
  directly (last ~1h). `MERRA2_S3_ANON=1` forces anonymous reads only for
  a genuinely public mirror.
- **Rate limited**: `/grid` is capped per client IP
  (`GRID_RATE_LIMIT_PER_MIN`, default 30 requests per 60 s) with
  `429` + `Retry-After` when exceeded — every request burns the
  deployer's Earthdata quota, so the limit guards it.
- Hourly, single-level NetCDF-4; one file per day, 24 steps at
  `00:30`–`23:30 UTC`; grid 576 lon × 361 lat.
- Servable variables (550 nm aerosol extinction AOT, dimensionless):
  `TOTEXTTAU DUEXTTAU BCEXTTAU OCEXTTAU SUEXTTAU SSEXTTAU`.
- Legacy shorthand aliases (`DUAOD`, `TOTAOD`, …) are accepted and
  canonicalized — see `agent/mappings.py`.

Build the index once, then serve:

```bash
# Needs temporary Earthdata Login credentials in the environment (see above).
# Scans s3://gesdisc-cumulus-prod-protected/MERRA2/M2T1NXAER.5.12.4/**/*.nc4,
# writes one JSON reference per granule + index.json. Resumable (skips
# existing refs); --limit N to index a cheap subset first.
python backend/kerchunk_index.py --out data/kerchunk --limit 3

# /grid needs KERCHUNK_INDEX_PATH pointing at the manifest (default
# ./data/kerchunk/index.json). Until it exists, /grid returns 501.
```

Notes and limitations:

- `MERRA2_S3_PREFIX` sets the bucket/prefix for the builder
  (default `s3://gesdisc-cumulus-prod-protected/MERRA2`, verified
  2026-09-12). Override it for a mirror or a different collection layout.
- The builder scans the full 5.12.4 collection directory; scope a first
  run with `--limit` and extend later — the slicer only serves dates
  present in the manifest (422 otherwise).
- MERRA-2 aerosol variables are chunked `(time=1, lat=91, lon=144)`
  (~52 KB per chunk for `DUEXTTAU`), so a query range-reads only the
  chunks intersecting (variable, bbox, time) — far cheaper than a
  whole-day fetch. The builder prints per-variable chunk sizes so you can
  see this before committing; if per-query reads ever prove too heavy,
  rechunk to Zarr/COG with spatial chunks (the documented upgrade path).
- MERRA-2 runs several weeks behind real time (newest granule observed
  2026-08-01 as of 2026-09-12). `/grid` windows entirely newer than the
  plausible archive edge return 422 saying so plainly; tune the
  assumption with `MERRA2_LATENCY_DAYS` (default 45).
- The time window collapses to ONE 2D field: `agg=hourly` averages the
  hourly steps; `daily` averages daily means; `monthly_mean` averages
  monthly means.
- Antimeridian-crossing bboxes (`w > e`) are served as two slices
  concatenated along lon (`[w..180] + [-180..e]`); `lons[]` wraps
  accordingly. Split the request yourself if you need monotonic lons.

## Frontend

Static single-page app (MapLibre GL JS via CDN — no build step, no npm).
Ask a question in the chat box; the app POSTs to `/ask`, then renders the
returned plan:

- **Air quality** (`level=surface`): Google Air Quality API heatmap tiles
  (`UAQI_RED_GREEN`), fetched **directly from the browser** with your own
  key. Click the ⚙ gear to store the key — it lives in `localStorage`
  only and is never sent to our backend. Get a key at Google Cloud
  Console (enable the *Air Quality API*). Without a key you get a
  friendly empty state, not a broken map.
- **Plume view** (`level=column`): the plan's `/grid` URL is fetched from
  the backend, mapped through a colormap onto an offscreen canvas
  (NaN/missing → transparent), and added as a MapLibre `image` source fit
  to the grid bounds. The colorbar legend is **data-driven**: gradient +
  min/mid/max labels from the fetched values (p2/p98 normalization so
  hot-pixel outliers don't crush the ramp), with units, variable name,
  and time range. Colormaps: yellow-orange-red `dust` ramp for `DU*`
  (dust AOD), viridis-like default otherwise.
- The mode toggle follows the plan: if a plan's level disagrees with the
  current toggle, the UI auto-switches with a one-line notice. Surface
  and column data are never rendered together.
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
2. Push the repo (or just `frontend/`) to Pages — `index.html` at the
   root of the published directory is the entry point.
3. No environment variables or build command needed on Pages; the only
   per-user secret (the Google key) is entered in the browser UI.

## Repo layout

| Path | What |
|---|---|
| `agent/` | QueryPlan schema, deterministic validator, prompts, mapping tables |
| `backend/` | FastAPI app, in-process plan cache, kerchunk index builder + `/grid` slicer |
| `frontend/` | Single-page MapLibre app (CDN), chat box, legend, time scrubber |
| `data/` | Gazetteer + event catalog + colormap samples |

## Status

Scaffold (v0.1): the full skeleton boots and the contracts are real, but
some integrations are still stubs — see the TODO list in `README` issues /
code comments marked `TODO(integration)`. Implemented so far: the LiteLLM
parse step, the kerchunk index builder (`backend/kerchunk_index.py`), the
`/grid` slicer (`backend/grid.py`, synthetic-tested; one live AWS
range-read still pending), Earthdata token auth + `/grid` rate limiting,
and the frontend grid renderer (plume canvas overlay with data-driven
colorbar, Google heatmap tiles with bring-your-own-key settings).
Still stubbed: the CAMS fetcher, time-frame prefetching, OpenAQ overlay.

## License

MIT — see `LICENSE`.
