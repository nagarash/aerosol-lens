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

## Repo layout

| Path | What |
|---|---|
| `agent/` | QueryPlan schema, deterministic validator, prompts, mapping tables |
| `backend/` | FastAPI app, in-process plan cache, kerchunk index builder stub |
| `frontend/` | Single-page MapLibre app (CDN), chat box, legend, time scrubber |
| `data/` | Gazetteer + event catalog + colormap samples |

## Status

Scaffold (v0.1): the full skeleton boots and the contracts are real, but
the integrations are stubs — see the TODO list in `README` issues / code
comments marked `TODO(integration)`. In particular: the LiteLLM call, the
kerchunk index build, the `/grid` slicer, and the CAMS fetcher.

## License

MIT — see `LICENSE`.
