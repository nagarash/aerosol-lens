"""Aerosol Lens backend: FastAPI service.

Pipeline per question:
    POST /ask  ->  cache? -> agent parse (LLM, stubbed) -> validate_plan()
                -> build data URLs + legend + caption -> JSON response

Data paths:
- source="google": the FRONTEND calls the Google Air Quality API directly
  with the user's own key. The backend only returns which endpoint/tile
  template to use. We never proxy the key.
- source="merra2": backend serves grid slices via kerchunk byte-range reads
  (GET /grid). TODO: wire kerchunk_index.py output + xarray here.
- source="cams": historical surface PM2.5. TODO: pick an access path
  (CAMS ADS API) and implement the fetcher.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.prompts import SYSTEM_PROMPT, build_user_message
from agent.query_plan import QueryPlan
from agent.validator import ValidationError, validate_plan

from .cache import PlanCache

app = FastAPI(title="Aerosol Lens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

plan_cache = PlanCache()
LITELLM_MODEL = os.environ.get("LITELLM_MODEL", "gemini/gemini-2.0-flash")


class AskRequest(BaseModel):
    question: str
    reference_date: str | None = None  # ISO date; defaults to today (UTC)
    user_location: str | None = None  # "lat,lon" when the user shares it


class AskResponse(BaseModel):
    plan: QueryPlan
    data_url: str  # what the frontend renders: tile template or /grid URL
    legend: dict  # {units, stops: [[value, color], ...], guideline?}
    cached: bool


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "model": LITELLM_MODEL, "cache": plan_cache.stats()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    reference_date = req.reference_date or datetime.now(timezone.utc).date().isoformat()

    # 1. Cache: repeated questions never touch the model.
    cached_plan = plan_cache.get(question)
    if cached_plan is not None:
        plan = cached_plan
        was_cached = True
    else:
        # 2. Agent parse (LLM). STUB: replace with litellm.completion().
        try:
            plan = _parse_with_agent(question, reference_date, req.user_location)
        except NotImplementedError as exc:
            # The LLM integration is a stub in this scaffold.
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        # 3. Deterministic validation with hard guardrails.
        try:
            plan = validate_plan(plan)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        plan_cache.put(question, plan)
        was_cached = False

    data_url = _data_url_for(plan)
    legend = _legend_for(plan)
    if not plan.caption:
        plan.caption = _auto_caption(plan)

    return AskResponse(plan=plan, data_url=data_url, legend=legend, cached=was_cached)


def _parse_with_agent(question: str, reference_date: str, user_location: str | None) -> QueryPlan:
    """Translate a question into a QueryPlan via the LLM.

    TODO(integration): replace this stub with:
        import litellm, json
        resp = litellm.completion(
            model=LITELLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(question, reference_date, user_location)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        plan = QueryPlan.model_validate(json.loads(resp.choices[0].message.content))
        # On ValidationError: retry once with the error appended, then fail.
    """
    _ = (SYSTEM_PROMPT, build_user_message)  # keep imports referenced until wired
    raise NotImplementedError(
        "TODO(integration): wire litellm.completion() here (see docstring). "
        "Set LITELLM_MODEL in .env."
    )


def _data_url_for(plan: QueryPlan) -> str:
    """Return the URL template the frontend should render for this plan."""
    if plan.source == "google":
        # The frontend calls Google directly with the user's key; we only
        # tell it which tile template to use. Never proxy the API key.
        # e.g. https://airquality.googleapis.com/v1/mapTypes/UAQI_RED_GREEN/heatmapTiles/{z}/{x}/{y}?key=USER_KEY
        return "google://heatmapTiles/{z}/{x}/{y}"
    if plan.source == "merra2":
        w, s, e, n = plan.bbox
        return (
            "/grid?source=merra2"
            f"&variable={plan.variable}"
            f"&bbox={w},{s},{e},{n}"
            f"&t0={plan.time_start.isoformat()}&t1={plan.time_end.isoformat()}"
            f"&agg={plan.aggregation}"
        )
    if plan.source == "cams":
        # TODO(integration): implement the CAMS fetcher, then return its URL.
        return "cams://pending"
    raise HTTPException(status_code=500, detail=f"unknown source {plan.source!r}")


def _legend_for(plan: QueryPlan) -> dict:
    """Legend spec: units + color stops (+ guideline for exceedance mode)."""
    # TODO(integration): load real stops from data/colormaps.json keyed by
    # plan.style.colormap, and attach WHO guideline thresholds for
    # mode="exceedance".
    if plan.level == "surface":
        return {"units": "µg/m³", "stops": "aqi", "guideline": "WHO 24h"}
    return {"units": "AOD (unitless)", "stops": "aod_sequential"}


def _auto_caption(plan: QueryPlan) -> str:
    t0 = plan.time_start.date().isoformat()
    t1 = plan.time_end.date().isoformat()
    place = plan.place_name or "requested area"
    return f"{plan.variable} over {place}, {t0}..{t1} ({plan.source})"


@app.get("/grid")
def grid(
    source: str,
    variable: str,
    bbox: str,
    t0: str,
    t1: str,
    agg: str = "daily",
) -> dict:
    """Serve a data grid slice for client-side rendering.

    TODO(integration): implement with kerchunk + xarray + fsspec:
    open the reference index built by kerchunk_index.py, select
    (variable, bbox, time), aggregate, and return a compact payload
    (downsampled to ~screen resolution, e.g. quantized float16 or PNG).
    """
    _ = (source, variable, bbox, t0, t1, agg)
    raise HTTPException(
        status_code=501,
        detail="TODO(integration): implement kerchunk/xarray grid slicing here.",
    )
