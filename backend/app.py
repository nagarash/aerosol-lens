"""Aerosol Lens backend: FastAPI service.

Pipeline per question:
    POST /ask  ->  cache? -> agent parse (LiteLLM) -> geocode place
                -> validate_plan() -> build data URLs + legend + caption
                -> JSON response

The agent's only job is translation: it emits a QueryPlanDraft (place
NAME, never coordinates). Deterministic code resolves the place via the
local gazetteer and validates the plan before anything executes.

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

import json
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError as PydanticValidationError

from agent.prompts import SYSTEM_PROMPT, build_user_message
from agent.query_plan import QueryPlan, QueryPlanDraft
from agent.validator import ValidationError, validate_plan

from .cache import PlanCache
from .geocode import UnknownPlaceError, known_places_hint, resolve_place

try:
    import litellm  # type: ignore
except ImportError:  # litellm is only needed once a model is configured
    litellm = None  # type: ignore

app = FastAPI(title="Aerosol Lens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

plan_cache = PlanCache()


class LLMNotConfiguredError(Exception):
    """The model backend cannot be used: missing package, model, or API key."""


class PlanRejectedError(Exception):
    """The model's output failed validation, even after one retry."""


class AskRequest(BaseModel):
    question: str
    reference_date: str | None = None  # ISO date; defaults to today (UTC)
    user_location: str | None = None  # "lat,lon" when the user shares it


class AskResponse(BaseModel):
    plan: QueryPlan
    data_url: str  # what the frontend renders: tile template or /grid URL
    legend: dict  # {units, stops: [[value, color], ...], guideline?}
    cached: bool


# Provider prefix (from the LiteLLM model string) -> env vars holding keys.
# Empty tuple = no key needed (local models). Absent = unknown provider,
# skip the check (LiteLLM will report auth problems itself).
_PROVIDER_KEY_ENV: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "azure": ("AZURE_API_KEY",),
    "bedrock": ("AWS_ACCESS_KEY_ID",),
    "mistral": ("MISTRAL_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "cohere": ("COHERE_API_KEY",),
    "together_ai": ("TOGETHERAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "ollama": (),
    "mock": (),
}


def _provider_of(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[0].lower()
    low = model.lower()
    if low.startswith(("gpt-", "o1", "o3", "chatgpt")):
        return "openai"
    if low.startswith("claude"):
        return "anthropic"
    if low.startswith("gemini"):
        return "gemini"
    return ""


def _require_llm() -> str:
    """Return the configured LiteLLM model string, or raise LLMNotConfiguredError.

    Read at call time (not import time) so tests and .env reloads work.
    There is deliberately NO default: an unconfigured model must produce
    an honest 501, never a fake plan.
    """
    if litellm is None:
        raise LLMNotConfiguredError(
            "the 'litellm' package is not installed. Install it "
            "(pip install litellm) or run the backend from the project image."
        )
    model = os.environ.get("LITELLM_MODEL", "").strip()
    if not model:
        raise LLMNotConfiguredError(
            "LITELLM_MODEL is not set. Set it to a LiteLLM model string, e.g. "
            "LITELLM_MODEL=gemini/gemini-2.0-flash (see .env.example)."
        )
    keys = _PROVIDER_KEY_ENV.get(_provider_of(model))
    if keys and not any(os.environ.get(k) for k in keys):
        need = " or ".join(keys)
        raise LLMNotConfiguredError(
            f"no API key found for model {model!r}: set {need}."
        )
    return model


@app.get("/healthz")
def healthz() -> dict:
    return {
        "ok": True,
        "model": os.environ.get("LITELLM_MODEL") or "not configured",
        "litellm_installed": litellm is not None,
        "cache": plan_cache.stats(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    reference_date = req.reference_date or datetime.now(timezone.utc).date().isoformat()
    # Scope the cache by date + location: "last week" asked on different
    # days, or "here" from different places, must not collide.
    scope = f"{reference_date}|{req.user_location or ''}"

    # 1. Cache: repeated questions never touch the model.
    cached_plan = plan_cache.get(question, scope=scope)
    if cached_plan is not None:
        plan = cached_plan
        was_cached = True
    else:
        # 2. Agent parse (LLM) -> geocode -> deterministic validation.
        #    _parse_with_agent retries once internally, then fails honestly.
        try:
            plan = _parse_with_agent(question, reference_date, req.user_location)
        except LLMNotConfiguredError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except (PlanRejectedError, UnknownPlaceError, PydanticValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        plan_cache.put(question, plan, scope=scope)
        was_cached = False

    data_url = _data_url_for(plan)
    legend = _legend_for(plan)
    if not plan.caption:
        plan.caption = _auto_caption(plan)

    return AskResponse(plan=plan, data_url=data_url, legend=legend, cached=was_cached)


def _parse_with_agent(
    question: str, reference_date: str, user_location: str | None
) -> QueryPlan:
    """Translate a question into a validated QueryPlan via the LLM.

    Flow: model emits a QueryPlanDraft (place NAME, never coordinates) ->
    backend resolves the place via the local gazetteer -> QueryPlan ->
    deterministic validate_plan(). On any failure the model gets exactly
    ONE retry with the error message appended; after that we fail honestly.
    No silent fallback plans, ever.
    """
    model = _require_llm()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + known_places_hint()},
        {
            "role": "user",
            "content": build_user_message(question, reference_date, user_location),
        },
    ]
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            draft = _draft_from_model(model, messages)
            if draft is None:  # model emitted {"error": "need_location"}
                raise PlanRejectedError(
                    "the question refers to 'here'/'my area' but no user "
                    "location was provided."
                )
            if not draft.place:
                raise ValueError(
                    "model returned neither a place nor an error; "
                    "a plan draft needs 'place' (a name, never coordinates)."
                )
            canonical_name, bbox = resolve_place(draft.place)
            plan = QueryPlan.from_draft(draft, bbox=bbox, place_name=canonical_name)
            return validate_plan(plan)
        except (LLMNotConfiguredError, PlanRejectedError):
            raise  # not retryable: config problems and explicit refusals
        except Exception as exc:  # noqa: BLE001 -- bad JSON/schema/place/plan: retry once
            last_error = exc
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous output was invalid:\n"
                        f"{exc}\n"
                        "Return corrected JSON only, no prose."
                    ),
                }
            )
    raise PlanRejectedError(
        f"the model produced an invalid plan twice; last error: {last_error}"
    )


def _draft_from_model(model: str, messages: list[dict]) -> QueryPlanDraft | None:
    """One LiteLLM call -> parsed QueryPlanDraft.

    Returns None when the model emits the {"error": "need_location"}
    escape hatch (detected before schema validation, since it carries no
    plan fields). Raises on transport errors, non-JSON output, or schema
    violations; the caller decides whether to retry.
    """
    assert litellm is not None  # guaranteed by _require_llm()
    try:
        resp = litellm.completion(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception as exc:
        auth_err = getattr(litellm, "AuthenticationError", None)
        if auth_err is not None and isinstance(exc, auth_err):
            raise LLMNotConfiguredError(
                f"authentication failed for model {model!r}: {exc}. "
                "Check the provider API key."
            ) from exc
        raise
    content = resp.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"model must return a JSON object, got: {content[:200]!r}")
    if data.get("error") == "need_location":
        return None
    return QueryPlanDraft.model_validate(data)


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
