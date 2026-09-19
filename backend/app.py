"""Aerosol Lens backend: FastAPI service.

Pipeline per question:
    POST /ask -> cache? -> Jev fast path (TypeSafe Jev classifier +
                deterministic place/time extraction) -> LLM fallback ->
                geocode place -> validate_plan() -> build data URLs +
                legend + caption -> JSON response

The agent's only job is translation: it emits a QueryPlanDraft (place
NAME, never coordinates). Deterministic code resolves the place via the
local gazetteer and validates the plan before anything executes.

Data paths:
- source="merra2": backend serves grid slices via kerchunk byte-range reads
  (GET /grid). Every plan routes to MERRA-2 column aerosol data.
- source="cams": historical surface PM2.5. TODO: pick an access path
  (CAMS ADS API) and implement the fetcher.

There is no surface source in this build: the Google Air Quality path was
removed and CAMS is not wired up, so health intents are refused with a
clean 422 rather than answered with column data.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

# Structured request logging. On Fly.io, stdout is captured by `fly logs`.
# Log level via LOG_LEVEL env (default INFO); set to DEBUG for per-chunk detail.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("aerosol-lens")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError as PydanticValidationError
from starlette.responses import JSONResponse

from agent.prompts import SYSTEM_PROMPT, build_user_message
from agent.query_plan import QueryPlan, QueryPlanDraft, StyleSpec
from agent.validator import ValidationError, validate_plan
from agent.mappings import MERRA2_COLUMN_VARIABLES, canonical_variable

from . import jev, rate_limit
from .cache import PlanCache
from .earthdata_auth import EarthdataExchangeError, EarthdataTokenMissingError
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


class GridRateLimitMiddleware:
    """Per-IP rate limiting for /grid: 429 + Retry-After when exceeded.

    Every /grid request range-reads the protected MERRA-2 bucket against
    the deployer's Earthdata credentials, so the quota needs guarding.
    Pure ASGI: short-circuits before the endpoint runs. Configured via
    GRID_RATE_LIMIT_PER_MIN (default 30 req/min/IP); see backend/rate_limit.py.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/grid"):
            limiter = rate_limit.get_limiter()
            ip = rate_limit.client_ip(scope)
            if not limiter.allow(ip):
                resp = JSONResponse(
                    {
                        "detail": (
                            f"rate limit exceeded for /grid "
                            f"({limiter.per_minute} requests per minute per "
                            f"IP). Back off and retry."
                        )
                    },
                    status_code=429,
                    headers={"Retry-After": str(limiter.retry_after(ip))},
                )
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


app.add_middleware(GridRateLimitMiddleware)

plan_cache = PlanCache()


class LLMNotConfiguredError(Exception):
    """The model backend cannot be used: missing package, model, or API key."""


class PlanRejectedError(Exception):
    """The model's output failed validation, even after one retry."""


class RateLimitedError(Exception):
    """The model provider rate-limited the request (HTTP 429).

    Surfaced immediately, never retried: a 429 will not heal inside a
    single retry loop, and burning the retry hides the real signal.
    """


def _is_rate_limit(exc: BaseException) -> bool:
    """True when exc is a provider 429 (litellm.RateLimitError or equivalent).

    Checks the real litellm class first, then falls back to the class name
    (mocks, wrapped/proxied exceptions) and to a 429 status_code, which
    litellm API errors carry.
    """
    rl = getattr(litellm, "RateLimitError", None)
    if rl is not None and isinstance(exc, rl):
        return True
    if type(exc).__name__ == "RateLimitError":
        return True
    return getattr(exc, "status_code", None) == 429


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


# --------------------------------------------------------------------------
# Admin: field-store backfill. Bearer-token auth; the endpoints do not
# exist at all (404) when ADMIN_TOKEN is unset, so a forgotten secret
# fails closed instead of leaving an open trigger.


class BackfillRequest(BaseModel):
    start: str | None = None  # YYYY-MM-DD; defaults to a year of archive
    end: str | None = None    # YYYY-MM-DD; defaults to the archive edge


def _require_admin(request: Request) -> None:
    token = os.environ.get("ADMIN_TOKEN", "").strip()
    if not token:
        # Fail closed: without a configured token there is no admin
        # surface to probe.
        raise HTTPException(status_code=404, detail="not found")
    presented = request.headers.get("authorization", "")
    if not secrets.compare_digest(presented, f"Bearer {token}"):
        raise HTTPException(status_code=403, detail="forbidden")


@app.post("/admin/backfill")
def admin_backfill(req: BackfillRequest, request: Request) -> dict:
    """Start a field-store backfill in a background thread.

    Only one backfill runs at a time; a second call while one is
    running returns 409 with the live status.
    """
    from . import backfill

    _require_admin(request)
    if backfill.is_running():
        raise HTTPException(
            status_code=409,
            detail={"message": "a backfill is already running",
                    "status": backfill.get_status()},
        )
    try:
        start, end = backfill.resolve_range(req.start, req.end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Day downloads are latency-bound (thousands of small HTTPS range
    # reads per day), so a few parallel workers give a near-linear
    # speedup; Zarr writes stay serial inside backfill_range.
    workers = int(os.environ.get("BACKFILL_WORKERS", "4"))
    thread = threading.Thread(
        target=backfill.backfill_range,
        kwargs={"start": start, "end": end, "workers": workers},
        name="fields-backfill",
        daemon=True,
    )
    thread.start()
    log.info("admin: backfill started %s..%s (%d workers)", start, end, workers)
    return {"started": True, "start": start.isoformat(), "end": end.isoformat(),
            "total_days": (end - start).days + 1}


@app.get("/admin/backfill/status")
def admin_backfill_status(request: Request) -> dict:
    """Live backfill progress: {running, total_days, done_days,
    current_date, errors[]}."""
    from . import backfill

    _require_admin(request)
    return backfill.get_status()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    reference_date = req.reference_date or datetime.now(timezone.utc).date().isoformat()
    # Scope the cache by date + location: "last week" asked on different
    # days, or "here" from different places, must not collide.
    scope = f"{reference_date}|{req.user_location or ''}"
    t_start = time.monotonic()
    log.info("ask: q=%r ref_date=%s location=%s", question, reference_date, req.user_location)

    # 1. Cache: repeated questions never touch the model.
    cached_plan = plan_cache.get(question, scope=scope)
    if cached_plan is not None:
        plan = cached_plan
        was_cached = True
        log.info("ask: cache HIT q=%r", question)
    else:
        # 2. Agent parse (LLM) -> geocode -> deterministic validation.
        #    _parse_with_agent retries once internally, then fails honestly.
        try:
            plan = _parse_with_agent(question, reference_date, req.user_location)
        except LLMNotConfiguredError as exc:
            log.warning("ask: LLM not configured q=%r", question)
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        except RateLimitedError as exc:
            log.warning("ask: rate limited q=%r", question)
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except (PlanRejectedError, UnknownPlaceError, PydanticValidationError) as exc:
            log.warning("ask: plan rejected q=%r err=%s", question, exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        plan_cache.put(question, plan, scope=scope)
        was_cached = False

    plan = _clamp_dateless_window(question, reference_date, plan)

    data_url = _data_url_for(plan)
    legend = _legend_for(plan)
    if not plan.caption:
        plan.caption = _auto_caption(plan)

    elapsed_ms = (time.monotonic() - t_start) * 1000
    log.info(
        "ask: done q=%r cached=%s plan=%s/%s/%s t=%s..%s bbox=%s data_url=%s %.0fms",
        question, was_cached, plan.source, plan.level, plan.variable,
        plan.time_start, plan.time_end, plan.bbox, data_url, elapsed_ms,
    )
    return AskResponse(plan=plan, data_url=data_url, legend=legend, cached=was_cached)


def _jev_enabled() -> bool:
    """Fast path kill switch: JEV_ENABLED=0 disables the Jev classifier."""
    return os.environ.get("JEV_ENABLED", "1").strip().lower() not in ("0", "false", "no")


def _clamp_dateless_window(question: str, reference_date: str, plan: QueryPlan) -> QueryPlan:
    """Dateless questions mean "latest available", not today.

    The MERRA-2 archive runs ~MERRA2_LATENCY_DAYS behind real time, so a
    default-to-today window would 422 at the /grid latency gate. When the
    question carries no time expression, clamp the window to the newest
    date with servable data (grid.newest_available_date: union of the
    Zarr field store and kerchunk-manifest coverage, capped at the
    plausible archive edge). Explicit dates (including "today") are never
    touched: asking for a day with no data still fails honestly.
    Idempotent, so cached plans are safe.
    """
    from .grid import newest_available_date

    if jev.extract_time(question, reference_date).explicit:
        return plan
    newest = newest_available_date()
    plan.time_start = datetime(newest.year, newest.month, newest.day, tzinfo=timezone.utc)
    plan.time_end = plan.time_start + timedelta(days=1) - timedelta(seconds=1)
    log.info("ask: dateless question; clamped window to newest available %s", newest)
    return plan


def _parse_with_jev(question: str, reference_date: str) -> QueryPlan | None:
    """Fast path: Jev classification + deterministic place/time extraction.

    Returns a validated QueryPlan, or None when the LLM path should take
    over (low confidence, no resolvable place). Raises PlanRejectedError
    for confident health intents -- there is no surface source in this
    build, so health gets an honest 422, never column data as an answer.
    Raises JevError (or anything else) on Jev failure; the caller treats
    that as "use the LLM path".
    """
    result = jev.classify_question(question)
    log.info("ask: jev classification: %s", result.summary())
    if not result.confident:
        log.info("ask: jev not confident; falling back to LLM path")
        return None
    if result.intent == "health":
        raise PlanRejectedError(_HEALTH_UNAVAILABLE_MSG)
    var = canonical_variable(result.variable)
    if var not in MERRA2_COLUMN_VARIABLES:
        log.info(
            "ask: jev variable %r not servable; falling back to LLM path",
            result.variable,
        )
        return None
    place_name = jev.extract_place(question)
    if place_name is None:
        log.info("ask: jev path found no gazetteer place; falling back to LLM path")
        return None
    window = jev.extract_time(question, reference_date)
    # Every plan routes to MERRA-2 column data: level/source are fixed, not
    # model outputs. Style is the same aod_sequential used for plume views.
    draft = QueryPlanDraft(
        intent=result.intent,
        level="column",
        source="merra2",
        variable=var,
        place=place_name,
        time_start=window.start,
        time_end=window.end,
        aggregation=result.aggregation,
        style=StyleSpec(colormap="aod_sequential", mode="continuous", opacity=0.65),
        caption=None,
    )
    canonical_name, bbox = resolve_place(place_name)
    return validate_plan(QueryPlan.from_draft(draft, bbox=bbox, place_name=canonical_name))


_HEALTH_UNAVAILABLE_MSG = (
    "health-related questions need surface air-quality data, which "
    "isn't available in this build"
)


def _parse_with_agent(
    question: str, reference_date: str, user_location: str | None
) -> QueryPlan:
    """Translate a question into a validated QueryPlan.

    Fast path first: the TypeSafe Jev classifier decides intent/variable/
    aggregation (~100-500ms) while place and time are extracted
    deterministically. Confident, fully-resolved plans return directly.
    Anything missing (low confidence, no place/time) falls back to the
    LLM path below, which is unchanged. Jev failures are logged and never
    break /ask.

    Flow: Jev classification -> deterministic extraction -> geocode ->
    validate_plan(); or: model emits a QueryPlanDraft (place NAME, never
    coordinates) -> backend resolves the place via the local gazetteer ->
    QueryPlan -> deterministic validate_plan(). On any LLM failure the
    model gets exactly ONE retry with the error message appended; after
    that we fail honestly. No silent fallback plans, ever.

    Health questions are refused with a 422 (no surface source in this
    build: Google removed, CAMS unbuilt) -- never answered with column
    data. source="google" can no longer be selected at all.
    """
    if _jev_enabled():
        try:
            plan = _parse_with_jev(question, reference_date)
        except PlanRejectedError:
            raise  # deliberate refusal (e.g. health intent): 422, no LLM fallback
        except Exception as exc:  # noqa: BLE001 -- JevError or anything unexpected
            log.warning("ask: jev fast path failed (%r); falling back to LLM path", exc)
            plan = None
        if plan is not None:
            return plan
        log.info("ask: jev fast path declined; using LLM path")
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
            log.info(
                "agent: attempt=%d model=%s draft=%s",
                _attempt, model,
                draft.model_dump_json() if draft is not None else "NEED_LOCATION",
            )
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
            if _is_rate_limit(exc):
                # 429s don't heal within one retry; surface immediately
                # as a 429 so the client can back off, instead of burning
                # the retry and misreporting it as an invalid plan.
                raise RateLimitedError(
                    f"model rate limit exceeded for {model!r}: {exc}"
                ) from exc
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
    if data.get("error") == "surface_unavailable":
        raise PlanRejectedError(_HEALTH_UNAVAILABLE_MSG)
    # Product refusals, checked on the raw dict so the model gets a clear
    # 422 instead of a schema retry: health intents have no servable source
    # in this build (Google removed, CAMS unbuilt), and source="google"
    # no longer exists in the schema at all.
    if data.get("intent") == "health":
        raise PlanRejectedError(_HEALTH_UNAVAILABLE_MSG)
    if data.get("source") == "google":
        raise PlanRejectedError("the Google Air Quality source was removed from this build")
    return QueryPlanDraft.model_validate(data)


def _data_url_for(plan: QueryPlan) -> str:
    """Return the URL template the frontend should render for this plan.

    Google no longer exists in this build: no plan can carry
    source="google" (it is not in the Source schema), and this function
    has no branch for it.
    """
    if plan.source == "merra2":
        w, s, e, n = plan.bbox
        # Build via urlencode, not an f-string: time_start/end are UTC-aware
        # datetimes, and .isoformat() renders the offset as "+00:00". An
        # un-encoded "+" in a query string decodes to a space on the server
        # (application/x-www-form-urlencoded convention), which every
        # client -- curl, fetch(), anything -- reproduces by sending the "+"
        # verbatim; the frontend fetches this URL as-is (frontend/app.js
        # renderPlume), so an unencoded query broke every real request.
        query = urlencode({
            "source": "merra2",
            "variable": plan.variable,
            "bbox": f"{w},{s},{e},{n}",
            "t0": plan.time_start.isoformat(),
            "t1": plan.time_end.isoformat(),
            "agg": plan.aggregation,
        })
        return f"/grid?{query}"
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
    """Serve an aggregated MERRA-2 grid slice for client-side rendering.

    Returns compact JSON: {variable, units, lats[], lons[], values[][],
    time_start, time_end, aggregation, source}. The time window collapses
    to a single 2D field (see backend/grid.py for the v1 semantics).
    """
    from .grid import (
        BadGridRequestError,
        GridDepsMissingError,
        GridFetchError,
        IndexNotBuiltError,
        UnknownVariableError,
        get_grid,
    )

    t_start = time.monotonic()
    log.info(
        "grid: source=%s var=%s bbox=%s t=%s..%s agg=%s",
        source, variable, bbox, t0, t1, agg,
    )
    try:
        # target_options=None -> auto: local reference targets need no auth;
        # remote (S3) targets resolve via backend.earthdata_auth --
        # EARTHDATA_TOKEN exchange, the AWS credential chain, or an honest
        # 501. MERRA2_S3_ANON=1 forces anonymous reads for public mirrors.
        result = get_grid(
            source=source,
            variable=variable,
            bbox=bbox,
            t0=t0,
            t1=t1,
            agg=agg,
        )
    except (GridDepsMissingError, IndexNotBuiltError) as exc:
        log.warning("grid: 501 %s", exc)
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except EarthdataTokenMissingError as exc:
        log.warning("grid: 501 %s", exc)
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except EarthdataExchangeError as exc:
        log.warning("grid: 502 %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (UnknownVariableError, BadGridRequestError) as exc:
        log.warning("grid: 422 %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GridFetchError as exc:
        log.warning("grid: 502 %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    elapsed_ms = (time.monotonic() - t_start) * 1000
    vals = result.get("values", [])
    nlat = len(vals)
    nlon = len(vals[0]) if nlat else 0
    log.info(
        "grid: done var=%s %dx%d t=%s..%s %.0fms",
        variable, nlon, nlat, result.get("time_start"), result.get("time_end"),
        elapsed_ms,
    )
    return result
