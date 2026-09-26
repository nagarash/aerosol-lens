"""Tests for the agent parse step: parse -> geocode -> validate -> cache.

The LLM is fully mocked: no API key, no network, no litellm install needed.
Run from the repo root with:
    /tmp/al-verify/bin/python backend/test_agent_parse.py
(pytest-compatible as well: every test_* function takes no arguments.)
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException

import backend.app as app_module
from backend.app import AskRequest, ask
from backend.cache import PlanCache
from agent.query_plan import QueryPlan
from agent.validator import ValidationError as AgentValidationError, validate_plan

DELHI_BBOX = [76.8, 28.4, 77.4, 28.9]
SAHARA_BBOX = [-17.0, 15.0, 35.0, 32.0]
REF = "2026-09-11"


# ---------------------------------------------------------------------------
# Fakes


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class FakeLiteLLM:
    """Scripted litellm stand-in. Script items are JSON strings or Exceptions."""

    class RateLimitError(Exception):
        """Stands in for litellm.RateLimitError (provider 429)."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


class harness:
    """Swaps in LITELLM_MODEL, a fake litellm module, and a fresh cache."""

    def __init__(self, script, model="mock/mock-model"):
        self.fake = FakeLiteLLM(script)
        self.model = model

    def __enter__(self):
        self._saved_env = os.environ.get("LITELLM_MODEL")
        os.environ["LITELLM_MODEL"] = self.model
        # These tests cover the legacy /ask planner; keep the hourly
        # default from rerouting them.
        self._saved_hourly = os.environ.get("HOURLY_PLUMES_ENABLED")
        os.environ["HOURLY_PLUMES_ENABLED"] = "0"
        self._saved_litellm = app_module.litellm
        self._saved_cache = app_module.plan_cache
        app_module.litellm = self.fake
        app_module.plan_cache = PlanCache()
        return self.fake

    def __exit__(self, *exc):
        if self._saved_env is None:
            os.environ.pop("LITELLM_MODEL", None)
        else:
            os.environ["LITELLM_MODEL"] = self._saved_env
        if self._saved_hourly is None:
            os.environ.pop("HOURLY_PLUMES_ENABLED", None)
        else:
            os.environ["HOURLY_PLUMES_ENABLED"] = self._saved_hourly
        app_module.litellm = self._saved_litellm
        app_module.plan_cache = self._saved_cache
        return False


def health_draft(place="Delhi"):
    return json.dumps(
        {
            "intent": "health",
            "level": "surface",
            "source": "google",
            "variable": "PM25",
            "place": place,
            "time_start": "2026-09-11T00:00:00Z",
            "time_end": "2026-09-11T23:59:59Z",
            "aggregation": "hourly",
            "style": {"colormap": "aqi", "mode": "exceedance", "opacity": 0.65},
            "caption": "PM2.5 over Delhi, 2026-09-11 (google)",
        }
    )


def plume_draft(place="Sahara"):
    return json.dumps(
        {
            "intent": "plume",
            "level": "column",
            "source": "merra2",
            "variable": "DUEXTTAU",
            "place": place,
            "time_start": "2026-08-31T00:00:00Z",
            "time_end": "2026-09-06T23:59:59Z",
            "aggregation": "daily",
            "style": {
                "colormap": "aod_sequential",
                "mode": "continuous",
                "opacity": 0.65,
            },
            "caption": "Dust AOD over Sahara, 2026-08-31..2026-09-06 (merra2)",
        }
    )


def bad_health_column_draft():
    """Guardrail violation: health intent answered with column AOD."""
    return json.dumps(
        {
            "intent": "health",
            "level": "column",
            "source": "merra2",
            "variable": "DUEXTTAU",
            "place": "Delhi",
            "time_start": "2026-09-11T00:00:00Z",
            "time_end": "2026-09-11T23:59:59Z",
            "aggregation": "hourly",
            "style": {"colormap": "aqi", "mode": "exceedance", "opacity": 0.65},
        }
    )


def expect_http(status, fn):
    try:
        fn()
    except HTTPException as exc:
        assert exc.status_code == status, (
            f"expected HTTP {status}, got {exc.status_code}: {exc.detail}"
        )
        return exc
    raise AssertionError(f"expected HTTPException {status}, none raised")


# ---------------------------------------------------------------------------
# Tests


def test_health_question_returns_422_surface_unavailable():
    """Health intents are refused: no surface source exists in this build
    (Google removed, CAMS unbuilt). Must never answer with column data."""
    with harness([health_draft()]) as fake:
        exc = expect_http(
            422,
            lambda: ask(
                AskRequest(
                    question="is it safe to run in Delhi today", reference_date=REF
                )
            ),
        )
    assert "surface" in exc.detail, f"422 must explain the gap: {exc.detail}"
    assert len(fake.calls) == 1, "refusal must not consume the retry"


def test_surface_unavailable_error_code_returns_422():
    """The new prompt's refusal escape hatch -> same clean 422."""
    with harness([json.dumps({"error": "surface_unavailable"})]) as fake:
        exc = expect_http(
            422,
            lambda: ask(
                AskRequest(
                    question="is it safe to run in Delhi today", reference_date=REF
                )
            ),
        )
    assert "surface" in exc.detail
    assert len(fake.calls) == 1


def test_plume_question_produces_column_plan():
    with harness([plume_draft()]) as fake:
        resp = ask(
            AskRequest(
                question="show me Saharan dust over the Atlantic last week",
                reference_date=REF,
            )
        )
    assert resp.plan.intent == "plume"
    assert resp.plan.level == "column"
    assert resp.plan.variable == "DUEXTTAU"
    assert resp.plan.source == "merra2"
    assert resp.plan.bbox == SAHARA_BBOX
    assert resp.plan.place_name == "Sahara"
    assert resp.data_url.startswith("/grid?source=merra2")
    assert "DUEXTTAU" in resp.data_url
    assert len(fake.calls) == 1
    # structured-output contract with the model
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    assert fake.calls[0]["model"] == "mock/mock-model"
    assert fake.calls[0]["temperature"] == 0


def test_cache_hit_avoids_second_model_call():
    with harness([plume_draft()]) as fake:
        r1 = ask(
            AskRequest(
                question="show me Saharan dust over the Atlantic last week",
                reference_date=REF,
            )
        )
        assert r1.cached is False
        # same question, different casing/punctuation -> same normalized key
        r2 = ask(
            AskRequest(
                question="Show me Saharan dust over the Atlantic last week?!",
                reference_date=REF,
            )
        )
        assert r2.cached is True
        assert r2.plan.bbox == SAHARA_BBOX
        assert len(fake.calls) == 1, "cache hit must not call the model again"


def test_cache_scope_separates_reference_dates():
    with harness([plume_draft(), plume_draft()]) as fake:
        ask(
            AskRequest(
                question="show me Saharan dust over the Atlantic last week",
                reference_date=REF,
            )
        )
        # same words, different day -> different plan, must NOT hit the cache
        r = ask(
            AskRequest(
                question="show me Saharan dust over the Atlantic last week",
                reference_date="2026-10-11",
            )
        )
        assert r.cached is False
        assert len(fake.calls) == 2


def bad_window_draft():
    """Deterministic validation failure: a 3-year window with daily
    aggregation exceeds the validator's 62-day daily limit."""
    d = json.loads(plume_draft())
    d["time_start"] = "2020-01-01T00:00:00Z"
    d["time_end"] = "2023-01-01T00:00:00Z"
    return json.dumps(d)


def test_invalid_output_triggers_single_retry():
    with harness([bad_window_draft(), plume_draft()]) as fake:
        resp = ask(
            AskRequest(
                question="show me Saharan dust over the Atlantic last week",
                reference_date=REF,
            )
        )
    assert resp.plan.intent == "plume"
    assert resp.plan.variable == "DUEXTTAU"
    assert len(fake.calls) == 2, "exactly one retry expected"
    retry_msg = fake.calls[1]["messages"][-1]["content"]
    assert "exceeds" in retry_msg, "retry must carry the validation error"


def test_health_plan_rejected_without_retry():
    """A health draft fails immediately with the surface-unavailable 422
    (PlanRejectedError is not retryable), without calling the model twice."""
    with harness([bad_health_column_draft()]) as fake:
        exc = expect_http(
            422,
            lambda: ask(
                AskRequest(
                    question="is it safe to run in Delhi today", reference_date=REF
                )
            ),
        )
    assert "health" in exc.detail
    assert len(fake.calls) == 1, "health refusal must not consume the retry"


def test_unknown_place_returns_422_naming_place():
    # A deliberately unreal name -- "Atlantis" used to serve this purpose
    # but GeoNames has an actual town called Atlantis (South Africa, pop
    # >15k), which now correctly resolves via the extended gazetteer.
    place = "Notarealplace9247"
    d = plume_draft(place=place)
    with harness([d, d]):  # retry can't fix an unknown place either
        exc = expect_http(
            422,
            lambda: ask(
                AskRequest(
                    question=f"show me dust over {place} last week",
                    reference_date=REF,
                )
            ),
        )
    assert place in exc.detail, f"422 must name the place: {exc.detail}"


def test_place_alias_resolves():
    with harness([plume_draft(place="New Delhi")]):
        resp = ask(
            AskRequest(
                question="show dust over New Delhi last week", reference_date=REF
            )
        )
    assert resp.plan.place_name == "Delhi"
    assert resp.plan.bbox == DELHI_BBOX


def test_need_location_returns_422():
    need_loc = json.dumps({"error": "need_location"})
    with harness([need_loc]):
        exc = expect_http(
            422,
            lambda: ask(
                AskRequest(question="is the air ok here today", reference_date=REF)
            ),
        )
    assert "location" in exc.detail


def test_guardrail_rejects_health_with_column_directly():
    plan = QueryPlan(
        intent="health",
        level="column",
        source="merra2",
        variable="DUEXTTAU",
        bbox=SAHARA_BBOX,
        time_start="2026-09-11T00:00:00Z",
        time_end="2026-09-11T23:59:59Z",
        aggregation="daily",
    )
    try:
        validate_plan(plan)
    except AgentValidationError:
        return
    raise AssertionError("guardrail did not reject health+column")


def test_missing_model_setting_returns_422():
    # In the hourly architecture a missing model is not a 501: the router
    # asks the user for a place, aerosol, and dates instead (422).
    saved_env = os.environ.pop("LITELLM_MODEL", None)
    saved_litellm, saved_cache = app_module.litellm, app_module.plan_cache
    app_module.litellm = FakeLiteLLM([])
    app_module.plan_cache = PlanCache()
    try:
        # A vague event reference forces the model fallback path (local
        # interpretation returns None), which is where the missing
        # LITELLM_MODEL surfaces. Health questions 422 deterministically
        # before any model is needed, so they can't exercise this.
        exc = expect_http(
            422,
            lambda: ask(
                AskRequest(
                    question="show me that orange sky event", reference_date=REF
                )
            ),
        )
        assert "model fallback is not configured" in exc.detail
    finally:
        if saved_env is not None:
            os.environ["LITELLM_MODEL"] = saved_env
        app_module.litellm = saved_litellm
        app_module.plan_cache = saved_cache


def test_missing_api_key_returns_501():
    with harness([], model="gemini/gemini-2.0-flash") as fake:
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            os.environ.pop(k, None)
        exc = expect_http(
            501,
            lambda: ask(
                AskRequest(
                    question="is it safe to run in Delhi today", reference_date=REF
                )
            ),
        )
        assert "GEMINI_API_KEY" in exc.detail
        assert len(fake.calls) == 0, "no model call without a key"


def test_non_json_model_output_retried_then_422():
    with harness(["not json at all", "still not json"]):
        exc = expect_http(
            422,
            lambda: ask(
                AskRequest(question="dust over Sahara", reference_date=REF)
            ),
        )
    assert "JSON" in exc.detail


def test_rate_limit_returns_429_without_consuming_retry():
    # Uses FakeLiteLLM.RateLimitError so _is_rate_limit() matches via
    # isinstance, exactly like the real litellm module in production.
    with harness([FakeLiteLLM.RateLimitError("429 Rate limit exceeded")]) as fake:
        exc = expect_http(
            429,
            lambda: ask(
                AskRequest(
                    question="is it safe to run in Delhi today", reference_date=REF
                )
            ),
        )
    assert len(fake.calls) == 1, (
        "rate limit must surface immediately, not consume the validation retry"
    )
    assert "429" in exc.detail


# ---------------------------------------------------------------------------
# Runner (works without pytest)


def main():
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
