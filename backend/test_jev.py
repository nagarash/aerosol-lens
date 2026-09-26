"""Tests for the Jev fast path: classify -> deterministic extract -> plan.

The OpenRouter HTTP call is fully mocked: no API key is needed and no
network is touched. The API key value "test-key" is a fixture, never a
real credential.

Run from the repo root:
    /tmp/ae-venv/bin/python backend/test_jev.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.app as app_module
import backend.jev as jev_module
from backend.app import AskRequest, ask
from backend.cache import PlanCache

REF = "2026-09-11"
SAHARA_BBOX = [-17.0, 15.0, 35.0, 32.0]
DELHI_BBOX = [76.8, 28.4, 77.4, 28.9]

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


def jev_answers(
    intent="plume",
    i_conf=0.95,
    variable="DUEXTTAU",
    v_conf=0.94,
    agg="daily",
    a_conf=0.9,
):
    """One scripted System One response with the three parallel questions."""
    return {
        "model": JEV_MODEL,
        "answers": {
            "intent": {
                "type": "choice",
                "choice": intent,
                "probabilities": {intent: i_conf},
                "confidence": i_conf,
            },
            "variable": {
                "type": "choice",
                "choice": variable,
                "probabilities": {variable: v_conf},
                "confidence": v_conf,
            },
            "aggregation": {
                "type": "choice",
                "choice": agg,
                "probabilities": {agg: a_conf},
                "confidence": a_conf,
            },
        },
    }


def plume_draft():
    return json.dumps(
        {
            "intent": "plume",
            "level": "column",
            "source": "merra2",
            "variable": "DUEXTTAU",
            "place": "Sahara",
            "time_start": "2026-08-31T00:00:00Z",
            "time_end": "2026-09-06T23:59:59Z",
            "aggregation": "daily",
            "style": {"colormap": "aod_sequential", "mode": "continuous", "opacity": 0.65},
            "place_name": None,
            "caption": "Dust AOD over Sahara, 2026-08-31..2026-09-06 (merra2)",
            "error": None,
        }
    )


class FakeLiteLLM:
    """Scripted litellm stand-in; records calls, fails loudly if unscripted."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("LLM was called but no script remained")
        payload = self._script.pop(0)

        class Msg:
            content = payload

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()


class harness:
    """Env + mocked Jev HTTP + fake litellm + fresh plan cache."""

    def __init__(self, jev_script, llm_script=None, jev_enabled="1", model="mock/mock-model"):
        self.jev_script = list(jev_script)
        self.jev_calls = []
        self.llm = FakeLiteLLM(llm_script or [])
        self.model = model
        self.jev_enabled = jev_enabled

    def __enter__(self):
        self._saved_env = {}
        for k, v in {
            "JEV_ENABLED": self.jev_enabled,
            "OPENROUTER_API_KEY": "test-key",
            "LITELLM_MODEL": self.model,
            "JEV_MODEL": JEV_MODEL,
            "JEV_MIN_CONFIDENCE": "0.7",
            # These tests cover the legacy /ask planner; keep the hourly
            # default from rerouting them.
            "HOURLY_PLUMES_ENABLED": "0",
        }.items():
            self._saved_env[k] = os.environ.get(k)
            os.environ[k] = v
        self._saved_post = jev_module._post_json
        script, calls = self.jev_script, self.jev_calls

        def fake_post(url, payload, headers, timeout):
            calls.append(
                {
                    "url": url,
                    "payload": payload,
                    "headers": headers,
                    "timeout": timeout,
                }
            )
            item = script.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        jev_module._post_json = fake_post
        self._saved_litellm = app_module.litellm
        self._saved_cache = app_module.plan_cache
        app_module.litellm = self.llm
        app_module.plan_cache = PlanCache()
        return self

    def __exit__(self, *exc):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        jev_module._post_json = self._saved_post
        app_module.litellm = self._saved_litellm
        app_module.plan_cache = self._saved_cache
        return False


def expect_http(status, fn):
    from fastapi import HTTPException

    try:
        fn()
    except HTTPException as exc:
        assert exc.status_code == status, (
            f"expected HTTP {status}, got {exc.status_code}: {exc.detail}"
        )
        return exc
    raise AssertionError(f"expected HTTP {status}, no exception raised")


# --------------------------------------------------------------------------
# classify_question
# --------------------------------------------------------------------------


def test_classify_confident():
    with harness([jev_answers()]):
        r = jev_module.classify_question("show Saharan dust")
    assert r.intent == "plume" and r.intent_confidence == 0.95
    assert r.variable == "DUEXTTAU" and r.variable_confidence == 0.94
    assert r.aggregation == "daily" and r.aggregation_confidence == 0.9
    assert r.confident is True


def test_classify_request_shape():
    with harness([jev_answers()]) as h:
        jev_module.classify_question("q")
    call = h.jev_calls[0]
    assert call["url"] == JEV_URL, f"must use OpenRouter's Jev route: {call['url']}"
    assert call["payload"]["model"] == JEV_MODEL
    assert set(call["payload"]["questions"]) == {"intent", "variable", "aggregation"}, (
        "exactly three parallel questions, no is_current noul"
    )
    assert call["payload"]["state"] == "q"
    assert call["timeout"] == 8.0
    assert call["headers"]["Authorization"] == "Bearer test-key"


def test_classify_low_confidence_flag():
    with harness([jev_answers(i_conf=0.5)]):
        r = jev_module.classify_question("q")
    assert r.confident is False, "one weak answer must fail the gate"


def test_classify_no_key_raises_jev_error():
    with harness([]):
        os.environ.pop("OPENROUTER_API_KEY")
        try:
            jev_module.classify_question("q")
        except jev_module.JevError:
            return
    raise AssertionError("expected JevError without OPENROUTER_API_KEY")


def test_classify_transport_error_becomes_jev_error():
    real = urllib.request.urlopen

    def boom(*a, **k):
        raise urllib.error.URLError("connection refused")

    urllib.request.urlopen = boom
    try:
        jev_module._post_json("https://x", {}, {}, 1)
    except jev_module.JevError:
        return
    finally:
        urllib.request.urlopen = real
    raise AssertionError("expected JevError for transport failure")


def test_classify_http_error_becomes_jev_error():
    with harness([jev_module.JevError("Jev endpoint HTTP 401: ...")]):
        try:
            jev_module.classify_question("q")
        except jev_module.JevError:
            return
    raise AssertionError("expected JevError for HTTP error")


def test_classify_malformed_response_becomes_jev_error():
    for bad in ({"nope": 1}, {"answers": {"intent": {"type": "noul"}}}):
        with harness([bad]):
            try:
                jev_module.classify_question("q")
            except jev_module.JevError:
                continue
            raise AssertionError(f"expected JevError for {bad}")


# --------------------------------------------------------------------------
# extract_place
# --------------------------------------------------------------------------


def test_extract_place_single_match():
    with harness([]):
        assert jev_module.extract_place("show Saharan dust") == "Sahara"


def test_extract_place_longest_match_wins():
    with harness([]):
        assert jev_module.extract_place("dust storm over the Sahara desert") == "Sahara"


def test_extract_place_alias():
    with harness([]):
        assert jev_module.extract_place("dust over New Delhi last week") == "Delhi"


def test_extract_place_transport_unions_bboxes():
    # "Saharan dust plume over Atlantic": the viewport is the plume's
    # path, so both regions are kept and geocode unions their bboxes.
    with harness([]):
        assert jev_module.extract_place(
            "show me saharan dust plume over Atlantic in July 2026"
        ) == "Sahara / Tropical Atlantic"


def test_extract_place_no_match_returns_none():
    # A deliberately unreal name -- "Atlantis" used to serve this purpose
    # but GeoNames has an actual town called Atlantis (South Africa, pop
    # >15k), which now correctly resolves via the extended gazetteer.
    with harness([]):
        assert jev_module.extract_place("dust over Notarealplace9247") is None


def test_resolve_place_unions_transport_viewport():
    from backend.geocode import resolve_place

    name, bbox = resolve_place("Sahara / Tropical Atlantic")
    assert name == "Sahara / Tropical Atlantic"
    assert bbox == [-60.0, 5.0, 35.0, 32.0]


def test_fast_path_transport_question_month_window():
    # The user's query end to end: July 2026 -> whole month, daily,
    # viewport spanning the Sahara and the Atlantic dust corridor.
    with harness([jev_answers()]):
        resp = ask(
            AskRequest(
                question="show me saharan dust plume over Atlantic in July 2026",
                reference_date=REF,
            )
        )
    plan = resp.plan
    assert plan.variable == "DUEXTTAU"
    assert plan.aggregation == "daily"
    assert plan.time_start.date().isoformat() == "2026-07-01"
    assert plan.time_end.date().isoformat() == "2026-07-31"
    assert plan.bbox == [-60.0, 5.0, 35.0, 32.0]
    assert "t0=2026-07-01" in resp.data_url


def test_extract_place_here_reference_returns_none():
    # 'here' needs user-location handling; the LLM path owns need_location.
    with harness([]):
        assert jev_module.extract_place("is the air ok here today") is None


def _disambiguation_payload(choice, conf):
    return {
        "model": JEV_MODEL,
        "answers": {
            "place": {
                "type": "choice",
                "choice": choice,
                "probabilities": {"Delhi": 0.2, "Beijing": 0.8},
                "confidence": conf,
            }
        },
    }


def test_extract_place_disambiguates_with_jev_choice():
    with harness([_disambiguation_payload("Beijing", 0.9)]) as h:
        got = jev_module.extract_place("is Delhi or Beijing dustier today")
    assert got == "Beijing"
    call = h.jev_calls[0]
    assert set(call["payload"]["questions"]) == {"place"}
    assert set(call["payload"]["questions"]["place"]["criteria"]) == {
        "Delhi",
        "Beijing",
    }


def test_extract_place_low_confidence_disambiguation_returns_none():
    with harness([_disambiguation_payload("Beijing", 0.5)]):
        assert jev_module.extract_place("is Delhi or Beijing dustier today") is None


def test_extract_place_unknown_choice_returns_none():
    payload = _disambiguation_payload("Paris", 0.95)
    with harness([payload]):
        assert jev_module.extract_place("is Delhi or Beijing dustier today") is None


# --------------------------------------------------------------------------
# extract_time
# --------------------------------------------------------------------------


def test_extract_time_rules():
    cases = [
        # (question, expected start date, expected end date, explicit?)
        ("show dust on 2026-07-29", "2026-07-29", "2026-07-29", True),
        ("show Saharan dust on July 29 2026", "2026-07-29", "2026-07-29", True),
        ("dust on 29 July 2026", "2026-07-29", "2026-07-29", True),
        ("dust today", REF, REF, True),
        ("dust yesterday", "2026-09-10", "2026-09-10", True),
        ("dust over the last 3 days", "2026-09-08", REF, True),
        ("dust the past 2 weeks", "2026-08-28", REF, True),
        ("dust last week", "2026-09-04", REF, True),
        ("dust over the last 2 months", "2026-07-13", REF, True),
        ("dust last month", "2026-08-12", REF, True),
        ("dust over the past year", "2025-09-11", REF, True),
        ("dust in July 2026", "2026-07-01", "2026-07-31", True),
        ("smoke over the Amazon in February 2024", "2024-02-01", "2024-02-29", True),
        ("show Saharan dust", REF, REF, False),  # no time expression -> today
    ]
    for question, start, end, explicit in cases:
        w = jev_module.extract_time(question, REF)
        got = (w.start.date().isoformat(), w.end.date().isoformat(), w.explicit)
        assert got == (start, end, explicit), f"{question!r}: got {got}"
        assert w.start.tzinfo is not None and w.end.tzinfo is not None


# --------------------------------------------------------------------------
# /ask integration
# --------------------------------------------------------------------------


def test_fast_path_plume_builds_plan_without_llm():
    with harness([jev_answers()]) as h:
        resp = ask(
            AskRequest(
                question="show Saharan dust on July 29 2026",
                reference_date=REF,
            )
        )
    assert resp.plan.intent == "plume"
    assert resp.plan.level == "column"
    assert resp.plan.source == "merra2"
    assert resp.plan.variable == "DUEXTTAU"
    assert resp.plan.aggregation == "daily"
    assert resp.plan.bbox == SAHARA_BBOX
    assert resp.plan.place_name == "Sahara"
    assert resp.plan.time_start.date().isoformat() == "2026-07-29"
    assert resp.data_url.startswith("/grid?source=merra2")
    assert "google" not in resp.data_url
    assert resp.cached is False
    assert len(h.llm.calls) == 0, "confident fast path must not call the LLM"


def test_fast_path_health_returns_422_without_llm():
    """Confident health classification -> honest 422, never column data,
    and no wasted LLM fallback call."""
    with harness([jev_answers(intent="health", variable="DUEXTTAU")]) as h:
        exc = expect_http(
            422,
            lambda: ask(
                AskRequest(
                    question="is it safe to run in Delhi today",
                    reference_date=REF,
                )
            ),
        )
    assert "surface" in exc.detail, f"422 must explain the gap: {exc.detail}"
    assert len(h.llm.calls) == 0, "confident health must 422 directly, not fall back"


def test_fast_path_low_confidence_falls_back_to_llm():
    with harness([jev_answers(i_conf=0.5)], llm_script=[plume_draft()]) as h:
        resp = ask(
            AskRequest(
                question="show Saharan dust on July 29 2026",
                reference_date=REF,
            )
        )
    assert resp.plan.intent == "plume"
    assert len(h.llm.calls) == 1, "low confidence must fall back to the LLM path"


def test_fast_path_jev_error_falls_back_to_llm():
    with harness(
        [jev_module.JevError("Jev request failed: timed out")],
        llm_script=[plume_draft()],
    ) as h:
        resp = ask(
            AskRequest(
                question="show Saharan dust on July 29 2026",
                reference_date=REF,
            )
        )
    assert resp.plan.intent == "plume"
    assert len(h.llm.calls) == 1, "JevError must fall back to the LLM path"


def test_fast_path_unexpected_exception_never_500s():
    real = jev_module.classify_question

    def boom(q):
        raise RuntimeError("weird internal failure")

    jev_module.classify_question = boom
    try:
        with harness([], llm_script=[plume_draft()]) as h:
            resp = ask(
                AskRequest(
                    question="show Saharan dust on July 29 2026",
                    reference_date=REF,
                )
            )
    finally:
        jev_module.classify_question = real
    assert resp.plan.intent == "plume"
    assert len(h.llm.calls) == 1, "unexpected Jev failure must fall back, not 500"


def test_jev_disabled_skips_fast_path():
    with harness([], llm_script=[plume_draft()], jev_enabled="0") as h:
        resp = ask(
            AskRequest(
                question="show Saharan dust on July 29 2026",
                reference_date=REF,
            )
        )
    assert len(h.jev_calls) == 0, "JEV_ENABLED=0 must skip Jev entirely"
    assert len(h.llm.calls) == 1
    assert resp.plan.intent == "plume"


def test_jev_variable_choice_excludes_pm25():
    """The Jev question set is the MERRA-2 speciated set only; PM25 must
    not appear as a variable option."""
    with harness([jev_answers()]) as h:
        jev_module.classify_question("q")
    criteria = h.jev_calls[0]["payload"]["questions"]["variable"]["criteria"]
    assert set(criteria) == {
        "TOTEXTTAU",
        "DUEXTTAU",
        "BCEXTTAU",
        "OCEXTTAU",
        "SUEXTTAU",
        "SSEXTTAU",
    }
    assert "PM25" not in criteria


def test_dateless_question_clamps_to_newest_available():
    """A dateless question ("saharan dust plume") must not default to today
    (no MERRA-2 data yet) -- it clamps to the newest plausible granule."""
    from backend.grid import newest_plausible_date

    with harness([jev_answers()]) as h:
        resp = ask(
            AskRequest(
                question="saharan dust plume",
                reference_date=REF,
            )
        )
    expected = newest_plausible_date().isoformat()
    assert resp.plan.time_start.date().isoformat() == expected, resp.plan.time_start
    assert resp.plan.time_end.date().isoformat() == expected
    assert "t0=" in resp.data_url
    assert len(h.llm.calls) == 0, "fast path must not call the LLM"


def test_explicit_date_is_never_clamped():
    """An explicit date stays exactly as asked, even if it is today (a day
    with no data yet -- that 422s honestly at /grid, not silently)."""
    with harness([jev_answers()]) as h:
        resp = ask(
            AskRequest(
                question="show Saharan dust today",
                reference_date=REF,
            )
        )
    assert resp.plan.time_start.date().isoformat() == REF
    assert resp.plan.time_end.date().isoformat() == REF


def test_clamp_is_idempotent_for_cached_plans():
    """Second identical ask hits the plan cache; the clamped window survives
    the round trip unchanged."""
    from backend.grid import newest_plausible_date

    with harness([jev_answers()]) as h:
        first = ask(AskRequest(question="saharan dust plume", reference_date=REF))
        second = ask(AskRequest(question="saharan dust plume", reference_date=REF))
    assert second.cached is True
    expected = newest_plausible_date().isoformat()
    assert second.plan.time_start.date().isoformat() == expected
    assert first.plan.time_start == second.plan.time_start


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

if __name__ == "__main__":
    tests = sorted(
        (name, fn)
        for name, fn in list(globals().items())
        if name.startswith("test_") and callable(fn)
    )
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- test runner reports failures
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    total = len(tests)
    print(f"\n{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)
