"""Eval harness for the LLM fallback path (backend/app.py::_parse_with_agent).

Runs a fixed set of hand-picked HARD questions -- multi-place
disambiguation, event-catalog references, figurative language without an
aerosol keyword, health refusals, places only the extended (GeoNames)
gazetteer knows -- against one or more candidate LiteLLM/OpenRouter models,
through the EXACT production code path (same SYSTEM_PROMPT, same retry
logic, same resolve_place()/validate_plan() guardrails). Jev is force-
disabled (JEV_ENABLED=0) so every question actually reaches the model being
evaluated instead of being answered by the deterministic fast path.

Results reflect what a real user would get, not a synthetic approximation
-- deliberately not a reimplementation of the parsing logic.

MERRA-2 DATA LATENCY: the archive runs ~45 days behind
(MERRA2_LATENCY_DAYS). A plan whose time window lands on "today" for a
dateless/recency question is CORRECT plan-drafting behavior --
backend/app.py's own dateless-clamp (or, for explicit "today"/"now"
wording, the /grid latency gate) is what maps that to the newest
actually-available data, not this step. This eval checks PLAN
CONSTRUCTION (place/variable/intent/error), not live data availability --
it never calls /grid.

TWO REAL GAPS THIS EVAL SURFACED WHILE WRITING IT (not model bugs):
1. data/event_catalog.sample.json is NEVER LOADED into the prompt (grep
   the repo -- agent/prompts.py only tells the model to "prefer the event
   catalog", the model never actually sees its contents). Any correct
   answer for a named-event question is the model's own training
   knowledge, not a grounded lookup. Still open. See case
   "named_event_indirect".
2. FIXED (backend/us_states_build.py): common US state / country names
   were colliding with unrelated, much smaller same-named places once the
   GeoNames extension resolved them -- "Georgia" -> the country instead
   of the US state, "Florida" -> a town in Cuba, "Arizona" -> a town in
   Honduras, "Tonga" -> a town in Cameroon (verified against
   data/gazetteer_extended.json on 2026-09-19). Root cause: no US-state/
   country admin-boundary layer had been ingested, so the right answer
   was never in the data at all. Now curated into
   data/gazetteer.sample.json from the US Census Bureau's state
   boundaries (+ Tonga from Natural Earth), which -- per
   backend/geocode.py's curated-over-extended precedence -- wins the
   collision with no other code changes. Cases "state_country_collision"
   and "relative_date_event" now assert a real place instead of just
   flagging the collision; "transport_phrasing" still can't fully pass
   (see its own known_issue) because the LLM path has no transport-union
   logic at all, a separate, still-open limitation.

Usage:
    export OPENROUTER_API_KEY=...   # or whatever your --models need
    python backend/eval_llm_fallback.py
    python backend/eval_llm_fallback.py --models openrouter/google/gemini-3.5-flash
    python backend/eval_llm_fallback.py --case multi_place_disambiguation --verbose

COSTS REAL MONEY: each (model x case) pair is 1-2 real API calls (the
production retry-once-on-invalid-output logic is preserved). Default: 3
models x 11 cases x <=2 calls = <=66 calls. Use --case to iterate cheaply
on one at a time, or --models to test just one candidate.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["JEV_ENABLED"] = "0"  # force every question through the LLM path

from backend.app import (  # noqa: E402
    _HEALTH_UNAVAILABLE_MSG,
    LLMNotConfiguredError,
    _parse_with_agent,
)

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore

from agent.query_plan import QueryPlan  # noqa: E402

# Approximate USD per million tokens (input, output), pulled from
# OpenRouter's /api/v1/models on 2026-09-19. Add an entry for any model you
# pass via --models to get a cost estimate; unknown models just show token
# counts.
_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "google/gemini-3.1-flash-lite": (0.25, 1.50),
    "google/gemini-3.5-flash-lite": (0.30, 2.50),
    "google/gemini-3.5-flash": (1.50, 9.00),
    "openai/gpt-5-nano": (0.05, 0.40),
    "openai/gpt-5-mini": (0.25, 2.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
}

_DEFAULT_MODELS = [
    "openrouter/google/gemini-2.5-flash-lite",
    "openrouter/google/gemini-3.5-flash",
    "openrouter/openai/gpt-5-mini",
]


# --------------------------------------------------------------------------
# outcome + check machinery
# --------------------------------------------------------------------------


@dataclass
class PlanOutcome:
    plan: QueryPlan | None
    error: Exception | None
    error_kind: str | None  # "health" | "need_location" | "unresolvable_place" | "other" | None


def _classify_error(exc: Exception) -> str:
    msg = str(exc)
    if _HEALTH_UNAVAILABLE_MSG in msg:
        return "health"
    if "no user location was provided" in msg:
        return "need_location"
    if "unknown place" in msg.lower():
        return "unresolvable_place"
    return "other"


CheckFn = Callable[[PlanOutcome], tuple[str, str]]  # -> (verdict, detail)


def _plan_summary(plan: QueryPlan) -> str:
    return (
        f"place={plan.place_name!r} bbox={plan.bbox} variable={plan.variable} "
        f"intent={plan.intent} window={plan.time_start.date()}..{plan.time_end.date()}"
    )


def expect_plan(
    place_any_of: tuple[str, ...] | None = None,
    variable_any_of: tuple[str, ...] | None = None,
    intent_any_of: tuple[str, ...] | None = None,
) -> CheckFn:
    def check(o: PlanOutcome) -> tuple[str, str]:
        if o.error is not None:
            return "FAIL", f"expected a plan, got error ({o.error_kind}): {o.error}"
        p = o.plan
        assert p is not None
        problems = []
        if place_any_of and p.place_name not in place_any_of:
            problems.append(f"place {p.place_name!r} not in {place_any_of}")
        if variable_any_of and p.variable not in variable_any_of:
            problems.append(f"variable {p.variable!r} not in {variable_any_of}")
        if intent_any_of and p.intent not in intent_any_of:
            problems.append(f"intent {p.intent!r} not in {intent_any_of}")
        if problems:
            return "FAIL", "; ".join(problems) + f" ({_plan_summary(p)})"
        return "PASS", _plan_summary(p)

    return check


def expect_error(kind_any_of: tuple[str, ...]) -> CheckFn:
    def check(o: PlanOutcome) -> tuple[str, str]:
        if o.error is None:
            assert o.plan is not None
            return "FAIL", f"expected a {kind_any_of} refusal, got a plan: {_plan_summary(o.plan)}"
        if o.error_kind not in kind_any_of:
            return "FAIL", f"expected {kind_any_of}, got {o.error_kind}: {o.error}"
        return "PASS", f"{o.error_kind}: {o.error}"

    return check


def info_only(note: str) -> CheckFn:
    def check(o: PlanOutcome) -> tuple[str, str]:
        if o.error is not None:
            return "INFO", f"error ({o.error_kind}): {o.error} -- {note}"
        return "INFO", f"{_plan_summary(o.plan)} -- {note}"

    return check


# --------------------------------------------------------------------------
# the 11 cases
# --------------------------------------------------------------------------


@dataclass
class Case:
    id: str
    question: str
    note: str
    check: CheckFn
    known_issue: str | None = None


CASES: list[Case] = [
    Case(
        "multi_place_disambiguation",
        "is Delhi or Beijing dustier today",
        "two named places, no explicit time-period comparison -- must pick one primary subject",
        expect_plan(place_any_of=("Delhi", "Beijing"), variable_any_of=("DUEXTTAU",), intent_any_of=("plume", "comparison")),
    ),
    Case(
        "state_country_collision",
        "dust over Georgia this week",
        "US state (near Atlanta) vs. the country of Georgia -- fixed by backend/us_states_build.py (curated wins over the extended layer's country entry)",
        expect_plan(place_any_of=("Georgia",), variable_any_of=("DUEXTTAU",)),
    ),
    Case(
        "transport_phrasing",
        "show the Saharan dust plume moving toward Florida over the next few days",
        "transport phrasing across two regions -- the LLM path (unlike jev.py's transport union) can only emit ONE place, so either endpoint is a defensible single answer",
        expect_plan(place_any_of=("Sahara", "Florida"), variable_any_of=("DUEXTTAU",)),
        known_issue="this path has no transport-union logic at all (unlike jev.py), so a single-place answer always loses half the question regardless of which place is picked",
    ),
    Case(
        "named_event_indirect",
        "that huge dust cloud everyone called Godzilla back in 2020",
        "named event via nickname, not the catalog's exact alias string",
        expect_plan(variable_any_of=("DUEXTTAU",), intent_any_of=("plume",)),
        known_issue="data/event_catalog.sample.json (ground truth: 2020-06-14..2020-06-28) is never loaded into the prompt -- a correct window here is the model's own knowledge, not a grounded lookup (gap #1)",
    ),
    Case(
        "figurative_no_keyword",
        "why was the sky orange over LA that one September",
        "figurative description, no aerosol keyword, implicit date -- tests variable inference beyond the keyword table",
        expect_plan(place_any_of=("Los Angeles",), variable_any_of=("BCEXTTAU", "OCEXTTAU", "TOTEXTTAU")),
    ),
    Case(
        "relative_date_event",
        "the volcanic eruption near Tonga last year",
        "relative date ('last year') + named event, needs reference-date math -- place collision fixed by backend/us_states_build.py's extra-countries list",
        expect_plan(place_any_of=("Tonga",), variable_any_of=("SUEXTTAU",)),
    ),
    Case(
        "health_refusal",
        "is it safe to go for a run in Beijing right now",
        "must 422, never quietly render column AOD as a safety answer",
        expect_error(("health",)),
    ),
    Case(
        "compound_comparison",
        "compare wildfire smoke in LA to dust storms in Phoenix last month",
        "two places, two variables -- the QueryPlanDraft schema has exactly one 'place' and one 'variable' field",
        info_only("schema can't represent this compound query; reporting what the model picks instead of asserting pass/fail"),
        known_issue="single place/variable fields can't losslessly represent a two-place, two-variable comparison",
    ),
    Case(
        "no_resolvable_place",
        "show me global dust levels right now",
        "no named place at all -- there is no whole-globe view in this app",
        info_only("no strict expectation; watch for a hallucinated 'representative' place standing in for 'global'"),
    ),
    Case(
        "here_reference_health",
        "is the air near me safe to breathe today",
        "'here'-reference AND health in one question, no user_location supplied -- either refusal is defensible",
        expect_error(("health", "need_location")),
    ),
    Case(
        "extended_gazetteer_reaches_llm",
        "dust storm over the Atacama Desert last week",
        "a place that exists ONLY in the extended (GeoNames) gazetteer, never in known_places_hint() shown to the model -- tests that resolve_place() reaches it from the LLM path too, not just Jev's",
        expect_plan(place_any_of=("Atacama Desert",), variable_any_of=("DUEXTTAU",)),
    ),
]


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

_usage_log: list[dict] = []


def _install_usage_spy() -> None:
    if litellm is None:
        return
    original = litellm.completion

    def wrapped(*args, **kwargs):
        resp = original(*args, **kwargs)
        usage = getattr(resp, "usage", None)
        _usage_log.append(
            {
                "model": kwargs.get("model"),
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            }
        )
        return resp

    litellm.completion = wrapped


def _cost_estimate(model: str, entries: list[dict]) -> str:
    key = model.split("openrouter/", 1)[-1]
    pricing = _PRICING_PER_M.get(key)
    prompt_tok = sum(e["prompt_tokens"] for e in entries)
    completion_tok = sum(e["completion_tokens"] for e in entries)
    if pricing is None:
        return f"{prompt_tok} prompt + {completion_tok} completion tokens (no pricing on file for {key!r})"
    in_price, out_price = pricing
    cost = prompt_tok / 1e6 * in_price + completion_tok / 1e6 * out_price
    return f"{prompt_tok} prompt + {completion_tok} completion tokens, ~${cost:.4f}"


def run_case(model: str, case: Case, reference_date: str) -> tuple[PlanOutcome, float]:
    os.environ["LITELLM_MODEL"] = model
    t0 = time.time()
    try:
        plan = _parse_with_agent(case.question, reference_date, user_location=None)
        outcome = PlanOutcome(plan=plan, error=None, error_kind=None)
    except LLMNotConfiguredError:
        raise
    except Exception as exc:  # noqa: BLE001 -- classify and report, don't crash the run
        outcome = PlanOutcome(plan=None, error=exc, error_kind=_classify_error(exc))
    return outcome, time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=str, default=",".join(_DEFAULT_MODELS), help="comma-separated LiteLLM model strings")
    parser.add_argument("--reference-date", type=str, default=date.today().isoformat())
    parser.add_argument("--case", type=str, default=None, help="run only this case id")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = [c for c in CASES if c.id == args.case] if args.case else CASES

    max_calls = len(models) * len(cases) * 2
    print(f"models: {models}")
    print(f"cases: {len(cases)}  reference_date: {args.reference_date}")
    print(f"up to {max_calls} real API calls (2 per case x cases x models, retry-on-invalid included)\n")

    _install_usage_spy()

    for model in models:
        print(f"=== {model} ===")
        start_idx = len(_usage_log)
        counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "INFO": 0}
        try:
            for case in cases:
                outcome, elapsed = run_case(model, case, args.reference_date)
                verdict, detail = case.check(outcome)
                counts[verdict] = counts.get(verdict, 0) + 1
                flag = f" [KNOWN ISSUE: {case.known_issue}]" if case.known_issue else ""
                print(f"  {verdict:5s} {case.id:32s} {elapsed:5.2f}s  {detail}{flag}")
                if args.verbose:
                    print(f"        question: {case.question!r}")
                    print(f"        note: {case.note}")
        except LLMNotConfiguredError as exc:
            print(f"  ABORTED: {exc}")
            continue
        model_entries = _usage_log[start_idx:]
        print(
            f"  -- {counts.get('PASS', 0)} PASS / {counts.get('FAIL', 0)} FAIL / "
            f"{counts.get('WARN', 0)} WARN / {counts.get('INFO', 0)} INFO  |  "
            f"{_cost_estimate(model, model_entries)}\n"
        )


if __name__ == "__main__":
    main()
