"""TypeSafe Jev 'System One' classifier for the /ask fast path.

Replaces the ~15s LLM parse call with a ~100-500ms structured-decision
call for the classification portion of a plan (intent, variable,
aggregation). Place and time are extracted deterministically -- a
longest-match scan over the local gazetteer plus date rules -- never by
the model.

Product scope: every plan routes to MERRA-2 column aerosol data. There
is no surface source in this build (Google Air Quality removed, CAMS
unbuilt), so health intents are refused, never answered with column
data. The Jev variable set is exactly the MERRA-2 speciated aerosol
list from agent/mappings.py (no PM25).

Transport: POST {JEV_ENDPOINT} (default
https://openrouter.ai/api/alpha/decisions -- OpenRouter's dedicated Jev
route; Jev is not a chat model and chat/completions-style calls are
refused for it) with the native System One body {state, model,
questions}. Auth: OPENROUTER_API_KEY as a Bearer token. Model from
JEV_MODEL (default "typesafe/jev-1.13").

Any failure raises JevError; the /ask layer treats that as "fall back to
the LLM path". Jev must never break /ask, and the API key never appears
in logs or error messages.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

from .geocode import name_variants

log = logging.getLogger("aerosol-lens.jev")


class JevError(Exception):
    """Jev classification failed: no key, transport error, or bad response."""


_DEFAULT_MODEL = "typesafe/jev-1.13"
_DEFAULT_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
_DEFAULT_TIMEOUT = 8.0
_DEFAULT_MIN_CONFIDENCE = 0.7

# The only variables the fast path can emit: the exact canonical
# MERRA-2 speciated aerosol set (agent/mappings.py::MERRA2_COLUMN_VARIABLES).
_VARIABLE_CRITERIA = {
    "TOTEXTTAU": "Total aerosol optical depth (column): overall aerosol load, haze.",
    "DUEXTTAU": "Dust aerosol optical depth (column): dust storms, Saharan dust, sandstorms, ash.",
    "BCEXTTAU": "Black carbon aerosol optical depth (column): wildfire smoke, combustion.",
    "OCEXTTAU": "Organic carbon aerosol optical depth (column): smoke, haze.",
    "SUEXTTAU": "Sulfate aerosol optical depth (column): volcanic eruptions.",
    "SSEXTTAU": "Sea salt aerosol optical depth (column).",
}

_INTENT_CRITERIA = {
    "health": (
        "The user asks whether air is safe to breathe: running or exercising "
        "outside, allergies, sensitive groups, AQI, health effects."
    ),
    "plume": (
        "The user wants to see, track, or identify an aerosol plume or event: "
        "dust storm, Saharan dust, wildfire smoke, ash cloud, haze."
    ),
    "comparison": (
        "The user compares two time periods or events, e.g. 'vs last year' "
        "or 'compared to the 2020 fires'."
    ),
}

_AGGREGATION_CRITERIA = {
    "hourly": "The question is about right now or today: hourly detail.",
    "daily": "The question covers several days to weeks: daily values.",
    "monthly_mean": "The question covers months to years: monthly averages.",
}


@dataclass
class JevClassification:
    """One batched Jev call: three parallel classifications + confidences."""

    intent: str
    intent_confidence: float
    variable: str
    variable_confidence: float
    aggregation: str
    aggregation_confidence: float
    confident: bool
    usage: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"intent={self.intent}({self.intent_confidence:.2f}) "
            f"variable={self.variable}({self.variable_confidence:.2f}) "
            f"aggregation={self.aggregation}({self.aggregation_confidence:.2f}) "
            f"confident={self.confident}"
        )


@dataclass
class TimeWindow:
    """Deterministically extracted time window (UTC)."""

    start: datetime
    end: datetime
    explicit: bool = False  # False when defaulted (no time expression found)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    """POST JSON and parse the JSON response. Module-level for test patching."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # The body carries the provider's error; it never contains our key
        # (the key travels in the Authorization header, never the body).
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise JevError(f"Jev endpoint HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise JevError(f"Jev request failed: {exc}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise JevError(f"Jev returned non-JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise JevError(f"Jev returned a non-object: {body[:200]!r}")
    return parsed


def _system_one(state: str, questions: dict) -> dict:
    """POST one System One request; return the full response dict."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise JevError("OPENROUTER_API_KEY is not set; cannot call Jev")
    model = os.environ.get("JEV_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    endpoint = (
        os.environ.get("JEV_ENDPOINT", _DEFAULT_ENDPOINT).strip() or _DEFAULT_ENDPOINT
    )
    timeout = _env_float("JEV_TIMEOUT", _DEFAULT_TIMEOUT)
    payload = {"state": state, "model": model, "questions": questions}
    headers = {
        "Authorization": "Bearer " + key,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/nagarash/aerosol-lens",
        "X-Title": "Aerosol Lens",
    }
    return _post_json(endpoint, payload, headers, timeout)


def _answers_of(data: dict) -> dict:
    answers = data.get("answers")
    if not isinstance(answers, dict):
        raise JevError(f"Jev response missing 'answers': {str(data)[:200]}")
    return answers


def _parse_choice(answers: dict, name: str) -> tuple[str, float]:
    a = answers.get(name)
    if not isinstance(a, dict) or a.get("type") != "choice":
        raise JevError(f"Jev answer {name!r} is not a choice: {str(a)[:160]}")
    choice, conf = a.get("choice"), a.get("confidence")
    if not isinstance(choice, str) or not choice:
        raise JevError(f"Jev answer {name!r} has no choice: {str(a)[:160]}")
    if not isinstance(conf, (int, float)):
        raise JevError(f"Jev answer {name!r} has no confidence: {str(a)[:160]}")
    return choice, float(conf)


def _build_questions() -> dict:
    """The three parallel classification questions for one user question."""
    return {
        "intent": {
            "type": "choice",
            "instructions": "What is the user asking about?",
            "criteria": dict(_INTENT_CRITERIA),
        },
        "variable": {
            "type": "choice",
            "instructions": "Which aerosol variable best matches the question?",
            "criteria": dict(_VARIABLE_CRITERIA),
        },
        "aggregation": {
            "type": "choice",
            "instructions": "What time aggregation fits the question?",
            "criteria": dict(_AGGREGATION_CRITERIA),
        },
    }


def classify_question(question: str) -> JevClassification:
    """Classify one question with a single batched Jev call.

    Raises JevError on any transport/API/response problem. `confident`
    is True only when every classification meets JEV_MIN_CONFIDENCE
    (default 0.7).
    """
    data = _system_one(question, _build_questions())
    answers = _answers_of(data)
    intent, i_conf = _parse_choice(answers, "intent")
    variable, v_conf = _parse_choice(answers, "variable")
    aggregation, a_conf = _parse_choice(answers, "aggregation")
    threshold = _env_float("JEV_MIN_CONFIDENCE", _DEFAULT_MIN_CONFIDENCE)
    confident = min(i_conf, v_conf, a_conf) >= threshold
    usage = data.get("usage")
    return JevClassification(
        intent=intent,
        intent_confidence=i_conf,
        variable=variable,
        variable_confidence=v_conf,
        aggregation=aggregation,
        aggregation_confidence=a_conf,
        confident=confident,
        usage=usage if isinstance(usage, dict) else {},
    )


def choose_place(question: str, candidates: list[str]) -> str | None:
    """Disambiguate multiple place candidates with one Jev choice call.

    Returns the chosen canonical name, or None when Jev is unsure /
    fails -- the caller falls back to the LLM path.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    questions = {
        "place": {
            "type": "choice",
            "instructions": (
                "Which single place is the primary subject of this question?"
            ),
            "criteria": {c: c for c in candidates[:16]},
        }
    }
    try:
        answers = _answers_of(_system_one(question, questions))
        choice, conf = _parse_choice(answers, "place")
    except JevError as exc:
        log.info("jev: place disambiguation failed: %s", exc)
        return None
    threshold = _env_float("JEV_MIN_CONFIDENCE", _DEFAULT_MIN_CONFIDENCE)
    if conf < threshold or choice not in candidates:
        return None
    return choice


_HERE_RE = re.compile(r"\b(here|my area|near me|nearby)\b", re.IGNORECASE)


def extract_place(question: str) -> str | None:
    """Find the gazetteer place a question refers to, without any model.

    Case-insensitive longest-match over gazetteer names + aliases. One
    distinct canonical match -> use it; several -> Jev choice
    disambiguation; none (or a 'here'-style reference needing the user's
    location) -> None, and the caller falls back to the LLM path.
    """
    if _HERE_RE.search(question):
        # Needs user-location handling; the LLM path owns need_location.
        return None
    variants = name_variants()  # lowercased variant -> canonical name
    hits: list[tuple[int, str]] = []
    for variant, canonical in variants.items():
        if re.search(r"\b" + re.escape(variant) + r"\b", question, re.IGNORECASE):
            hits.append((len(variant), canonical))
    if not hits:
        return None
    hits.sort(reverse=True)  # longest match first
    ordered: list[str] = []
    for _, canonical in hits:
        if canonical not in ordered:
            ordered.append(canonical)
    return choose_place(question, ordered)


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = (
    "(january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
)


def _day_start(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _day_end(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_time(question: str, reference_date: str) -> TimeWindow:
    """Deterministic time-window extraction. Never calls a model.

    Priority: explicit ISO date; "Month D YYYY" / "D Month YYYY";
    relative words ("today"/"now", "yesterday", "tomorrow",
    "last|past N days/weeks/months", "last|past week/month/year",
    "this week/month"). No recognizable expression -> today, mirroring
    the prompt's "now"/"today" default for the parse step.
    """
    ref = date.fromisoformat(reference_date)
    q = question.lower()

    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", q)
    if m:
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            return TimeWindow(_day_start(d), _day_end(d), explicit=True)

    m = re.search(rf"\b{_MONTH_RE}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d{{2}})\b", q)
    if m:
        d = _safe_date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))
        if d:
            return TimeWindow(_day_start(d), _day_end(d), explicit=True)
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH_RE},?\s+(20\d{{2}})\b", q)
    if m:
        d = _safe_date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
        if d:
            return TimeWindow(_day_start(d), _day_end(d), explicit=True)

    if re.search(r"\b(today|now)\b", q):
        return TimeWindow(_day_start(ref), _day_end(ref), explicit=True)
    if re.search(r"\byesterday\b", q):
        d = ref - timedelta(days=1)
        return TimeWindow(_day_start(d), _day_end(d), explicit=True)
    if re.search(r"\btomorrow\b", q):
        d = ref + timedelta(days=1)
        return TimeWindow(_day_start(d), _day_end(d), explicit=True)

    m = re.search(r"\b(?:last|past)\s+(\d+)\s+days?\b", q)
    if m:
        d0 = ref - timedelta(days=int(m.group(1)))
        return TimeWindow(_day_start(d0), _day_end(ref), explicit=True)
    m = re.search(r"\b(?:last|past)\s+(\d+)\s+weeks?\b", q)
    if m:
        d0 = ref - timedelta(days=7 * int(m.group(1)))
        return TimeWindow(_day_start(d0), _day_end(ref), explicit=True)
    if re.search(r"\b(?:last|past)\s+week\b", q):
        d0 = ref - timedelta(days=7)
        return TimeWindow(_day_start(d0), _day_end(ref), explicit=True)
    m = re.search(r"\b(?:last|past)\s+(\d+)\s+months?\b", q)
    if m:
        d0 = ref - timedelta(days=30 * int(m.group(1)))
        return TimeWindow(_day_start(d0), _day_end(ref), explicit=True)
    if re.search(r"\b(?:last|past)\s+month\b", q):
        d0 = ref - timedelta(days=30)
        return TimeWindow(_day_start(d0), _day_end(ref), explicit=True)
    if re.search(r"\b(?:last|past)\s+year\b", q):
        d0 = ref - timedelta(days=365)
        return TimeWindow(_day_start(d0), _day_end(ref), explicit=True)
    if re.search(r"\bthis\s+week\b", q):
        d0 = ref - timedelta(days=6)
        return TimeWindow(_day_start(d0), _day_end(ref), explicit=True)
    if re.search(r"\bthis\s+month\b", q):
        d0 = ref - timedelta(days=29)
        return TimeWindow(_day_start(d0), _day_end(ref), explicit=True)

    return TimeWindow(_day_start(ref), _day_end(ref), explicit=False)
