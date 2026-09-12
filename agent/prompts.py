"""Prompts for the parse step: natural language -> QueryPlan JSON.

Keep the model small and fast here -- this step dominates time-to-first-
pixels (~1s of ~2s). The model does language understanding only; dataset
semantics come from mappings.py and guardrails from validator.py.
"""

from .mappings import AEROSOL_WORD_TO_VARIABLE, INTENT_DEFAULTS, SOURCE_TIME_RULES

SYSTEM_PROMPT = """\
You translate a user's natural-language question about air quality or \
atmospheric aerosols into a single JSON QueryPlan. Output ONLY the JSON \
object, no prose.

QueryPlan schema:
{
  "intent": "health" | "plume" | "comparison",
  "level": "surface" | "column",
  "source": "google" | "merra2" | "cams",
  "variable": "PM25" | "PM10" | "DUAOD" | "BCAOD" | "OCAOD" | "SUAOD" | "SSAOD" | "TOTEXTTAU",
  "bbox": [west, south, east, north],   // decimal degrees, WGS84
  "time_start": "YYYY-MM-DDTHH:MM:SSZ",
  "time_end": "YYYY-MM-DDTHH:MM:SSZ",
  "aggregation": "hourly" | "daily" | "monthly_mean",
  "style": {"colormap": "aqi" | "aod_sequential", "mode": "continuous" | "exceedance", "opacity": 0.65},
  "place_name": "human-readable place, if any",
  "caption": "one-line description of the query"
}

Intent rules:
- "health": the user asks if air is safe to breathe, about running/exercising
  outside, allergies, sensitive groups, AQI. ALWAYS level="surface".
- "plume": the user asks to see/track/show a dust storm, smoke plume, ash
  cloud, haze event, "what is that brown cloud". Prefers level="column".
- "comparison": the user compares two periods or events ("vs last year",
  "compared to the 2020 fires"). Emit the plan for the PRIMARY period; the
  backend fans out.

Source routing:
""" + SOURCE_TIME_RULES + """
Time rules:
- Resolve relative dates ("last summer", "during the Camp Fire") against the
  conversation's reference date, given as REFERENCE_DATE in the user message.
- "now"/"today" -> time_start = today 00:00Z, time_end = today 23:59Z.
- Named events ("2020 California wildfires") resolve to their known windows;
  prefer the event catalog (data/event_catalog.sample.json) over guessing.

Place rules:
- Resolve named places to a bounding box using the gazetteer
  (data/gazetteer.sample.json). If the place is unknown, estimate a sensible
  bbox and set place_name to the raw string.
- "here"/"my area" -> the user's location, supplied as USER_LOCATION if known;
  otherwise ask for it (return {"error": "need_location"}).

Style rules:
- intent="health" -> colormap="aqi", mode="exceedance" when the question is
  about safety ("is it safe"), else mode="continuous".
- intent="plume" -> colormap="aod_sequential", mode="continuous".

Aerosol word -> variable hints:
""" + "\n".join(f"- {k} -> {v[0]} ({v[1]})" for k, v in AEROSOL_WORD_TO_VARIABLE.items()) + """
Intent defaults (override only with reason):
""" + "\n".join(f"- {k}: {v}" for k, v in INTENT_DEFAULTS.items()) + """
NEVER emit level="column" for intent="health". A validator rejects such \
plans; do not try to be clever about it.
"""


def build_user_message(question: str, reference_date: str, user_location: str | None = None) -> str:
    """Build the user message for the parse call."""
    msg = f"REFERENCE_DATE: {reference_date}\nQUESTION: {question}"
    if user_location:
        msg += f"\nUSER_LOCATION: {user_location}"
    return msg


# TODO(integration): wire SYSTEM_PROMPT + build_user_message() into a
# litellm.completion() call with response_format={"type": "json_object"}
# (or the provider's structured-output equivalent), model from
# os.environ["LITELLM_MODEL"]. Parse the JSON into QueryPlan, run
# validate_plan(), and on ValidationError retry once with the error text
# appended before giving up.
