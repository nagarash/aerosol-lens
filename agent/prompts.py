"""Prompts for the parse step: natural language -> QueryPlan JSON.

Keep the model small and fast here -- this step dominates time-to-first-
pixels (~1s of ~2s). The model does language understanding only; dataset
semantics come from mappings.py and guardrails from validator.py.
"""

from .mappings import AEROSOL_WORD_TO_VARIABLE, INTENT_DEFAULTS, SOURCE_TIME_RULES

SYSTEM_PROMPT = """\
You translate a user's natural-language question about air quality or \
atmospheric aerosols into a single JSON plan draft. Output ONLY the JSON \
object, no prose.

You NEVER emit coordinates or bounding boxes. You return a PLACE NAME; the \
backend resolves it to a bounding box from its own gazetteer. Invented \
coordinates are impossible by design -- do not try.

Plan draft schema:
{
  "intent": "health" | "plume" | "comparison",
  "level": "surface" | "column",
  "source": "google" | "merra2" | "cams",
  "variable": "PM25" | "PM10" | "DUEXTTAU" | "BCEXTTAU" | "OCEXTTAU" | "SUEXTTAU" | "SSEXTTAU" | "TOTEXTTAU",
  "place": "place name, e.g. 'Delhi', 'Sahara', 'US Midwest'",
  "time_start": "YYYY-MM-DDTHH:MM:SSZ",
  "time_end": "YYYY-MM-DDTHH:MM:SSZ",
  "aggregation": "hourly" | "daily" | "monthly_mean",
  "style": {"colormap": "aqi" | "aod_sequential", "mode": "continuous" | "exceedance", "opacity": 0.65},
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
- "place" is a NAME, never coordinates. The backend resolves names to
  bounding boxes from its own gazetteer; unknown names are reported as
  errors, never guessed at.
- Prefer the KNOWN_PLACES names appended to this prompt when the question
  refers to one ("Sahara desert" -> "Sahara", "New Delhi" -> "Delhi").
- If nothing matches, return your best single place name anyway.
- "here"/"my area" -> "place" = the USER_LOCATION string when provided;
  otherwise return {"error": "need_location"} and nothing else.

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


# NOTE: the parse step is wired in backend/app.py::_parse_with_agent, which
# uses SYSTEM_PROMPT + build_user_message() with litellm.completion(),
# response_format={"type": "json_object"}, model from LITELLM_MODEL, and
# appends backend/geocode.py::known_places_hint() to the system prompt.
