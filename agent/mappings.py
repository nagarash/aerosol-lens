"""Deterministic mapping tables: words -> variables, intent -> defaults.

The LLM proposes; these tables ground its choices. Keep the LLM on
language understanding and let tables decide dataset semantics.
"""

from typing import TypedDict


class IntentDefaults(TypedDict):
    level: str
    source: str
    aggregation: str


# How the agent should phrase aerosol words as dataset variables.
# Keys are lowercase trigger words; values are (variable, level) pairs.
# Column variable names are the true MERRA-2 tavg1_2d_aer_Nx names
# (extinction aerosol optical thickness at 550 nm, "*EXTTAU").
AEROSOL_WORD_TO_VARIABLE: dict[str, tuple[str, str]] = {
    # Surface concentrations (health-relevant)
    "pm2.5": ("PM25", "surface"),
    "pm25": ("PM25", "surface"),
    "fine particles": ("PM25", "surface"),
    "pm10": ("PM10", "surface"),
    "coarse particles": ("PM10", "surface"),
    "smog": ("PM25", "surface"),
    "haze": ("PM25", "surface"),
    # Column aerosol optical depth, speciated (plume-relevant)
    "dust": ("DUEXTTAU", "column"),
    "saharan dust": ("DUEXTTAU", "column"),
    "sandstorm": ("DUEXTTAU", "column"),
    "smoke": ("BCEXTTAU", "column"),  # black carbon traces combustion
    "wildfire smoke": ("BCEXTTAU", "column"),
    "ash": ("DUEXTTAU", "column"),  # volcanic ash reads closest to dust AOD
    "volcanic": ("SUEXTTAU", "column"),  # sulfate from eruptions
    "sulfate": ("SUEXTTAU", "column"),
    "sea salt": ("SSEXTTAU", "column"),
    "aerosol": ("TOTEXTTAU", "column"),
    "aod": ("TOTEXTTAU", "column"),
}

# Legacy/alternate spellings for MERRA-2 column variables, canonicalized
# to the true "*EXTTAU" names before validation. Kept so older cached plans
# and hand-written queries using the pre-v1 shorthand still resolve.
MERRA2_VARIABLE_ALIASES: dict[str, str] = {
    "DUAOD": "DUEXTTAU",
    "BCAOD": "BCEXTTAU",
    "OCAOD": "OCEXTTAU",
    "SUAOD": "SUEXTTAU",
    "SSAOD": "SSEXTTAU",
    "TOTAOD": "TOTEXTTAU",
}

# All column variables the MERRA-2 slicer knows how to serve.
MERRA2_COLUMN_VARIABLES: frozenset[str] = frozenset(
    {"TOTEXTTAU", "DUEXTTAU", "BCEXTTAU", "OCEXTTAU", "SUEXTTAU", "SSEXTTAU"}
)


def canonical_variable(name: str) -> str:
    """Map a variable name (or legacy alias) to its canonical dataset name."""
    return MERRA2_VARIABLE_ALIASES.get(name.strip().upper(), name.strip().upper())

# Default source/level/aggregation per intent. The agent may override the
# source when the time window demands it (see SOURCE_TIME_RULES below).
# "health" has no entry: health intents are unservable in this build (no
# surface source -- Google removed, CAMS unbuilt) and are refused outright.
INTENT_DEFAULTS: dict[str, IntentDefaults] = {
    "plume": {"level": "column", "source": "merra2", "aggregation": "daily"},
    "comparison": {"level": "surface", "source": "cams", "aggregation": "daily"},
}

# Time-based source routing. The agent applies these BEFORE emitting a plan.
# LEVEL DECIDES FIRST -- recency never overrides it:
# - level='column' (plume tracking, or "what kind of aerosol"): source='merra2',
#   ALWAYS, even for "today". MERRA-2 is the only servable source.
# - level='surface', older than 30 days: source='cams' (EAC4 reanalysis).
# The Google Air Quality source no longer exists in this build. NEVER emit
# source="google"; health questions return {"error": "surface_unavailable"}.
SOURCE_TIME_RULES = """
- If level='column' (plume, or the question asks what KIND of aerosol -- dust vs smoke vs sulfate): source='merra2', regardless of date. MERRA-2 is the only servable source.
- Historical surface questions (PM2.5/PM10, older periods): source='cams'.
- NEVER emit source="google": the Google Air Quality path was removed from this build. Health questions are answered with {"error": "surface_unavailable"}.
"""

# WHO guideline values (ug/m^3) used by the 'exceedance' view mode.
WHO_GUIDELINES_UG_M3 = {
    "PM25": {"annual": 5.0, "24h": 15.0},
    "PM10": {"annual": 15.0, "24h": 45.0},
    "O3": {"8h": 100.0},
    "NO2": {"annual": 10.0, "24h": 25.0},
    "SO2": {"24h": 40.0},
    "CO": {"8h": 10.0},  # mg/m^3
}
