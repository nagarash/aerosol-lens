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
    "dust": ("DUAOD", "column"),
    "saharan dust": ("DUAOD", "column"),
    "sandstorm": ("DUAOD", "column"),
    "smoke": ("BCAOD", "column"),  # black carbon traces combustion
    "wildfire smoke": ("BCAOD", "column"),
    "ash": ("DUAOD", "column"),  # volcanic ash reads closest to dust AOD
    "volcanic": ("SUAOD", "column"),  # sulfate from eruptions
    "sulfate": ("SUAOD", "column"),
    "sea salt": ("SSAOD", "column"),
    "aerosol": ("TOTEXTTAU", "column"),
    "aod": ("TOTEXTTAU", "column"),
}

# Default source/level/aggregation per intent. The agent may override the
# source when the time window demands it (see SOURCE_TIME_RULES below).
INTENT_DEFAULTS: dict[str, IntentDefaults] = {
    "health": {"level": "surface", "source": "google", "aggregation": "hourly"},
    "plume": {"level": "column", "source": "merra2", "aggregation": "daily"},
    "comparison": {"level": "surface", "source": "cams", "aggregation": "daily"},
}

# Time-based source routing. The agent applies these BEFORE emitting a plan:
# - "now", "today", "forecast", or anything within the last 30 days -> google
#   (live/forecast/history; frontend calls the API directly)
# - older than 30 days AND surface-level -> cams (EAC4 reanalysis)
# - older than 30 days AND column/speciated -> merra2
# - "what kind of aerosol" (dust vs smoke vs sulfate) -> merra2 regardless
SOURCE_TIME_RULES = """
- If the question is about now, today, the coming days, or the last 30 days: source='google'.
- Else if level='surface': source='cams' (historical surface PM2.5/PM10).
- Else (level='column', or the question asks what KIND of aerosol): source='merra2'.
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
