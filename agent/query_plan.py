"""Typed QueryPlan: the single contract between the agent and the executors.

Everything downstream (caching, validation, data fetching, tile/grid
rendering, legend generation) consumes this model. The LLM fills it in;
deterministic code checks it (validator.py) before anything executes.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Intent = Literal["health", "plume", "comparison"]
Level = Literal["surface", "column"]
Source = Literal["google", "merra2", "cams"]
Aggregation = Literal["hourly", "daily", "monthly_mean"]
ViewMode = Literal["continuous", "exceedance"]


def _ensure_time_order(time_end: datetime, time_start: datetime | None) -> datetime:
    """Shared time-order check for QueryPlan and QueryPlanDraft."""
    if time_start is not None and time_end < time_start:
        raise ValueError("time_end must be >= time_start")
    return time_end


class StyleSpec(BaseModel):
    """How the result should be rendered. Never affects which data is fetched."""

    colormap: str = Field(
        default="aqi",
        description="Key into data/colormaps.json (e.g. 'aqi', 'aod_sequential').",
    )
    mode: ViewMode = Field(
        default="continuous",
        description="'continuous' = concentration heatmap; "
        "'exceedance' = binary above/below health guideline.",
    )
    opacity: float = Field(default=0.65, ge=0.0, le=1.0)


class QueryPlan(BaseModel):
    """A fully-specified, executable data request derived from a question."""

    intent: Intent = Field(
        description="'health' = is it safe to breathe (surface PM only); "
        "'plume' = track an aerosol plume/event (column AOD ok); "
        "'comparison' = compare two periods/events (may fan out)."
    )
    level: Level = Field(
        description="'surface' = ground-level concentrations (health-relevant); "
        "'column' = column-integrated AOD (plume tracking)."
    )
    source: Source = Field(
        description="'google' = live/forecast/<=30d history (frontend calls directly); "
        "'merra2' = historical archive, incl. speciated column AOD; "
        "'cams' = historical surface PM2.5/PM10."
    )
    variable: str = Field(
        description="Dataset variable, e.g. 'PM25', 'PM10', 'DUAOD' (dust AOD), "
        "'BCAOD', 'OCAOD', 'TOTEXTTAU' (total AOD)."
    )
    bbox: list[float] = Field(
        description="Bounding box [west, south, east, north] in decimal degrees, "
        "WGS84. Must satisfy w < e and s < n."
    )
    time_start: datetime = Field(description="Inclusive start of the time window (UTC).")
    time_end: datetime = Field(description="Inclusive end of the time window (UTC).")
    aggregation: Aggregation = Field(
        default="daily",
        description="Temporal aggregation of the underlying data.",
    )
    style: StyleSpec = Field(default_factory=StyleSpec)
    place_name: Optional[str] = Field(
        default=None, description="Human-readable place the bbox was resolved from."
    )
    caption: Optional[str] = Field(
        default=None,
        description="Auto-generated one-line description, e.g. "
        "'Dust AOD over the Sahara, 2026-08-01..2026-08-07 (MERRA-2)'.",
    )

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: list[float]) -> list[float]:
        if len(v) != 4:
            raise ValueError("bbox must be [west, south, east, north]")
        w, s, e, n = v
        if not (-180 <= w <= 180 and -180 <= e <= 180):
            raise ValueError("longitudes must be within [-180, 180]")
        if not (-90 <= s <= 90 and -90 <= n <= 90):
            raise ValueError("latitudes must be within [-90, 90]")
        if not (w < e):
            raise ValueError("bbox west must be < east (no antimeridian wrap yet)")
        if not (s < n):
            raise ValueError("bbox south must be < north")
        return v

    @field_validator("time_end")
    @classmethod
    def _check_time_order(cls, v: datetime, info) -> datetime:
        return _ensure_time_order(v, info.data.get("time_start"))

    @classmethod
    def from_draft(
        cls, draft: "QueryPlanDraft", bbox: list[float], place_name: str
    ) -> "QueryPlan":
        """Build an executable plan from a validated draft + resolved bbox.

        The draft carries the model's language understanding (intent, level,
        variable, time window); the bbox comes from deterministic geocoding,
        never from the model.
        """
        data = draft.model_dump(exclude={"place", "error"})
        data["bbox"] = bbox
        data["place_name"] = place_name
        return cls(**data)


class QueryPlanDraft(BaseModel):
    """What the LLM emits. Deliberately has NO bbox.

    The model returns a place NAME; the backend resolves it to coordinates
    via the local gazetteer (backend/geocode.py). Keeping coordinates out
    of the model's output makes coordinate hallucination structurally
    impossible: the model never sees or emits geography as numbers.
    """

    intent: Intent = Field(
        description="Same semantics as QueryPlan.intent."
    )
    level: Level = Field(description="Same semantics as QueryPlan.level.")
    source: Source = Field(description="Same semantics as QueryPlan.source.")
    variable: str = Field(
        description="Dataset variable, e.g. 'PM25', 'DUAOD', 'TOTEXTTAU'."
    )
    place: Optional[str] = Field(
        default=None,
        description="Place NAME only, e.g. 'Delhi', 'Sahara', 'US Midwest'. "
        "Never coordinates; the backend resolves the name to a bbox. "
        "May be omitted only when 'error' is set.",
    )
    time_start: datetime = Field(description="Inclusive start of the time window (UTC).")
    time_end: datetime = Field(description="Inclusive end of the time window (UTC).")
    aggregation: Aggregation = Field(default="daily")
    style: StyleSpec = Field(default_factory=StyleSpec)
    place_name: Optional[str] = Field(
        default=None, description="Echo of the place; backend overwrites with the canonical gazetteer name."
    )
    caption: Optional[str] = Field(default=None)
    error: Optional[str] = Field(
        default=None,
        description="Set to 'need_location' (and nothing else) when the "
        "question needs the user's location ('here'/'my area') and none was provided.",
    )

    @field_validator("place")
    @classmethod
    def _check_place(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v  # allowed only when error='need_location'
        if not v.strip():
            raise ValueError("place must be a non-empty place name")
        return v.strip()

    @field_validator("time_end")
    @classmethod
    def _check_time_order(cls, v: datetime, info) -> datetime:
        return _ensure_time_order(v, info.data.get("time_start"))
