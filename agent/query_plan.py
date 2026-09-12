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
        start = info.data.get("time_start")
        if start is not None and v < start:
            raise ValueError("time_end must be >= time_start")
        return v
