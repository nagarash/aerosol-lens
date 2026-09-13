"""Deterministic validation + hard guardrails for QueryPlans.

The LLM proposes; this module disposes. No plan executes without passing
here. Guardrails are rules in code, not prompt suggestions.
"""

from .query_plan import QueryPlan


class ValidationError(Exception):
    """Raised when a QueryPlan fails deterministic validation."""


# Variables that are surface-level concentrations (health-relevant).
SURFACE_VARIABLES = {"PM25", "PM10", "O3", "NO2", "SO2", "CO"}

# Variables that are column-integrated optical depths (plume-relevant).
# True MERRA-2 tavg1_2d_aer_Nx names: extinction AOT at 550 nm.
COLUMN_VARIABLES = {
    "TOTEXTTAU",  # total aerosol optical depth
    "DUEXTTAU",  # dust
    "BCEXTTAU",  # black carbon
    "OCEXTTAU",  # organic carbon
    "SUEXTTAU",  # sulfate
    "SSEXTTAU",  # sea salt
}

# Max sensible time window per aggregation (keeps queries interactive).
MAX_WINDOW_DAYS = {"hourly": 7, "daily": 62, "monthly_mean": 365 * 5}


def validate_plan(plan: QueryPlan) -> QueryPlan:
    """Validate a QueryPlan, raising ValidationError on any violation."""
    _check_health_guardrail(plan)
    _check_level_variable_consistency(plan)
    _check_source_level_consistency(plan)
    _check_single_view(plan)
    _check_time_window(plan)
    return plan


def _check_health_guardrail(plan: QueryPlan) -> None:
    """GUARDRAIL: health questions must NEVER be answered with column data.

    A glowing AOD plume is not a breathing-safety answer. This is a hard
    rule in code -- the model does not get a vote.
    """
    if plan.intent == "health":
        if plan.level != "surface":
            raise ValidationError(
                f"intent='health' requires level='surface', got level={plan.level!r}. "
                "Column AOD must never answer a breathing-safety question."
            )
        if plan.variable in COLUMN_VARIABLES:
            raise ValidationError(
                f"intent='health' with column variable {plan.variable!r} rejected. "
                "Use a surface concentration variable (e.g. PM25)."
            )


def _check_level_variable_consistency(plan: QueryPlan) -> None:
    """The variable must live at the level the plan claims."""
    if plan.level == "surface" and plan.variable in COLUMN_VARIABLES:
        raise ValidationError(
            f"level='surface' is inconsistent with column variable {plan.variable!r}."
        )
    if plan.level == "column" and plan.variable in SURFACE_VARIABLES:
        raise ValidationError(
            f"level='column' is inconsistent with surface variable {plan.variable!r}."
        )


def _check_source_level_consistency(plan: QueryPlan) -> None:
    """GUARDRAIL: the source must be able to serve the plan's level.

    - google = Google Air Quality API, surface PM2.5 only (frontend calls it
      directly). It has no column/AOD data of any kind.
    - merra2 = MERRA-2 reanalysis, column AOD only (speciated aerosols).
    - cams = CAMS EAC4 reanalysis, historical surface PM2.5/PM10 only.
    A plume/column question about "today" must still use merra2 -- recency
    never overrides the level.
    """
    if plan.source == "google":
        if plan.level != "surface":
            raise ValidationError(
                f"source='google' serves surface data only, got level={plan.level!r}. "
                "Column/plume questions must use source='merra2'."
            )
        if plan.variable != "PM25":
            raise ValidationError(
                f"source='google' serves PM2.5 only, got variable={plan.variable!r}."
            )
    if plan.source == "merra2" and plan.level != "column":
        raise ValidationError(
            f"source='merra2' serves column AOD only, got level={plan.level!r}. "
            "Surface questions must use source='google' (recent) or 'cams' (historical)."
        )
    if plan.source == "cams" and plan.level != "surface":
        raise ValidationError(
            f"source='cams' serves surface PM2.5/PM10 only, got level={plan.level!r}."
        )


def _check_single_view(plan: QueryPlan) -> None:
    """GUARDRAIL: never mix surface and column in one view.

    A plan describes exactly one level. Comparing surface vs. column for
    the same event is done with two plans and a UI toggle, never one
    blended layer.
    """
    # Structural: QueryPlan has a single `level` field, so mixing is
    # impossible by construction. This check documents the invariant and
    # guards future schema changes.
    if not isinstance(plan.level, str) or plan.level not in ("surface", "column"):
        raise ValidationError(f"plan must have exactly one level, got {plan.level!r}.")


def _check_time_window(plan: QueryPlan) -> None:
    """Reject time windows that would make a query non-interactive."""
    days = (plan.time_end - plan.time_start).days + 1
    max_days = MAX_WINDOW_DAYS[plan.aggregation]
    if days > max_days:
        raise ValidationError(
            f"time window of {days} days exceeds the {max_days}-day limit for "
            f"aggregation={plan.aggregation!r}. Narrow the window or coarsen "
            "the aggregation."
        )
