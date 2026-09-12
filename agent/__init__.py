"""Aerosol Lens agent: natural language -> typed QueryPlan.

The agent's only job is translation. It never touches data directly;
it emits a QueryPlan (see query_plan.py) which deterministic code in
backend/ validates (validator.py) and executes.
"""

from .query_plan import QueryPlan, QueryPlanDraft, StyleSpec
from .validator import validate_plan, ValidationError
from .mappings import AEROSOL_WORD_TO_VARIABLE, INTENT_DEFAULTS

__all__ = [
    "QueryPlan",
    "QueryPlanDraft",
    "StyleSpec",
    "validate_plan",
    "ValidationError",
    "AEROSOL_WORD_TO_VARIABLE",
    "INTENT_DEFAULTS",
]
