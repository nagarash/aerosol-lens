"""Deterministic place -> bbox resolution against the local gazetteer.

The LLM returns place NAMES, never coordinates. This module is the only
code that turns names into bounding boxes, which makes coordinate
hallucination structurally impossible: the model never sees or emits
geography as numbers.

Resolution is a local dict lookup (<50ms, no API call). Unknown places
raise UnknownPlaceError -- the API surfaces them as 422, naming the
place, instead of guessing.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_GAZETTEER_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "gazetteer.sample.json"
)

# Hand-curated aliases -> canonical gazetteer names. Deterministic; the
# model is instructed to prefer canonical names, this is the safety net.
_ALIASES: dict[str, str] = {
    "sahara desert": "Sahara",
    "saharan": "Sahara",
    "saharan dust": "Sahara",
    "new delhi": "Delhi",
    "la": "Los Angeles",
    "los angeles ca": "Los Angeles",
    "midwest": "US Midwest",
    "united states midwest": "US Midwest",
    "american midwest": "US Midwest",
    "amazon": "Amazon Basin",
    "amazon rainforest": "Amazon Basin",
    "gobi": "Gobi Desert",
    "indo gangetic plain": "Indo-Gangetic Plain",
    "arctic ocean": "Arctic",
    "se asia": "Southeast Asia",
    "atlanta ga": "Atlanta",
    "beijing china": "Beijing",
    "delhi india": "Delhi",
}


class UnknownPlaceError(Exception):
    """Raised when a place name is not in the gazetteer."""

    def __init__(self, place: str, known: list[str]):
        self.place = place
        super().__init__(
            f"unknown place {place!r}: not in the gazetteer, so no bounding "
            f"box can be resolved (and coordinates are never guessed). "
            f"Known places include: {', '.join(sorted(known)[:12])}..."
        )


@lru_cache(maxsize=1)
def _load() -> dict[str, list[float]]:
    """Canonical place name -> bbox [w, s, e, n]. Loaded once, cached."""
    with open(_GAZETTEER_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {p["name"]: list(p["bbox"]) for p in data["places"]}


def _lookup() -> dict[str, tuple[str, list[float]]]:
    """Lowercased name/alias -> (canonical name, bbox)."""
    table: dict[str, tuple[str, list[float]]] = {}
    for name, bbox in _load().items():
        table[name.lower()] = (name, bbox)
    for alias, canonical in _ALIASES.items():
        if canonical in _load():
            table[alias] = (canonical, _load()[canonical])
    return table


def name_variants() -> dict[str, str]:
    """All matchable place names: lowercased variant -> canonical name.

    Used by the deterministic longest-match place extraction
    (backend/jev.py). Covers gazetteer canonical names and aliases.
    """
    return {variant: canonical for variant, (canonical, _bbox) in _lookup().items()}


def known_places() -> list[str]:
    """Canonical gazetteer place names."""
    return sorted(_load())


def known_places_hint() -> str:
    """One-line prompt hint listing the place names the model may use."""
    return (
        "KNOWN_PLACES (use these exact names when the question refers to "
        "one): " + ", ".join(known_places()) + "."
    )


def resolve_place(place: str) -> tuple[str, list[float]]:
    """Resolve a place name to (canonical_name, bbox).

    Raises UnknownPlaceError if the name is not in the gazetteer.
    """
    key = place.strip().lower()
    hit = _lookup().get(key)
    if hit is None:
        raise UnknownPlaceError(place.strip(), known_places())
    return hit
