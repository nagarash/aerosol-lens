"""Tests for backend/geocode.py, in particular the optional GeoNames
extension (data/gazetteer_extended.json, built by backend/gazetteer_build.py):
it must widen what resolve_place()/name_variants() can match without ever
widening known_places()/known_places_hint(), since the hint is appended to
every LLM fallback prompt (agent/prompts.py).

Run from the repo root:
    python backend/test_geocode.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.geocode as geocode_module
from backend.geocode import UnknownPlaceError, resolve_place


def _with_extended(places: list[dict], fn):
    """Point geocode at a throwaway extended gazetteer for the duration of fn()."""
    real_path = geocode_module._GAZETTEER_EXTENDED_PATH
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "gazetteer_extended.json"
        tmp_path.write_text(json.dumps({"places": places}), encoding="utf-8")
        geocode_module._GAZETTEER_EXTENDED_PATH = tmp_path
        geocode_module._load_extended.cache_clear()
        try:
            fn()
        finally:
            geocode_module._GAZETTEER_EXTENDED_PATH = real_path
            geocode_module._load_extended.cache_clear()


def test_extended_place_resolves_but_curated_list_is_unchanged():
    curated_known = geocode_module.known_places()
    curated_hint = geocode_module.known_places_hint()

    def check():
        name, bbox = resolve_place("Cairo")
        assert name == "Cairo"
        assert bbox == [30.9, 29.9, 31.5, 30.5]
        # known_places()/known_places_hint() -- what reaches the LLM prompt
        # -- must be untouched by the extended file.
        assert geocode_module.known_places() == curated_known
        assert geocode_module.known_places_hint() == curated_hint
        assert "Cairo" not in geocode_module.known_places()

    _with_extended(
        [{"name": "Cairo", "bbox": [30.9, 29.9, 31.5, 30.5], "source": "geonames:cities"}],
        check,
    )


def test_curated_entry_wins_on_name_collision():
    def check():
        # "Sahara" is curated; an extended entry with the same name (a
        # bogus, clearly-wrong bbox) must never shadow it.
        name, bbox = resolve_place("Sahara")
        assert name == "Sahara"
        assert bbox == [-17.0, 15.0, 35.0, 32.0]

    _with_extended(
        [{"name": "Sahara", "bbox": [0.0, 0.0, 0.1, 0.1], "source": "geonames:regions"}],
        check,
    )


def test_extended_name_appears_in_name_variants_for_jev_matching():
    def check():
        variants = geocode_module.name_variants()
        assert variants.get("cairo") == "Cairo"

    _with_extended(
        [{"name": "Cairo", "bbox": [30.9, 29.9, 31.5, 30.5], "source": "geonames:cities"}],
        check,
    )


def test_missing_extended_file_is_a_silent_noop():
    """A fresh checkout with no gazetteer_extended.json (no build script run
    yet) must behave exactly as before this feature existed."""
    real_path = geocode_module._GAZETTEER_EXTENDED_PATH
    geocode_module._GAZETTEER_EXTENDED_PATH = Path("/nonexistent/gazetteer_extended.json")
    geocode_module._load_extended.cache_clear()
    try:
        assert geocode_module._load_extended() == {}
        name, bbox = resolve_place("Sahara")
        assert name == "Sahara"
        try:
            resolve_place("nowhere in particular")
        except UnknownPlaceError:
            pass
        else:
            raise AssertionError("expected UnknownPlaceError")
    finally:
        geocode_module._GAZETTEER_EXTENDED_PATH = real_path
        geocode_module._load_extended.cache_clear()


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

if __name__ == "__main__":
    tests = sorted(
        (name, fn)
        for name, fn in list(globals().items())
        if name.startswith("test_") and callable(fn)
    )
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- test runner reports failures
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")
    total = len(tests)
    print(f"\n{total - failed}/{total} passed")
    sys.exit(1 if failed else 0)
