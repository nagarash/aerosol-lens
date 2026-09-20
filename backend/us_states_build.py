"""One-time builder: merges US states + DC + Puerto Rico into the CURATED
gazetteer (data/gazetteer.sample.json), so common queries like "dust over
Georgia" or "smoke over Florida" resolve to the expected US region by
default instead of needing a disambiguating follow-up question.

WHY CURATED, NOT THE GEONAMES EXTENSION (data/gazetteer_extended.json,
backend/gazetteer_build.py): states/DC/PR are a small (52), fixed,
unambiguous, universally-recognized set -- exactly what the curated file
is for (see backend/geocode.py's module docstring: curated names are the
ONLY ones shown to the LLM via known_places_hint(), and curated always
wins a name collision with the extended GeoNames layer). This is also
what actually fixes the collision, with no code changes: once "Georgia"
is curated as the US state, backend/geocode.py::_lookup()'s existing
curated-over-extended precedence makes it win over the extended layer's
"Georgia" = the country (Tbilisi region), automatically.

WHY A REAL SHAPEFILE INSTEAD OF HAND-TYPED BBOXES: 52 states x 4
coordinates is a lot of numbers to get right from memory, and a wrong one
silently serves the wrong map region. Uses the US Census Bureau's
cartographic boundary file (public domain, cb_2022_us_state_20m) for
authoritative per-state extents, computed straight from the actual
geometry.

ALASKA CROSSES THE ANTIMERIDIAN: the western Aleutians have positive
longitude (~172E) while the mainland is negative (~-130W to -180W), so a
naive min/max bbox spans nearly the whole globe. Detected automatically
(bbox width > 180) and represented the same way the rest of this app
already does (see README's /grid antimeridian handling): west > east,
covering [west, 180] union [-180, east].

Also adds ONE extra entry found while eval-testing the LLM fallback path
(see backend/eval_llm_fallback.py's "relative_date_event" case): "Tonga"
was resolving to an unrelated same-named town in Cameroon via the
GeoNames extension, when every plausible query ("the eruption near
Tonga") means the Pacific island nation -- directly relevant since it's
one of the three events in data/event_catalog.sample.json. Sourced from
Natural Earth's admin-0 countries file (also public domain), same
methodology as the states.

Usage:
    pip install pyshp   # not in backend/requirements.txt -- this script's
                         # own one-time dependency, not a runtime one
    python backend/us_states_build.py
    python backend/us_states_build.py --cache-dir /tmp/geo_cache  # reuse downloads
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import shapefile  # pyshp

_STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_20m.zip"
_STATES_BASENAME = "cb_2022_us_state_20m"
_COUNTRIES_URL = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
_COUNTRIES_BASENAME = "ne_50m_admin_0_countries"

# name (as it'll appear in the curated gazetteer) -> NAME field value in
# the Natural Earth countries file to pull in alongside the US states.
# Kept to entries a real query in this app needs (see eval_llm_fallback.py);
# not a general country gazetteer -- data/gazetteer_extended.json's GeoNames
# layer already covers everything else, collisions and all, until someone
# hits one worth curating like this.
_EXTRA_COUNTRIES = {"Tonga": "Tonga"}

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CURATED_PATH = _DATA_DIR / "gazetteer.sample.json"
_DEFAULT_CACHE_DIR = _DATA_DIR / ".geo_cache"

# Renamed on the way in, both to read naturally and to avoid a second
# ambiguous "Washington" (the state already claims that name) -- see the
# module docstring in backend/geocode.py for the curated/extended split
# this feeds. Aliases for the renamed entry are added to backend/geocode.py
# by hand (_ALIASES), not here.
_RENAME = {"District of Columbia": "Washington, D.C."}


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return dest


def _extract(zip_path: Path, out_dir: Path, basename: str) -> Path:
    if not (out_dir / f"{basename}.shp").exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
    return out_dir / f"{basename}.shp"


def _bbox_from_points(points: list[tuple[float, float]]) -> list[float]:
    """[west, south, east, north], antimeridian-aware.

    A naive (min x, max x) is wrong once a shape has points on both sides
    of the +/-180 seam (see module docstring re: Alaska) -- detected by
    an implausible >180-degree naive width, in which case west/east come
    from the point closest to the seam on each side instead of the
    absolute extremes, matching this app's existing west>east convention
    for antimeridian bboxes.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    south, north = min(ys), max(ys)
    naive_west, naive_east = min(xs), max(xs)
    if naive_east - naive_west <= 180:
        return [round(naive_west, 6), round(south, 6), round(naive_east, 6), round(north, 6)]
    pos = [x for x in xs if x >= 0]
    neg = [x for x in xs if x < 0]
    west, east = min(pos), max(neg)
    return [round(west, 6), round(south, 6), round(east, 6), round(north, 6)]


def _read_places(shp_path: Path, name_field: str, wanted_names: set[str] | None = None) -> dict[str, list[float]]:
    sf = shapefile.Reader(str(shp_path))
    out: dict[str, list[float]] = {}
    for r in sf.shapeRecords():
        name = r.record[name_field]
        if wanted_names is not None and name not in wanted_names:
            continue
        out[name] = _bbox_from_points(r.shape.points)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--curated-path", type=Path, default=_CURATED_PATH)
    args = parser.parse_args()

    states_zip = _download(_STATES_URL, args.cache_dir / "us_states.zip")
    states_shp = _extract(states_zip, args.cache_dir / "us_states", _STATES_BASENAME)
    states = _read_places(states_shp, "NAME")
    print(f"read {len(states)} US states/DC/PR from {_STATES_URL}")

    countries_zip = _download(_COUNTRIES_URL, args.cache_dir / "countries.zip")
    countries_shp = _extract(countries_zip, args.cache_dir / "countries", _COUNTRIES_BASENAME)
    countries = _read_places(countries_shp, "NAME", wanted_names=set(_EXTRA_COUNTRIES.values()))
    print(f"read {len(countries)}/{len(_EXTRA_COUNTRIES)} extra countries from Natural Earth")

    new_places: dict[str, list[float]] = {}
    for name, bbox in states.items():
        new_places[_RENAME.get(name, name)] = bbox
    for curated_name, ne_name in _EXTRA_COUNTRIES.items():
        if ne_name in countries:
            new_places[curated_name] = countries[ne_name]

    with open(args.curated_path, encoding="utf-8") as f:
        curated = json.load(f)
    existing_names = {p["name"] for p in curated["places"]}

    added = 0
    for name in sorted(new_places):
        if name in existing_names:
            print(f"  skip (already curated): {name}")
            continue
        curated["places"].append({"name": name, "bbox": new_places[name]})
        added += 1
    curated["places"].sort(key=lambda p: p["name"].lower())

    with open(args.curated_path, "w", encoding="utf-8") as f:
        json.dump(curated, f, ensure_ascii=False, indent=None)
    print(f"added {added} places -> {args.curated_path} ({len(curated['places'])} total curated places)")


if __name__ == "__main__":
    main()
