"""One-time builder: data/gazetteer_extended.json from GeoNames.

Extends the deterministic place matcher (backend/geocode.py,
backend/jev.py::extract_place) with two GeoNames-derived layers beyond the
curated places in data/gazetteer.sample.json. DEFAULTS TO "WELL-KNOWN
PLACES PEOPLE ACTUALLY ASK ABOUT", not GeoNames' full catalog -- this
isn't a general-purpose geocoder, it's a matcher for a dust/aerosol app's
questions, and a name most people wouldn't recognize is just a source of
false-positive matches (and, pre-tuning, ~38k places / 4.7MB for what
should be a few thousand):

- cities: populated places at or above --min-city-population (default
  500,000 -- ~1,200 cities worldwide, not the ~32k you get at GeoNames'
  own cities15000.txt floor of 15,000)
  (download.geonames.org/export/dump/cities15000.zip, ~24k rows scanned)
- regions: named physical features useful for dust/smoke transport
  questions -- deserts, mountain ranges, basins, plateaus, plains, deltas
  -- filtered out of allCountries.zip (~420MB download, ~12M rows) by
  feature code AND a notability floor (see build_regions() and
  _REGION_MIN_ALT_NAMES_OVERRIDE: deserts get a much lower floor than
  mountain ranges, since there are genuinely few well-known deserts but
  thousands of well-known-adjacent mountain ridges). GeoNames only gives
  a lat/lon POINT for these, never a polygon, so each becomes a bbox of
  that point padded by a fixed per-feature-code radius (_REGION_PAD_DEG
  below) -- a real approximation, not a boundary. Natural Earth's
  physical-regions polygons would be more accurate; out of scope here.

Deliberately a SEPARATE file from data/gazetteer.sample.json: that file
is small on purpose because agent/prompts.py appends
geocode.known_places_hint() -- built from it -- to every LLM fallback
prompt. gazetteer_extended.json is merged in only for local matching
(geocode._lookup(), used by jev.extract_place() and geocode.resolve_place());
it is never shown to the model. See geocode.py's module docstring.

Usage:
    python backend/gazetteer_build.py                    # well-known cities + regions
    python backend/gazetteer_build.py --cities-only       # skip the 420MB pull
    python backend/gazetteer_build.py --min-city-population 100000  # broader (~4,300 cities)
    python backend/gazetteer_build.py --cache-dir /tmp/geonames  # reuse downloads

Downloads are cached in --cache-dir (default: data/.geonames_cache/, already
gitignored-adjacent -- add it to .gitignore if you keep it) so reruns don't
re-fetch. Safe to rerun; output is fully regenerated each time, never
merged incrementally.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

_CITIES_URL = "https://download.geonames.org/export/dump/cities15000.zip"
_CITIES_MEMBER = "cities15000.txt"
_ALL_COUNTRIES_URL = "https://download.geonames.org/export/dump/allCountries.zip"
_ALL_COUNTRIES_MEMBER = "allCountries.txt"

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_OUT = _DATA_DIR / "gazetteer_extended.json"
_DEFAULT_CACHE_DIR = _DATA_DIR / ".geonames_cache"

# GeoNames feature codes worth surfacing as named regions for a dust/smoke
# transport app -- deserts (dust sources), basins/plains/plateaus (where
# dust settles or is channeled), mountain ranges (transport barriers),
# named cultural/economic regions. See
# https://www.geonames.org/export/codes.html for the full list.
#
# Deliberately EXCLUDES ISL (island) and PEN (peninsula) and VAL (valley):
# each has 10k-100k+ rows in allCountries.txt, overwhelmingly tiny local
# features with no relevance to aerosol transport (a Finnish peninsula
# named "Aartoniemi", a swamp named "15 Mile Swamp") -- pure noise for
# this app, checked against a real allCountries.txt pull on 2026-09-19.
_REGION_FEATURE_CODES = {
    "DSRT",  # desert
    "ERG",   # sandy desert
    "MTS",   # mountains
    "PLAT",  # plateau
    "PLN",   # plain(s)
    "DLTA",  # delta
    "BSND",  # drainage basin
    "RGN",   # region
    "RGNL",  # natural region
    "RGNH",  # historical region
    "RGNE",  # economic region
    "UPLD",  # upland
}

# Fixed bbox half-width (degrees) around the GeoNames point for each region
# feature code -- rough, hand-picked per feature type, not derived from any
# actual boundary. Deserts/basins/mountain ranges are large; deltas/islands
# are small.
_REGION_PAD_DEG: dict[str, float] = {
    "DSRT": 3.0, "ERG": 2.0, "MTS": 2.0, "PLAT": 2.0, "PLN": 2.0,
    "BSND": 2.5, "UPLD": 1.5, "RGN": 2.0, "RGNL": 2.0, "RGNH": 1.5,
    "RGNE": 1.5, "VAL": 0.8, "DLTA": 0.6, "PEN": 1.0, "ISL": 0.3,
}
_DEFAULT_REGION_PAD_DEG = 1.0

_MIN_NAME_LEN = 3  # drop 1-2 char names -- too likely to false-positive match


def _city_pad_deg(population: int) -> float:
    """Bbox half-width for a city, scaled to population (bigger metro -> bigger viewport)."""
    if population >= 10_000_000:
        return 0.45
    if population >= 3_000_000:
        return 0.30
    if population >= 1_000_000:
        return 0.20
    if population >= 300_000:
        return 0.12
    return 0.08


def _bbox(lat: float, lon: float, pad: float) -> list[float]:
    # Clamp rather than wrap at the antimeridian/poles -- these are rare
    # for the feature set we pull, and /grid already has its own
    # antimeridian handling for whatever bbox ends up in a plan.
    west = max(lon - pad, -180.0)
    east = min(lon + pad, 180.0)
    south = max(lat - pad, -90.0)
    north = min(lat + pad, 90.0)
    return [round(west, 4), round(south, 4), round(east, 4), round(north, 4)]


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  using cached {dest} ({dest.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        chunk = resp.read(1 << 20)
        while chunk:
            out.write(chunk)
            read += len(chunk)
            if total:
                print(f"\r  {read / 1e6:.1f}/{total / 1e6:.1f} MB", end="", file=sys.stderr)
            chunk = resp.read(1 << 20)
    print(file=sys.stderr)
    tmp.rename(dest)
    return dest


def _iter_geonames_rows(zip_path: Path, member: str):
    """Stream tab-delimited rows out of a GeoNames zip without loading it whole."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as raw:
            for line in io.TextIOWrapper(raw, encoding="utf-8"):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 15:
                    continue
                yield fields


def _load_curated_names(sample_path: Path) -> set[str]:
    """Canonical names already in the curated gazetteer -- never overridden."""
    with open(sample_path, encoding="utf-8") as f:
        data = json.load(f)
    return {p["name"].strip().lower() for p in data["places"]}


def build_cities(cache_dir: Path, min_population: int, skip_names: set[str]) -> dict[str, dict]:
    zip_path = _download(_CITIES_URL, cache_dir / "cities15000.zip")
    out: dict[str, dict] = {}
    for fields in _iter_geonames_rows(zip_path, _CITIES_MEMBER):
        name, lat, lon = fields[1], fields[4], fields[5]
        population = int(fields[14] or 0)
        if population < min_population:
            continue
        key = name.strip().lower()
        if len(key) < _MIN_NAME_LEN or key in skip_names:
            continue
        existing = out.get(key)
        if existing is not None and existing["population"] >= population:
            continue  # keep the more populous of same-named cities
        out[key] = {
            "name": name.strip(),
            "bbox": _bbox(float(lat), float(lon), _city_pad_deg(population)),
            "population": population,
            "source": "geonames:cities",
        }
    print(f"  {len(out)} cities >= {min_population:,} population", file=sys.stderr)
    return out


# Per-feature-code override of the notability floor: DSRT/ERG (deserts)
# are the category most directly relevant to a dust-transport app, and
# there just aren't that many named ones worldwide -- 386 at alt>=3, vs.
# mountain ranges (MTS) at ~3,893 for alt>=5. A single global threshold
# high enough to cut MTS's noise (15) also cuts genuinely well-known
# deserts -- "Atacama Desert" and "Kalahari Desert" both sit below that
# (alt-name counts of 11 and 3 respectively, verified 2026-09-20). floor=3
# keeps them while still dropping the clearest junk (at floor=0 the same
# category includes a mistagged hotel, "Dessert Inn Hotel", and a raw
# GIS label, "Barren Landslides Areas" -- both gone by floor=3).
# BSND/DLTA/RGNH stay low too since they're tiny categories regardless of
# threshold (a handful of entries either way) -- no noise to cut.
# Everything else (MTS, PLN, PLAT, UPLD, RGN, RGNL, RGNE) uses the
# caller's floor, which should stay high.
_REGION_MIN_ALT_NAMES_OVERRIDE: dict[str, int] = {
    "DSRT": 3, "ERG": 3, "BSND": 3, "DLTA": 3, "RGNH": 3,
}


def build_regions(cache_dir: Path, skip_names: set[str], min_alt_names: int) -> dict[str, dict]:
    """Physical/cultural regions filtered by feature code AND a notability
    proxy: GeoNames has no population figure for these, but a row's
    alternatenames column (translations/local-language names) tracks
    real-world prominence surprisingly well -- "Gobi Desert" has 29,
    "15 Mile Swamp" has 0. min_alt_names is the floor for most feature
    codes (see _REGION_MIN_ALT_NAMES_OVERRIDE for the exceptions); 15
    was checked by hand against a real allCountries.txt pull
    (2026-09-19/20): cuts mountain ranges (MTS) from 29,654 raw matches
    to ~317 genuinely-known ranges.
    """
    zip_path = _download(_ALL_COUNTRIES_URL, cache_dir / "allCountries.zip")
    out: dict[str, dict] = {}
    seen = 0
    for fields in _iter_geonames_rows(zip_path, _ALL_COUNTRIES_MEMBER):
        seen += 1
        feature_code = fields[7]
        if feature_code not in _REGION_FEATURE_CODES:
            continue
        alt_names = fields[3]
        n_alt = len(alt_names.split(",")) if alt_names else 0
        floor = _REGION_MIN_ALT_NAMES_OVERRIDE.get(feature_code, min_alt_names)
        if n_alt < floor:
            continue
        name, lat, lon = fields[1], fields[4], fields[5]
        population = int(fields[14] or 0)
        key = name.strip().lower()
        if len(key) < _MIN_NAME_LEN or key in skip_names:
            continue
        existing = out.get(key)
        if existing is not None and existing["population"] >= population:
            continue
        pad = _REGION_PAD_DEG.get(feature_code, _DEFAULT_REGION_PAD_DEG)
        out[key] = {
            "name": name.strip(),
            "bbox": _bbox(float(lat), float(lon), pad),
            "population": population,
            "source": "geonames:regions",
            "feature_code": feature_code,
        }
        if seen % 2_000_000 == 0:
            print(f"  scanned {seen / 1e6:.0f}M rows...", file=sys.stderr)
    print(f"  {len(out)} regions across {len(_REGION_FEATURE_CODES)} feature codes", file=sys.stderr)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--min-city-population",
        type=int,
        default=500_000,
        help="default is 'well-known city' scale (~1,200 cities worldwide), not every town GeoNames tracks",
    )
    parser.add_argument(
        "--min-region-alt-names",
        type=int,
        default=15,
        help="notability floor for MOST physical regions -- deserts/basins/deltas use a lower fixed floor regardless of this flag (see _REGION_MIN_ALT_NAMES_OVERRIDE)",
    )
    parser.add_argument("--cities-only", action="store_true", help="skip the allCountries.zip pull (~420MB)")
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=_DATA_DIR / "gazetteer.sample.json",
        help="curated gazetteer whose names take precedence and are excluded here",
    )
    args = parser.parse_args()

    skip_names = _load_curated_names(args.sample_path)
    print(f"excluding {len(skip_names)} curated names (they already resolve)", file=sys.stderr)

    print("building cities layer...", file=sys.stderr)
    cities = build_cities(args.cache_dir, args.min_city_population, skip_names)

    regions: dict[str, dict] = {}
    if not args.cities_only:
        print("building regions layer...", file=sys.stderr)
        regions = build_regions(args.cache_dir, skip_names, args.min_region_alt_names)

    merged = {**cities, **regions}
    places = sorted(merged.values(), key=lambda p: p["name"].lower())
    payload = {
        "_note": (
            "GeoNames-derived extension to gazetteer.sample.json, built by "
            "backend/gazetteer_build.py. MATCH-ONLY: merged into "
            "backend/geocode.py's local lookup for jev.py place extraction "
            "and resolve_place(), but never shown to the LLM (known_places()/"
            "known_places_hint() read gazetteer.sample.json only, so the "
            "fallback prompt's size doesn't grow with this file)."
        ),
        "generated_from": "https://www.geonames.org (CC-BY 4.0)",
        "places": places,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"wrote {len(places)} places -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)


if __name__ == "__main__":
    main()
