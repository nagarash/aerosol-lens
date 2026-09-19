"""Pre-aggregated daily-mean field store (local Zarr, stdlib + zarr + numpy).

Why this exists: serving /grid straight from NASA's archive costs
30-70s per query (dozens of HTTPS range reads per granule). A daily
mean field is 576x361 float32 (~0.8 MB/day/variable uncompressed); a
full year of the six canonical aerosol variables is ~1.8 GB
uncompressed, ~1 GB compressed -- small enough to live on the Fly
volume, where a query becomes a local disk read plus a numpy mean
(<200 ms) instead of thousands of network round trips.

Layout: {FIELDS_DIR}/{VAR}.zarr, one zarr Array per canonical
MERRA-2 column variable, dims (time, lat, lon), chunks (1, 361, 576),
float32, zstd-compressed. The time axis is an ordered list of
YYYY-MM-DD strings kept in the array attrs ("dates"); days are kept
sorted so consecutive-date reads are contiguous. Shared 1D coordinate
arrays live at {FIELDS_DIR}/lat.zarr and {FIELDS_DIR}/lon.zarr.

Grid convention (matches MERRA-2 tavg1_2d_aer_Nx natively):
    lat: 361 points, -90..90 step 0.5, ASCENDING
    lon: 576 points, -180..179.375 step 0.625, ASCENDING

bbox slicing in read() replicates backend/grid.py's _slice_lon /
_slice_lat semantics exactly (inclusive label slicing, antimeridian
wrap as [w..max]+[min..e] concatenated in that order), so the fields
fast path and the kerchunk path serve identical cells for the same
query. Aggregation equivalence (see grid.py):
    hourly/daily  = mean of daily means  (each day has 24 hourly steps)
    monthly_mean  = mean of monthly means, where each monthly mean is
                    itself the mean of that month's daily means.
Both are exact given uniform 24-step days.

zarr is imported lazily-guarded: without it every public function
raises FieldsError, and grid.py falls back to the kerchunk path.
"""

from __future__ import annotations

import bisect
import logging
import os
from datetime import date
from pathlib import Path

try:
    import zarr  # noqa: F401  (pinned in backend/requirements.txt)
except ImportError:  # pragma: no cover -- geo stack not installed
    zarr = None  # type: ignore[assignment]

import numpy as np

log = logging.getLogger("aerosol_lens.fields")

ENV_DIR = "FIELDS_DIR"
DEFAULT_DIR = "/data/fields"

# MERRA-2 tavg1_2d_aer_Nx native grid (see backend/kerchunk_index.py).
NLAT, NLON = 361, 576
LAT0, DLAT = -90.0, 0.5
LON0, DLON = -180.0, 0.625
CHUNKS = (1, NLAT, NLON)
_EPS = 1e-9  # tolerance for inclusive bbox-bound comparisons

COLLECTION = "tavg1_2d_aer_Nx"
SHORTNAME = "M2T1NXAER"


class FieldsError(Exception):
    """The field store is unavailable or a read/append is invalid."""


def _require_zarr():
    if zarr is None:
        raise FieldsError(
            "the 'zarr' package is not installed; the field store is "
            "unavailable (grid falls back to the kerchunk path)."
        )


def fields_dir() -> Path:
    return Path(os.environ.get(ENV_DIR, DEFAULT_DIR))


def available() -> bool:
    """True when the store directory exists and zarr is importable."""
    return zarr is not None and fields_dir().is_dir()


def _latitudes() -> np.ndarray:
    return LAT0 + DLAT * np.arange(NLAT, dtype=np.float64)


def _longitudes() -> np.ndarray:
    return LON0 + DLON * np.arange(NLON, dtype=np.float64)


def _var_path(var: str) -> Path:
    return fields_dir() / f"{var}.zarr"


def _open_var(var: str, mode: str = "r"):
    """Open a variable's store array (mode 'r' or 'a')."""
    _require_zarr()
    p = _var_path(var)
    if mode == "r" and not p.exists():
        raise FieldsError(f"no field store for variable {var!r} at {p}.")
    arr = zarr.open_array(str(p), mode=mode)
    if arr.ndim != 3 or arr.shape[1:] != (NLAT, NLON):
        raise FieldsError(
            f"field store {p} has unexpected shape {arr.shape}; "
            f"expected (*, {NLAT}, {NLON})."
        )
    return arr


def _ensure_coords() -> None:
    """Write the shared lat/lon coordinate arrays once (idempotent)."""
    _require_zarr()
    d = fields_dir()
    d.mkdir(parents=True, exist_ok=True)
    for name, values in (("lat", _latitudes()), ("lon", _longitudes())):
        p = d / f"{name}.zarr"
        if p.exists():
            continue
        a = zarr.create_array(
            str(p), shape=values.shape, dtype="float64",
            attributes={"standard_name": name},
        )
        a[:] = values


def append_day(var: str, date_str: str, data_2d: np.ndarray) -> bool:
    """Append one daily-mean field. Idempotent: returns False when the
    date is already present (no write). Returns True on append.

    Raises FieldsError on bad date format or wrong shape/dtype issues.
    Days are kept sorted by date so consecutive-date reads stay
    contiguous; out-of-order appends shift the tail (cheap at 365 days).
    """
    _require_zarr()
    try:
        date.fromisoformat(date_str)
    except ValueError:
        raise FieldsError(
            f"date must be YYYY-MM-DD, got {date_str!r}."
        ) from None
    data = np.asarray(data_2d, dtype=np.float32)
    if data.shape != (NLAT, NLON):
        raise FieldsError(
            f"daily field for {var} {date_str} has shape {data.shape}; "
            f"expected ({NLAT}, {NLON})."
        )
    _ensure_coords()
    p = _var_path(var)
    if not p.exists():
        arr = zarr.create_array(
            str(p), shape=(0, NLAT, NLON), chunks=CHUNKS, dtype="float32",
            compressors={"name": "zstd", "configuration": {"level": 3}},
            attributes={
                "dates": [],
                "source_collection": f"{SHORTNAME}/{COLLECTION}",
                "grid": f"lat {NLAT} x lon {NLON} "
                        f"({LAT0}..{-LAT0} step {DLAT}, "
                        f"{LON0}..{LON0 + DLON * (NLON - 1)} step {DLON})",
                "aggregation": "daily mean of 24 hourly MERRA-2 steps",
                "built_by": "backend/backfill.py",
            },
        )
    else:
        arr = zarr.open_array(str(p), mode="a")
    dates: list = list(arr.attrs.get("dates", []))
    if date_str in dates:
        return False  # idempotent: already stored
    idx = bisect.bisect_left(dates, date_str)
    n = len(dates)
    arr.resize((n + 1, NLAT, NLON))
    if idx < n:
        # Shift the tail down one slot. Read first: zarr does not
        # guarantee correct results for overlapping region copies.
        tail = arr[idx:n, :, :]
        arr[idx + 1:n + 1, :, :] = tail
    arr[idx, :, :] = data
    dates.insert(idx, date_str)
    arr.attrs["dates"] = dates
    log.info("fields: appended %s %s (%d days stored)", var, date_str, n + 1)
    return True


def coverage(var: str) -> set[str]:
    """Dates stored for a variable; empty set when there is no store."""
    if zarr is None:
        return set()
    p = _var_path(var)
    if not p.exists():
        return set()
    try:
        arr = zarr.open_array(str(p), mode="r")
        return set(arr.attrs.get("dates", []))
    except Exception as exc:  # corrupt store reads as empty; backfill repairs
        log.warning("fields: coverage read failed for %s: %s", var, exc)
        return set()


def _lon_index(w: float, e: float) -> tuple[np.ndarray, np.ndarray]:
    """Integer lon indices + lon values in RESPONSE order.

    Replicates grid._slice_lon: inclusive label slicing on ascending
    lons; antimeridian wrap (w > e) concatenates [w..max] + [min..e] in
    that order. Raises FieldsError when the bbox selects no cells.
    """
    lons = _longitudes()
    lo, hi = float(lons[0]), float(lons[-1])

    def piece(a: float, b: float) -> np.ndarray | None:
        a = max(a, lo)
        b = min(b, hi)
        if b <= a:
            return None
        sel = np.where((lons >= a - _EPS) & (lons <= b + _EPS))[0]
        return sel if len(sel) else None

    if w <= e:
        sel = piece(w, e)
        if sel is None:
            raise FieldsError("bbox selects no longitude cells.")
        return sel, lons[sel]
    left = piece(w, hi)
    right = piece(lo, e)
    parts = [p for p in (left, right) if p is not None]
    if not parts:
        raise FieldsError("bbox selects no longitude cells.")
    idx = np.concatenate(parts) if len(parts) == 2 else parts[0]
    return idx, lons[idx]


def _lat_index(s: float, n: float) -> tuple[np.ndarray, np.ndarray]:
    """Integer lat indices + lat values, ascending (response order).

    Replicates grid._slice_lat on ascending lats + grid.to_grid_json's
    ascending-latitude response convention.
    """
    if not s < n:
        raise FieldsError("bbox south must be < north.")
    lats = _latitudes()
    sel = np.where((lats >= s - _EPS) & (lats <= n + _EPS))[0]
    if not len(sel):
        raise FieldsError("bbox selects no latitude cells.")
    return sel, lats[sel]


def read(var: str, date_list: list[str],
         bbox: tuple[float, float, float, float]
         ) -> tuple[np.ndarray, list[float], list[float]]:
    """Read daily-mean fields for the given dates, sliced to bbox.

    Returns (data, lats, lons): data shape (ndays, ny, nx) float32 with
    NaN for missing cells; lats ascending; lons in slice order (wraps
    for antimeridian bboxes, exactly like the kerchunk path).
    Raises FieldsError when a date is missing or the bbox is empty.
    """
    arr = _open_var(var, mode="r")
    dates: list = list(arr.attrs.get("dates", []))
    pos = {d: i for i, d in enumerate(dates)}
    try:
        t_idx = np.array([pos[d] for d in date_list], dtype=np.int64)
    except KeyError as exc:
        raise FieldsError(
            f"field store for {var!r} has no {exc.args[0]}; "
            f"covered: {len(dates)} days."
        ) from None
    w, s, e, n = (float(v) for v in bbox)
    lon_idx, lons = _lon_index(w, e)
    lat_idx, lats = _lat_index(s, n)
    data = np.asarray(
        arr.oindex[t_idx, lat_idx, lon_idx], dtype=np.float32
    )
    return data, [float(v) for v in lats], [float(v) for v in lons]


def store_size_mb() -> float:
    """Total bytes of the field store in MB (0 when absent)."""
    d = fields_dir()
    if not d.is_dir():
        return 0.0
    total = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    return total / (1024 * 1024)
