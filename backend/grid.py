"""Grid slicer: serve MERRA-2 column-AOD slices as compact JSON for GET /grid.

Pipeline:
    manifest (KERCHUNK_INDEX_PATH) -> pick reference files for [t0, t1]
    -> open via fsspec ReferenceFileSystem + kerchunk MultiZarrToZarr
    -> select (variable, bbox, time) -> aggregate -> downsample
    -> compact JSON {variable, units, lats, lons, values, ...}

Only byte ranges covering the requested chunks cross the network; whole
NetCDF files are never downloaded. xarray/fsspec/kerchunk are imported
lazily so the backend stays importable without the geo stack (the /grid
endpoint then fails with an honest 501).

v1 simplifications (documented, not hidden):
- The time window collapses to ONE 2D field. `aggregation` selects the
  resampling rule applied before collapsing (hourly = straight mean of
  hourly steps; daily = daily means then averaged; monthly_mean =
  monthly means then averaged). Per-step animation frames are a later
  increment; the frontend time scrubber requests sub-windows.
- Antimeridian-crossing bboxes (w > e) are served as two slices
  concatenated along lon ([w..180] + [-180..e]); lons in the response
  wrap accordingly.
- Grid caps at 360 (lon) x 180 (lat) cells; larger slices are coarsened
  by integer factors (mean). Values rounded to 4 decimals.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from agent.mappings import MERRA2_COLUMN_VARIABLES, canonical_variable

AGGREGATIONS = ("hourly", "daily", "monthly_mean")
MAX_NLON, MAX_NLAT = 360, 180

DEFAULT_INDEX_PATH = str(
    Path(__file__).resolve().parent.parent / "data" / "kerchunk" / "index.json"
)


# --------------------------------------------------------------------------
# Errors. app.py maps these to HTTP status codes (see module docstring).


class GridError(Exception):
    """Base class for all grid-slicer failures."""


class GridDepsMissingError(GridError):
    """The geo stack (xarray/fsspec/kerchunk) is not installed. -> 501"""


class IndexNotBuiltError(GridError):
    """No kerchunk manifest found at KERCHUNK_INDEX_PATH. -> 501"""


class UnknownVariableError(GridError):
    """Variable is not a servable MERRA-2 column variable. -> 422"""


class BadGridRequestError(GridError):
    """Bad bbox/time/aggregation, or no indexed coverage. -> 422"""


class GridFetchError(GridError):
    """The index opened but the data fetch failed (S3, decode, ...). -> 502"""


# --------------------------------------------------------------------------
# Config


def _index_path() -> str:
    return os.environ.get("KERCHUNK_INDEX_PATH", DEFAULT_INDEX_PATH)


def _default_target_options() -> dict:
    """fsspec target options for opening the S3 byte ranges in references.

    The GES DISC bucket is protected: default to the standard AWS
    credential chain (AWS_* env vars, ~/.aws, IAM role) so deployments
    can use temporary Earthdata Login credentials. Set MERRA2_S3_ANON=1
    only for a genuinely public mirror.
    """
    if os.environ.get("MERRA2_S3_ANON") == "1":
        return {"anon": True}
    return {}


def _require_geo_deps() -> None:
    try:
        import xarray  # noqa: F401
        import fsspec  # noqa: F401
        import kerchunk  # noqa: F401
    except ImportError as exc:
        raise GridDepsMissingError(
            "grid reads need the geo stack (xarray, fsspec, kerchunk); "
            f"not installed: {exc}. Install backend/requirements.txt "
            "or run the backend from the project image."
        ) from exc


@lru_cache(maxsize=4)
def _load_manifest(path: str) -> dict:
    if not Path(path).exists():
        raise IndexNotBuiltError(
            f"no kerchunk manifest at {path!r}. Set KERCHUNK_INDEX_PATH to a "
            "manifest built by backend/kerchunk_index.py "
            "(python backend/kerchunk_index.py --help)."
        )
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------
# Request parsing / validation


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    try:
        w, s, e, n = (float(x) for x in bbox.split(","))
    except ValueError:
        raise BadGridRequestError(
            f"bbox must be 'w,s,e,n' in decimal degrees, got {bbox!r}."
        ) from None
    if not (-180 <= w <= 180 and -180 <= e <= 180):
        raise BadGridRequestError("bbox longitudes must be within [-180, 180].")
    if not (-90 <= s <= 90 and -90 <= n <= 90):
        raise BadGridRequestError("bbox latitudes must be within [-90, 90].")
    if not s < n:
        raise BadGridRequestError("bbox south must be < north.")
    return w, s, e, n


def _parse_time(value: str, name: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise BadGridRequestError(
            f"{name} must be ISO-8601, got {value!r}."
        ) from None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _check_args(source: str, variable: str, agg: str) -> str:
    if source != "merra2":
        raise BadGridRequestError(
            f"gridded reads are implemented for source='merra2' only in v1, "
            f"got {source!r}."
        )
    var = canonical_variable(variable)
    if var not in MERRA2_COLUMN_VARIABLES:
        raise UnknownVariableError(
            f"unknown MERRA-2 column variable {variable!r} "
            f"(canonicalized: {var!r}). Servable: "
            f"{sorted(MERRA2_COLUMN_VARIABLES)}."
        )
    if agg not in AGGREGATIONS:
        raise BadGridRequestError(
            f"agg must be one of {AGGREGATIONS}, got {agg!r}."
        )
    return var


def _refs_for_window(manifest: dict, t0: datetime, t1: datetime) -> list[str]:
    """Reference-file paths whose granule date overlaps [t0, t1]."""
    index_dir = Path(_index_path()).parent
    wanted = []
    for day, ref_name in sorted(manifest.get("files", {}).items()):
        try:
            d = datetime.fromisoformat(day).date()
        except ValueError:
            continue  # non-date keys (raw URLs) are skipped in v1
        if t0.date() <= d <= t1.date():
            wanted.append(str(index_dir / ref_name))
    if not wanted:
        raise BadGridRequestError(
            f"no indexed MERRA-2 data covers {t0.date()}..{t1.date()}. "
            "Extend the index (backend/kerchunk_index.py) or narrow the window."
        )
    return wanted


# --------------------------------------------------------------------------
# Dataset IO (fsspec + kerchunk + xarray)


def _open_dataset(ref_paths: list[str], target_options: dict | None = None):
    """Open reference files as one xarray dataset (lazy; no data read yet)."""
    import fsspec
    import xarray as xr

    ref_dicts = []
    for p in ref_paths:
        with open(p) as f:
            ref_dicts.append(json.load(f))

    if len(ref_dicts) == 1:
        combined = ref_dicts[0]
    else:
        from kerchunk.combine import MultiZarrToZarr

        # NOTE: concatenates on RAW time values, so this assumes the
        # collection uses a fixed time origin across granules (true for
        # MERRA-2: "hours since 1980-01-01"). Per-file time origins would
        # silently collapse to one granule's steps.
        combined = MultiZarrToZarr(
            ref_dicts, concat_dims=["time"], identical_dims=["lat", "lon"]
        ).translate()

    fs = fsspec.filesystem(
        "reference", fo=combined, target_options=target_options or {}
    )
    try:
        return xr.open_dataset(fs.get_mapper(""), engine="zarr", chunks={})
    except Exception as exc:
        raise GridFetchError(f"failed to open kerchunk references: {exc}") from exc


# --------------------------------------------------------------------------
# Pure slicing / aggregation / downsampling (unit-testable on any dataset)


def _slice_lon(da, w: float, e: float):
    """Slice longitude, handling antimeridian wrap by concatenating two pieces."""
    import numpy as np

    lons = da["lon"].values
    lo, hi = float(np.min(lons)), float(np.max(lons))

    def piece(a: float, b: float):
        a = max(a, lo)
        b = min(b, hi)
        if b <= a:
            return None
        if lons[0] > lons[-1]:  # descending
            return da.sel(lon=slice(b, a))
        return da.sel(lon=slice(a, b))

    if w <= e:
        part = piece(w, e)
        if part is None:
            raise BadGridRequestError("bbox selects no longitude cells.")
        return part
    # Antimeridian wrap: [w..max] + [min..e], concatenated along lon.
    left = piece(w, hi)
    right = piece(lo, e)
    parts = [p for p in (left, right) if p is not None]
    if not parts:
        raise BadGridRequestError("bbox selects no longitude cells.")
    if len(parts) == 1:
        return parts[0]
    import xarray as xr

    return xr.concat(parts, dim="lon")


def _slice_lat(da, s: float, n: float):
    lats = da["lat"].values
    if lats[0] > lats[-1]:  # descending
        return da.sel(lat=slice(n, s))
    return da.sel(lat=slice(s, n))


def slice_variable(ds, variable: str, bbox: tuple[float, float, float, float],
                  t0: datetime, t1: datetime):
    """Select (variable, bbox, time window) -> DataArray(time, lat, lon)."""
    import numpy as np

    if variable not in ds:
        raise UnknownVariableError(
            f"variable {variable!r} not present in the indexed collection."
        )
    da = ds[variable]
    w, s, e, n = bbox
    da = _slice_lon(da, w, e)
    da = _slice_lat(da, s, n)
    if da.sizes.get("lat", 0) == 0 or da.sizes.get("lon", 0) == 0:
        raise BadGridRequestError("bbox selects no grid cells.")
    # Time selection tolerant to datetime64 vs cftime coordinates.
    # Bounds are already UTC (see _parse_time); drop tzinfo for np.datetime64.
    t0_naive, t1_naive = t0.replace(tzinfo=None), t1.replace(tzinfo=None)
    try:
        da = da.sel(time=slice(np.datetime64(t0_naive), np.datetime64(t1_naive)))
    except Exception:
        try:
            import cftime

            da = da.sel(
                time=slice(
                    cftime.DatetimeGregorian(*t0.timetuple()[:6]),
                    cftime.DatetimeGregorian(*t1.timetuple()[:6]),
                )
            )
        except Exception as exc:
            raise GridFetchError(f"time selection failed: {exc}") from exc
    if da.sizes.get("time", 0) == 0:
        raise BadGridRequestError("time window selects no steps in the index.")
    return da


def collapse_time(da, aggregation: str):
    """Collapse the time dimension to one 2D field per the aggregation rule."""
    if aggregation == "hourly":
        return da.mean("time", skipna=True)
    if aggregation == "daily":
        return da.resample(time="1D").mean(skipna=True).mean("time", skipna=True)
    if aggregation == "monthly_mean":
        return da.resample(time="1MS").mean(skipna=True).mean("time", skipna=True)
    raise BadGridRequestError(f"unknown aggregation {aggregation!r}.")


def downsample(da, max_nlon: int = MAX_NLON, max_nlat: int = MAX_NLAT):
    """Coarsen to <= (max_nlon x max_nlat) cells. Returns (da, was_downsampled)."""
    ny, nx = da.sizes["lat"], da.sizes["lon"]
    fy = max(1, -(-ny // max_nlat))
    fx = max(1, -(-nx // max_nlon))
    if fy == 1 and fx == 1:
        return da, False
    da = da.isel(lat=slice(0, (ny // fy) * fy), lon=slice(0, (nx // fx) * fx))
    return da.coarsen(lat=fy, lon=fx, boundary="trim").mean(skipna=True), True


def to_grid_json(da, variable: str, t0: datetime, t1: datetime,
                 aggregation: str, downsampled: bool, steps: int) -> dict:
    """Serialize one 2D field to the /grid response shape."""
    import numpy as np

    # Latitude ascending in the response; lon keeps slice order
    # (wraps for antimeridian bboxes).
    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    lats = [round(float(v), 4) for v in da["lat"].values]
    lons = [round(float(v), 4) for v in da["lon"].values]
    values = np.asarray(da.values, dtype=float)
    values = np.round(values, 4).tolist()
    return {
        "variable": variable,
        "units": "dimensionless",  # aerosol optical thickness at 550 nm
        "lats": lats,
        "lons": lons,
        "values": values,
        "nx": len(lons),
        "ny": len(lats),
        "time_start": t0.isoformat(),
        "time_end": t1.isoformat(),
        "time_steps_used": steps,
        "aggregation": aggregation,
        "downsampled": downsampled,
        "source": "merra2",
    }


# --------------------------------------------------------------------------
# Orchestration


def get_grid(source: str, variable: str, bbox: str, t0: str, t1: str,
             agg: str = "daily", target_options: dict | None = None) -> dict:
    """Serve one aggregated MERRA-2 grid slice as compact JSON.

    Raises GridError subclasses; app.py maps them to HTTP statuses.
    `target_options` is passed to fsspec for remote targets; None (default)
    means "auto": anonymous only if MERRA2_S3_ANON=1, otherwise the standard
    AWS credential chain (temporary Earthdata Login credentials for the
    protected GES DISC bucket). Tests use local files ({} is safe there).
    """
    _require_geo_deps()
    if target_options is None:
        target_options = _default_target_options()
    var = _check_args(source, variable, agg)
    w, s, e, n = _parse_bbox(bbox)
    start, end = _parse_time(t0, "t0"), _parse_time(t1, "t1")
    if end < start:
        raise BadGridRequestError("t1 must be >= t0.")
    manifest = _load_manifest(_index_path())
    ref_paths = _refs_for_window(manifest, start, end)

    try:
        ds = _open_dataset(ref_paths, target_options=target_options)
        da = slice_variable(ds, var, (w, s, e, n), start, end)
        steps = int(da.sizes["time"])
        field = collapse_time(da, agg)
        # Materialize only the selected slice (single range-read burst).
        field = field.compute()
        field, was_downsampled = downsample(field)
        return to_grid_json(field, var, start, end, agg, was_downsampled, steps)
    except GridError:
        raise
    except Exception as exc:
        raise GridFetchError(
            f"failed to read grid slice for {var} {bbox} {t0}..{t1}: {exc}"
        ) from exc
