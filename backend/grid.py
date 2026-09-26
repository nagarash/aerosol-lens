"""Grid slicer: serve MERRA-2 column-AOD slices as compact JSON for GET /grid.

Pipeline (kerchunk path):
    manifest (KERCHUNK_INDEX_PATH) -> pick reference files for [t0, t1]
    -> open each reference individually with xarray (each granule's time
       decodes against its OWN CF units; xr.concat along time)
    -> select (variable, bbox, time) -> aggregate -> downsample
    -> compact JSON {variable, units, lats, lons, values, ...}

Fast path: when backend/fields.py's pre-aggregated daily-mean Zarr
store covers every requested date, the manifest and the archive are
skipped entirely -- daily means are read from local disk and combined
per the aggregation rule (exact equivalents of collapse_time() given
uniform 24-step days). Responses carry "data_source": "fields" or
"data_source": "kerchunk" so callers can see which path served them.

Responses are cached on disk (see grid_cache: identical queries hit the
cache instead of re-reading the archive); each granule's slice is
materialized on its own thread (see grid_fetch.parallel_compute).

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
import logging
import os
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

from agent.mappings import MERRA2_COLUMN_VARIABLES, canonical_variable

try:  # imported as part of the backend package
    from backend.earthdata_auth import https_target_options, s3_target_options
except ImportError:  # run as a script: python backend/grid.py
    from earthdata_auth import https_target_options, s3_target_options

try:  # imported as part of the backend package
    from backend import grid_cache, grid_fetch
except ImportError:  # run as a script: python backend/grid.py
    import grid_cache
    import grid_fetch

try:  # imported as part of the backend package
    from backend import fields as fields_store
except ImportError:  # run as a script: python backend/fields.py
    import fields as fields_store

log = logging.getLogger("aerosol_lens.grid")

AGGREGATIONS = ("hourly", "daily", "monthly_mean")
MAX_NLON, MAX_NLAT = 360, 180

# MERRA-2 trails real time by several weeks. Observed 2026-09-12: the
# newest published tavg1_2d_aer_Nx granule was 2026-08-01 (~6 weeks
# behind). A requested window entirely newer than
# (today - MERRA2_LATENCY_DAYS) gets a 422 that says so plainly, rather
# than the generic "no indexed data" message. Overridable via env
# (tests set it to 0 against synthetic fixtures).
def _latency_days() -> int:
    try:
        return max(0, int(os.environ.get("MERRA2_LATENCY_DAYS", "45")))
    except (TypeError, ValueError):
        return 45


def newest_plausible_date() -> date:
    """Newest calendar date plausibly present in the MERRA-2 archive.

    The archive runs ~MERRA2_LATENCY_DAYS behind real time. Shared by the
    /grid latency gate and the /ask dateless-question clamp so both agree
    on where "latest available data" is.
    """
    return date.today() - timedelta(days=_latency_days())


def newest_available_date() -> date:
    """Newest calendar date with servable data, across local stores.

    Ground truth, not latency math: the newest date (<= newest_plausible_date)
    covered by the Zarr field store or the kerchunk manifest. The /ask
    dateless-question clamp uses this so plans always point at a date
    /grid can actually serve -- the latency edge alone can overshoot the
    index when the builder last ran before the archive's newest granule.

    Falls back to newest_plausible_date() when neither store reports
    anything; /grid then fails honestly (501/422) as before. Never raises:
    the clamp must not break /ask.
    """
    plausible = newest_plausible_date()
    newest: date | None = None

    def consider(day: date) -> None:
        nonlocal newest
        if day <= plausible and (newest is None or day > newest):
            newest = day

    # Zarr daily-mean store: union coverage across canonical variables.
    if fields_store.available():
        for var in MERRA2_COLUMN_VARIABLES:
            for d in fields_store.coverage(var):
                try:
                    consider(date.fromisoformat(d))
                except ValueError:
                    continue

    # Kerchunk manifest: ISO-date keys into per-granule reference files.
    try:
        manifest = _load_manifest(_index_path())
    except Exception:  # missing/corrupt manifest reads as no coverage
        manifest = {}
    for key in manifest.get("files", {}):
        try:
            consider(date.fromisoformat(key))
        except ValueError:
            continue

    return newest if newest is not None else plausible

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


def _ref_remote_protocol(ref_paths: list[str]) -> str | None:
    """The protocol the reference targets point at: "s3", "https", or None.

    None means every target is a local path (tests, local mirrors), which
    needs no credentials at all. The protocol decides which credential
    style the targets need -- S3 session keys or an Earthdata bearer
    header -- so it must be read from the references themselves rather
    than assumed.
    """
    for p in ref_paths:
        with open(p) as f:
            protocol = _protocol_of_refs(json.load(f))
        if protocol is not None:
            return protocol
    return None


def _protocol_of_refs(refs: dict) -> str | None:
    """Protocol of the targets inside one already-loaded reference dict."""
    for val in refs.get("refs", {}).values():
        target = val[0] if isinstance(val, (list, tuple)) else val
        if isinstance(target, str) and "://" in target:
            scheme = target.split("://", 1)[0]
            if scheme == "s3":
                return "s3"
            if scheme in ("http", "https"):
                return "https"
    return None


def _default_target_options(ref_paths: list[str]) -> dict:
    """fsspec options for reading the byte ranges the references name.

    Local targets need no auth ({}). HTTPS targets (the default for this
    deployment) carry the Earthdata bearer token. s3:// targets resolve
    via MERRA2_S3_ANON=1 -> anonymous, EARTHDATA_TOKEN -> exchanged
    session credentials, AWS_* in env -> standard AWS chain -- but note
    that direct S3 only works from inside AWS us-west-2; see
    backend.earthdata_auth.https_target_options.
    """
    protocol = _ref_remote_protocol(ref_paths)
    if protocol is None:
        return {}
    if protocol == "https":
        return https_target_options()
    return s3_target_options()


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


def _is_date_only(value: str) -> bool:
    """True when an ISO string carries a date but no time-of-day."""
    v = value.strip().rstrip("Z")
    return "T" not in v and " " not in v


def _parse_end_time(value: str, name: str) -> datetime:
    """Parse the INCLUSIVE end bound of a window.

    A bare date parses to midnight, which as an inclusive end would select
    nothing: MERRA-2's hourly steps sit at 00:30..23:30, so "2026-07-01"
    to "2026-07-01" would be an empty window over a fully indexed day.
    QueryPlan.time_end is documented as inclusive and the agent routinely
    emits bare dates, so a date-valued end means the whole of that day.
    """
    dt = _parse_time(value, name)
    if _is_date_only(value):
        dt = dt + timedelta(days=1) - timedelta(microseconds=1)
    return dt


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


def _open_datasets(ref_paths: list[str], target_options: dict | None = None):
    """Open reference files as a list of lazy xarray datasets (no data read yet).

    Each granule is opened INDIVIDUALLY so its time coordinate decodes
    against its own CF units, then the caller concatenates the decoded
    datasets along time. This is deliberate: real MERRA-2 granules use
    PER-FILE time origins (e.g. "minutes since 2026-08-01 00:30:00"), so
    the raw time values are identical across files. kerchunk's
    MultiZarrToZarr concatenates on raw values and would silently collapse
    every day onto the first granule's date; per-file open + xr.concat is
    the straightforward correct approach.

    Returning the list (instead of one concatenated dataset) also lets
    the caller materialize each granule's slice on its own thread
    (see grid_fetch.parallel_compute): granules are fully independent
    until the final concat.
    """
    import fsspec
    import xarray as xr

    datasets = []
    for p in ref_paths:
        with open(p) as f:
            refs = json.load(f)
        # remote_options -- NOT target_options -- carries credentials for
        # the referenced granules. target_options configures reading the
        # reference document itself, which is already a dict here, so it
        # silently left remote reads unauthenticated.
        # remote_options -- NOT target_options -- carries credentials for
        # the referenced granules; target_options configures reading the
        # reference document, which is already a dict here, so using it left
        # remote reads unauthenticated.
        #
        # asynchronous=True on BOTH sides is required by zarr 3: it
        # serialises the store to JSON and rebuilds it, and the rebuild
        # rejects any mismatch with "Reference-FS's target filesystem must
        # have same value of asynchronous". Local targets stay synchronous.
        remote_kwargs: dict = {}
        protocol = _protocol_of_refs(refs)
        if protocol is not None:
            remote_kwargs = {
                "remote_protocol": protocol,
                "remote_options": {**(target_options or {}), "asynchronous": True},
                "asynchronous": True,
            }
        fs = fsspec.filesystem("reference", fo=refs, **remote_kwargs)
        try:
            datasets.append(
                xr.open_dataset(fs.get_mapper(""), engine="zarr", chunks={})
            )
        except Exception as exc:
            raise GridFetchError(
                f"failed to open kerchunk reference {p!r}: {exc}"
            ) from exc

    return datasets


def _open_dataset(ref_paths: list[str], target_options: dict | None = None):
    """Open reference files as one xarray dataset, concatenated along time."""
    import xarray as xr

    datasets = _open_datasets(ref_paths, target_options=target_options)
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.concat(datasets, dim="time")
    except Exception as exc:
        raise GridFetchError(
            f"failed to concatenate granules along time: {exc}"
        ) from exc


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
    # NaN is not valid JSON (browsers reject it outright); missing cells
    # go out as null, which the frontend renders transparent.
    values = [
        [None if (isinstance(v, float) and np.isnan(v)) else round(v, 4) for v in row]
        for row in values.tolist()
    ]
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
# Pre-aggregated field store fast path (backend/fields.py)


def _window_dates(start: datetime, end: datetime) -> list[str]:
    """ISO date strings covering [start.date(), end.date()] inclusive."""
    days = []
    d = start.date()
    while d <= end.date():
        days.append(d.isoformat())
        d += timedelta(days=1)
    return days


def _aggregate_daily_means(data, date_strs: list[str], aggregation: str):
    """Collapse (ndays, ny, nx) daily means to one 2D field.

    Exact equivalents of collapse_time() on hourly data, given uniform
    24-step days (true for tavg1_2d_aer_Nx): hourly/daily = mean of the
    daily means; monthly_mean = mean of monthly means, where each
    monthly mean is the mean of that month's daily means (equal hours
    per day, so this equals the mean of the month's hourly steps).
    """
    import warnings

    import numpy as np

    with warnings.catch_warnings():
        # All-NaN cells (fully missing days) stay NaN, mirroring
        # skipna=True on the kerchunk path.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if aggregation in ("hourly", "daily"):
            return np.nanmean(data, axis=0)
        # monthly_mean: group days by calendar month, mean within each
        # month, then mean across months (equal weight per month).
        order: list[str] = []
        groups: dict[str, list[int]] = {}
        for i, d in enumerate(date_strs):
            key = d[:7]  # YYYY-MM
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(i)
        monthly = np.stack(
            [np.nanmean(data[groups[k]], axis=0) for k in order], axis=0
        )
        return np.nanmean(monthly, axis=0)


def _get_grid_fields(var: str, bbox: tuple[float, float, float, float],
                     start: datetime, end: datetime, agg: str,
                     date_strs: list[str]) -> dict:
    """Serve one aggregated slice from the pre-aggregated field store.

    Same response shape as the kerchunk path (downsample/to_grid_json
    are shared); time_steps_used counts hourly steps (24 per stored
    day) so the two paths report comparable provenance.
    """
    import numpy as np
    import xarray as xr

    try:
        data, lats, lons = fields_store.read(var, date_strs, bbox)
    except fields_store.FieldsError as exc:
        raise GridFetchError(
            f"field-store read failed for {var} {bbox}: {exc}"
        ) from exc
    field = _aggregate_daily_means(data, date_strs, agg)
    da = xr.DataArray(
        np.asarray(field, dtype=np.float64),
        dims=("lat", "lon"),
        coords={"lat": ("lat", lats), "lon": ("lon", lons)},
    )
    field, was_downsampled = downsample(da)
    result = to_grid_json(field, var, start, end, agg, was_downsampled,
                          steps=24 * len(date_strs))
    result["data_source"] = "fields"
    result["cache_hit"] = False
    return result


# --------------------------------------------------------------------------
# Orchestration


def get_grid(source: str, variable: str, bbox: str, t0: str, t1: str,
             agg: str = "daily", target_options: dict | None = None) -> dict:
    """Serve one aggregated MERRA-2 grid slice as compact JSON.

    Raises GridError subclasses; app.py maps them to HTTP statuses.
    `target_options` is passed to fsspec for remote targets; None (default)
    means "auto": local reference targets need no auth, remote ones resolve
    via backend.earthdata_auth (EARTHDATA_TOKEN exchange, AWS chain, or an
    honest 501). Tests use local files ({} is safe there).
    """
    _require_geo_deps()
    var = _check_args(source, variable, agg)
    # Snap the bbox to 2 decimals (~1 km): equivalent queries share cache
    # keys, and the served slice always matches the key exactly.
    w, s, e, n = grid_cache.normalize_bbox(_parse_bbox(bbox))
    start, end = _parse_time(t0, "t0"), _parse_end_time(t1, "t1")
    if end < start:
        raise BadGridRequestError("t1 must be >= t0.")
    # Archive-latency gate: MERRA-2 runs several weeks behind real time.
    # A window entirely newer than the plausible archive edge gets a
    # plain-spoken 422 (checked before the manifest, so the message names
    # the cause instead of "no indexed data").
    newest_plausible = newest_plausible_date()
    if start.date() > newest_plausible:
        raise BadGridRequestError(
            f"no MERRA-2 data yet for {start.date()}..{end.date()}: the "
            f"MERRA-2 archive runs several weeks behind real time "
            f"(newest plausible granule ~{newest_plausible}). "
            "Narrow the window to earlier dates."
        )
    date_strs = _window_dates(start, end)
    # Fast path: when the field store covers EVERY requested date, serve
    # from local disk -- no manifest, no archive reads. Partial coverage
    # falls through to the kerchunk path for the whole window.
    ref_paths: list[str] | None = None
    use_fields = (
        fields_store.available()
        and start.hour == 0 and start.minute == 0 and start.second == 0
        and end.hour == 23 and end.minute == 59 and end.second >= 59
        and set(date_strs) <= fields_store.coverage(var)
    )
    data_source = "fields" if use_fields else "kerchunk"
    if use_fields:
        log.info("grid: fields fast path var=%s t=%s..%s (%d days)",
                 var, start.date(), end.date(), len(date_strs))
    else:
        manifest = _load_manifest(_index_path())
        ref_paths = _refs_for_window(manifest, start, end)
        if target_options is None:
            target_options = _default_target_options(ref_paths)

    # Disk cache: identical queries are common (demos, shared links,
    # frontend re-asks) and historical granules are immutable.
    cdir = grid_cache.cache_dir()
    ckey = grid_cache.cache_key(source, var, agg, (w, s, e, n), start, end,
                                data_source=data_source)
    if cdir is not None:
        ttl = grid_cache.ttl_for_window(
            end.date(), date.today() - timedelta(days=_latency_days() + 1))
        hit = grid_cache.read(cdir, ckey, ttl)
        if hit is not None:
            hit = dict(hit)
            hit["cache_hit"] = True
            log.info("grid: cache HIT var=%s t=%s..%s src=%s",
                     var, start.date(), end.date(), data_source)
            return hit

    try:
        if use_fields:
            result = _get_grid_fields(var, (w, s, e, n), start, end, agg,
                                      date_strs)
        else:
            assert ref_paths is not None
            datasets = _open_datasets(ref_paths, target_options=target_options)
            # Slice each granule lazily, then materialize the slices in
            # parallel: granules are independent until the final concat, and
            # the HTTPS byte-range reads dominate wall-clock time.
            das = [slice_variable(ds, var, (w, s, e, n), start, end)
                   for ds in datasets]
            steps = sum(int(da.sizes["time"]) for da in das)
            fetched = grid_fetch.parallel_compute(das)
            import xarray as xr
            da = fetched[0] if len(fetched) == 1 else xr.concat(fetched, dim="time")
            field = collapse_time(da, agg)
            # fetched arrays are already in memory; collapse/downsample are eager.
            field, was_downsampled = downsample(field)
            result = to_grid_json(field, var, start, end, agg, was_downsampled, steps)
            result["data_source"] = "kerchunk"
            result["cache_hit"] = False
    except GridError:
        raise
    except Exception as exc:
        raise GridFetchError(
            f"failed to read grid slice for {var} {bbox} {t0}..{t1}: {exc}"
        ) from exc
    if cdir is not None:
        grid_cache.write(cdir, ckey, result)
    return result
