"""Backfill the pre-aggregated daily-mean field store (backend/fields.py).

For each date in [start, end]: read that day's hourly MERRA-2 granule
over HTTPS (Earthdata bearer token -- the same auth as the grid path),
compute the daily mean of each canonical aerosol variable, and append
it to the field store. Days are processed one at a time to bound memory
(~120 MB of chunk reads per day: 6 variables x 24 hourly steps x
361 x 576 float32); runs are idempotent (dates already present are
skipped), so an interrupted run resumes by simply re-running.

Transfer estimate: one variable-day range-reads 24 x 16 chunks of
~52 KB ~= 20 MB; six variables ~= 120 MB/day, ~= 44 GB for a full year.
The resulting store is ~5 MB/day (~1.8 GB/year) uncompressed.

Usage:
    export EARTHDATA_TOKEN=...   # server-side only, never in the repo
    python backend/backfill.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Defaults: end = today - MERRA2_LATENCY_DAYS (45), start = end - 364
days (one year). Env overrides: BACKFILL_START / BACKFILL_END.

The same entry points drive the admin endpoints (POST
/admin/backfill, GET /admin/backfill/status); only one backfill runs
at a time.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from datetime import date, timedelta

try:  # imported as part of the backend package
    from backend import fields as fields_store
    from backend.kerchunk_index import (
        DEFAULT_COLLECTION,
        DEFAULT_SHORTNAME,
        _collection_prefix,
        _s3_prefix,
        https_url_for_s3,
        urls_for_date_range,
    )
    from agent.mappings import MERRA2_COLUMN_VARIABLES
except ImportError:  # run as a script: python backend/backfill.py
    # Make the repo root importable, then use the package imports above.
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from backend import fields as fields_store  # noqa: E402
    from backend.kerchunk_index import (  # noqa: E402
        DEFAULT_COLLECTION,
        DEFAULT_SHORTNAME,
        _collection_prefix,
        _s3_prefix,
        https_url_for_s3,
        urls_for_date_range,
    )
    from agent.mappings import MERRA2_COLUMN_VARIABLES  # noqa: E402

log = logging.getLogger("aerosol_lens.backfill")

ENV_START = "BACKFILL_START"
ENV_END = "BACKFILL_END"
MAX_ERRORS_KEPT = 50


def _latency_days() -> int:
    try:
        return max(0, int(os.environ.get("MERRA2_LATENCY_DAYS", "45")))
    except (TypeError, ValueError):
        return 45


def default_range() -> tuple[date, date]:
    """Default backfill window: the last 365 days of plausible archive."""
    end = date.today() - timedelta(days=_latency_days())
    return end - timedelta(days=364), end


def resolve_range(start: str | date | None = None,
                  end: str | date | None = None) -> tuple[date, date]:
    """Resolve (start, end) from explicit args, else BACKFILL_* env, else
    default_range(). Raises ValueError on bad input or start > end."""
    def _parse(v: str | date | None, env_name: str, label: str) -> date | None:
        raw = v if v is not None else os.environ.get(env_name, "").strip()
        if not raw:
            return None
        if isinstance(raw, date):
            return raw
        try:
            return date.fromisoformat(raw)
        except ValueError:
            raise ValueError(
                f"{label} must be YYYY-MM-DD, got {raw!r}."
            ) from None

    s = _parse(start, ENV_START, "start")
    e = _parse(end, ENV_END, "end")
    d0, d1 = default_range()
    s = s or d0
    e = e or d1
    if s > e:
        raise ValueError(f"backfill start {s} is after end {e}.")
    return s, e


def granule_url(d: date) -> str:
    """HTTPS URL of the MERRA-2 granule for one date."""
    base = _collection_prefix(_s3_prefix(None), DEFAULT_SHORTNAME)
    s3_url = urls_for_date_range(base, DEFAULT_COLLECTION, d, d)[0]
    return https_url_for_s3(s3_url)


def read_day_means(d: date) -> dict[str, "object"]:
    """Read one granule over HTTPS and return {var: daily-mean float32}.

    Only the six canonical aerosol variables are read (chunked
    (time=1, lat=91, lon=144): ~20 MB/variable/day of range reads).
    Geo deps are imported lazily so this module stays importable
    without them; tests inject a fake reader instead.
    """
    import numpy as np

    import fsspec
    import xarray as xr

    from backend.earthdata_auth import https_target_options

    url = granule_url(d)
    opts = https_target_options()  # raises honestly when EARTHDATA_TOKEN is unset
    with fsspec.open(url, "rb", **opts) as f:
        ds = xr.open_dataset(f, engine="h5netcdf", chunks={})
        try:
            _check_grid(ds, d)
            out: dict[str, object] = {}
            for var in sorted(MERRA2_COLUMN_VARIABLES):
                if var not in ds:
                    raise RuntimeError(
                        f"granule for {d} has no variable {var!r}."
                    )
                da = ds[var]
                if da.sizes.get("time", 0) != 24:
                    log.warning("backfill: %s %s has %s time steps (expected 24)",
                                d, var, da.sizes.get("time"))
                mean = da.mean("time", skipna=True).compute()
                arr = np.asarray(mean.values, dtype=np.float32)
                if arr.shape != (fields_store.NLAT, fields_store.NLON):
                    raise RuntimeError(
                        f"granule for {d}: {var} daily mean has shape "
                        f"{arr.shape}, expected "
                        f"({fields_store.NLAT}, {fields_store.NLON})."
                    )
                out[var] = arr
            return out
        finally:
            ds.close()


def _check_grid(ds, d: date) -> None:
    """Fail fast if a granule's grid is not the expected MERRA-2 grid."""
    import numpy as np

    lat = np.asarray(ds["lat"].values, dtype=np.float64)
    lon = np.asarray(ds["lon"].values, dtype=np.float64)
    if lat.shape != (fields_store.NLAT,) or lon.shape != (fields_store.NLON,):
        raise RuntimeError(
            f"granule for {d}: unexpected grid shapes lat{lat.shape} "
            f"lon{lon.shape}."
        )
    if not (np.allclose(lat, fields_store._latitudes())
            and np.allclose(lon, fields_store._longitudes())):
        raise RuntimeError(
            f"granule for {d}: lat/lon coordinates do not match the "
            "expected MERRA-2 grid."
        )


# --------------------------------------------------------------------------
# Run state (shared with the admin endpoints)


_status_lock = threading.Lock()
_status: dict = {
    "running": False,
    "total_days": 0,
    "done_days": 0,
    "current_date": None,
    "errors": [],
}


def get_status() -> dict:
    with _status_lock:
        return {
            "running": _status["running"],
            "total_days": _status["total_days"],
            "done_days": _status["done_days"],
            "current_date": _status["current_date"],
            "errors": list(_status["errors"]),
        }


def is_running() -> bool:
    with _status_lock:
        return _status["running"]


def _set_status(**kwargs) -> None:
    with _status_lock:
        _status.update(kwargs)


def _record_error(d: date, exc: Exception) -> None:
    with _status_lock:
        errs = _status["errors"]
        if len(errs) < MAX_ERRORS_KEPT:
            errs.append({"date": d.isoformat(),
                         "error": f"{type(exc).__name__}: {exc}"})


# --------------------------------------------------------------------------
# The backfill itself


def _is_transient(exc: Exception) -> bool:
    """Best-effort check for retryable network/rate-limit failures.

    A missing granule (404 -> FileNotFoundError) is permanent: retrying
    won't help. Throttling (429), bad-gateway class errors, and dropped
    connections are worth another attempt.
    """
    if isinstance(exc, FileNotFoundError):
        return False
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (429, 500, 502, 503, 504):
        return True
    if type(exc).__name__ in (
        "ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout",
        "ChunkedEncodingError", "RemoteDisconnected",
    ):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in (
        "429", "503", "504", "rate limit", "too many requests",
        "timeout", "timed out", "connection reset",
        "temporarily unavailable", "service unavailable",
    ))


def _read_with_retry(read, d: date, attempts: int = 3):
    """Read one day's means, retrying transient failures with backoff.

    Returns (means, elapsed_seconds). Permanent failures raise immediately.
    """
    t0 = time.monotonic()
    last: Exception | None = None
    for i in range(attempts):
        try:
            means = read(d)
            return means, time.monotonic() - t0
        except Exception as exc:  # noqa: BLE001 - recorded per day below
            last = exc
            if not _is_transient(exc) or i == attempts - 1:
                raise
            wait = 5 * (2 ** i)
            log.warning("backfill: %s transient failure (%s), "
                        "retry %d/%d in %ds",
                        d, exc, i + 1, attempts - 1, wait)
            time.sleep(wait)
    raise last  # unreachable; keeps type checkers honest


def _store_day(d: date, needed: list, means: dict, elapsed: float) -> None:
    """Append one downloaded day to the field store (call on one thread)."""
    nbytes = 0
    for var in needed:
        arr = means[var]
        fields_store.append_day(var, d.isoformat(), arr)
        nbytes += getattr(arr, "nbytes", 0)
    log.info("backfill: %s done in %.1fs (+%.1f MB, %s)",
             d, elapsed, nbytes / 1e6, ",".join(needed))


def backfill_range(start: date | str | None = None,
                   end: date | str | None = None,
                   reader=None, workers: int = 1) -> dict:
    """Backfill daily means for [start, end] into the field store.

    `reader(d)` returns {var: (361, 576) float32}; defaults to
    read_day_means (HTTPS + Earthdata token). Idempotent: dates already
    present for ALL canonical variables are skipped without reading.
    Per-day failures are recorded in status["errors"] and the run
    continues. Returns the final status dict.

    `workers` parallelizes the day downloads (I/O-bound threads); the
    Zarr writes still happen serially on the calling thread, so the
    store never sees concurrent writers. workers=1 keeps the original
    strictly sequential behavior.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    s, e = resolve_range(start, end)
    read = reader or read_day_means
    days = []
    d = s
    while d <= e:
        days.append(d)
        d += timedelta(days=1)

    # Coverage check up front (cheap, local): only days missing at least
    # one variable get a download slot.
    todo: list = []
    for d in days:
        needed = [v for v in sorted(MERRA2_COLUMN_VARIABLES)
                  if d.isoformat() not in fields_store.coverage(v)]
        if not needed:
            log.info("backfill: %s skip (already stored)", d)
        else:
            todo.append((d, needed))

    _set_status(running=True, total_days=len(days),
                done_days=len(days) - len(todo),
                current_date=None, errors=[])
    log.info("backfill: %s..%s (%d days, %d to fetch, %d workers) -> %s",
             s, e, len(days), len(todo), workers,
             fields_store.fields_dir())
    done = len(days) - len(todo)
    try:
        if workers <= 1 or len(todo) <= 1:
            for d, needed in todo:
                _set_status(current_date=d.isoformat())
                try:
                    means, elapsed = _read_with_retry(read, d)
                    _store_day(d, needed, means, elapsed)
                except Exception as exc:  # per-day failure: record, continue
                    log.warning("backfill: %s FAILED: %s: %s",
                                d, type(exc).__name__, exc)
                    _record_error(d, exc)
                done += 1
                _set_status(done_days=done)
        else:
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="backfill") as ex:
                futs = {ex.submit(_read_with_retry, read, d): (d, needed)
                        for d, needed in todo}
                for fut in as_completed(futs):
                    d, needed = futs[fut]
                    try:
                        means, elapsed = fut.result()
                    except Exception as exc:  # per-day failure: record, continue
                        log.warning("backfill: %s FAILED: %s: %s",
                                    d, type(exc).__name__, exc)
                        _record_error(d, exc)
                    else:
                        _store_day(d, needed, means, elapsed)
                    done += 1
                    _set_status(done_days=done, current_date=d.isoformat())
    finally:
        _set_status(running=False, current_date=None)
    status = get_status()
    log.info("backfill: finished %s..%s: %d/%d days, %d errors, store=%.1f MB",
             s, e, status["done_days"], status["total_days"],
             len(status["errors"]), fields_store.store_size_mb())
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill daily-mean MERRA-2 fields into the local Zarr store.")
    parser.add_argument("--start", default=None,
                        help="First date (YYYY-MM-DD). Default: 364 days "
                             "before --end.")
    parser.add_argument("--end", default=None,
                        help="Last date (YYYY-MM-DD). Default: today - "
                             "MERRA2_LATENCY_DAYS.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    backfill_range(args.start, args.end)


if __name__ == "__main__":
    main()
