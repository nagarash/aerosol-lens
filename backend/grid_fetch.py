"""Parallel materialization for /grid's lazy per-granule slices (stdlib only).

The MERRA-2 read path opens each daily granule as a lazy xarray dataset
backed by HTTPS byte-range reads against the GES DISC archive. The
expensive step is `.compute()` on the selected slice: dozens of hourly
time steps fetched over separate TLS connections. Fetching each
granule's slice on its own thread turns the observed ~1s/step sequential
cost into roughly (steps/workers).

Thread-safety: every granule carries its own fsspec ReferenceFileSystem
instance (created per file in grid._open_datasets), so worker threads
share no mutable I/O state. Dask's threaded scheduler was already in
play inside a single compute(); this only adds one outer level.

GRID_FETCH_WORKERS (default 8) caps the pool; a single granule skips
the pool entirely and computes inline.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os

log = logging.getLogger("aerosol_lens.grid_fetch")

ENV_WORKERS = "GRID_FETCH_WORKERS"
DEFAULT_WORKERS = 8


def fetch_workers() -> int:
    try:
        return max(1, int(os.environ.get(ENV_WORKERS, DEFAULT_WORKERS)))
    except ValueError:
        return DEFAULT_WORKERS


def parallel_compute(das: list):
    """Materialize lazy per-granule DataArrays, fetching granules in parallel.

    Returns the computed arrays in input order. Any worker failure raises
    RuntimeError (grid.get_grid maps it to GridFetchError); the remaining
    workers still run to completion so one bad granule doesn't wedge the
    pool.
    """
    if len(das) == 1:
        return [das[0].compute()]
    workers = min(len(das), fetch_workers())
    log.info("grid_fetch: materializing %d granule slices on %d workers",
             len(das), workers)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="grid-fetch"
    ) as pool:
        futures = [pool.submit(_compute_one, da) for da in das]
        out = []
        for i, fut in enumerate(futures):
            try:
                out.append(fut.result())
            except Exception as exc:
                raise RuntimeError(
                    f"parallel fetch of granule slice {i} failed: {exc}"
                ) from exc
        return out


def _compute_one(da):
    return da.compute()
