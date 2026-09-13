"""kerchunk index builder: one-time scan of MERRA-2 NetCDF on AWS Open Data.

MERRA-2 lives as NetCDF-4 on S3. Downloading whole files per query
would kill latency; instead we build a JSON index of byte ranges ONCE with
kerchunk, then per-query xarray does lazy HTTP range reads of only the
chunks covering (variable, bbox, time).

Verified source layout (2026-09-12; bucket confirmed via the AWS Registry
of Open Data page and GES DISC tutorials)::

    s3://gesdisc-cumulus-prod-protected/MERRA2/M2T1NXAER.5.12.4/2024/01/MERRA2_400.tavg1_2d_aer_Nx.20240101.nc4

Collection facts (tavg1_2d_aer_Nx, M2T1NXAER):
- hourly, time-averaged; one file per day, time = 24 steps at
  00:30, 01:30, ..., 23:30 UTC
- grid: lon 576 (0.625 deg, -180..179.375), lat 361 (0.5 deg, -90..90)
- aerosol extinction AOT variables at 550 nm: TOTEXTTAU, DUEXTTAU,
  BCEXTTAU, OCEXTTAU, SUEXTTAU, SSEXTTAU
- granule size ~476 MB/day (all variables); aerosol variables are chunked
  (time=1, lat=91, lon=144), i.e. ~52 KB per chunk for DUEXTTAU (f4).
  A query range-reads only the chunks intersecting (variable, bbox, time)
  -- far cheaper than the old whole-day-chunk assumption implied.
- time is PER-FILE: each granule's time units are its own, e.g. "minutes
  since 2026-08-01 00:30:00". The slicer opens each reference individually
  and concatenates the decoded datasets; never merge on raw time values.

AUTHENTICATION (important): the archive is *protected* — anonymous
reads are rejected. Access requires an Earthdata Login account; set
EARTHDATA_TOKEN to a long-lived bearer token (generate one in the
Earthdata Login profile; server-side only, never in the repo).

ACCESS PATH (verified 2026-09-13): GES DISC grants *direct S3* access
only to callers running inside AWS us-west-2. From anywhere else — a
laptop, Fly.io, any non-AWS host — the s3credentials exchange succeeds
and returns valid session keys, but every GetObject still comes back
403 Forbidden, because the denial is by request origin rather than by
token. So the builder defaults to MERRA2_ACCESS=https, reading the same
bytes over https://data.gesdisc.earthdata.nasa.gov/data/... with the
bearer token, which works from any network and supports the same ranged
reads. Set MERRA2_ACCESS=s3 only when genuinely running in us-west-2.

Usage:
    export EARTHDATA_TOKEN=...   # or the three AWS_* vars, see above
    python kerchunk_index.py --collection tavg1_2d_aer_Nx --out ../data/kerchunk/
    python kerchunk_index.py --collection tavg1_2d_aer_Nx --out ../data/kerchunk/ --limit 3
    MERRA2_S3_PREFIX=s3://other-bucket/path python kerchunk_index.py ...

Output: one JSON reference file per NetCDF file, plus a manifest
``index.json`` mapping ISO date -> reference file. The builder is
resumable: files that already have a reference JSON are skipped.
Scope it with --limit for a first cheap run; the slicer only needs the
dates your queries cover.

Chunking (verified 2026-09-12 on a real granule): MERRA-2 aerosol
variables are chunked (time=1, lat=91, lon=144), ~52 KB per chunk for
DUEXTTAU. A bbox slice range-reads only the chunks covering
(variable, bbox, time), so per-query byte reads are far cheaper than a
whole-day fetch. If access patterns ever outgrow that, the documented
upgrade path is rechunking to Zarr/COG with spatial chunks (see README).
The builder prints per-variable chunk sizes so you can see this before
committing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

try:  # imported as part of the backend package
    from backend.earthdata_auth import (
        https_target_options,
        https_url_for_s3,
        s3_target_options,
    )
except ImportError:  # run as a script: python backend/kerchunk_index.py
    from earthdata_auth import (  # type: ignore[no-redef]
        https_target_options,
        https_url_for_s3,
        s3_target_options,
    )

DEFAULT_COLLECTION = "tavg1_2d_aer_Nx"
DEFAULT_SHORTNAME = "M2T1NXAER"
# Verified 2026-09-12 via registry.opendata.aws/nasa-merra-2 and GES DISC
# tutorials. NOTE: this bucket is *protected* — Earthdata Login temporary
# credentials are required (see module docstring); anonymous reads fail.
DEFAULT_PREFIX = "s3://gesdisc-cumulus-prod-protected/MERRA2"

FILENAME_DATE_RE = re.compile(r"\.(\d{8})\.nc4$")


def _s3_prefix(prefix: str | None) -> str:
    return (prefix or os.environ.get("MERRA2_S3_PREFIX") or DEFAULT_PREFIX).rstrip("/")


def _access_mode() -> str:
    """"https" (default) or "s3" -- how granule bytes are fetched.

    HTTPS is the default because GES DISC grants direct S3 access only to
    callers inside AWS us-west-2; everywhere else GetObject returns 403
    even with valid session credentials. Set MERRA2_ACCESS=s3 when the
    builder genuinely runs in-region.
    """
    mode = (os.environ.get("MERRA2_ACCESS") or "https").strip().lower()
    if mode not in ("https", "s3"):
        raise ValueError(f"MERRA2_ACCESS must be 'https' or 's3', got {mode!r}")
    return mode


def _options_for(url: str) -> dict:
    """fsspec options appropriate to the URL's protocol."""
    if url.startswith(("http://", "https://")):
        return https_target_options()
    return s3_target_options()


def _collection_prefix(prefix: str, shortname: str) -> str:
    # <prefix>/M2T1NXAER.5.12.4/**.nc4  (version dir mirrors GES DISC)
    return f"{prefix}/{shortname}.5.12.4"


def urls_for_date_range(
    collection_prefix: str, collection: str, start: date, end: date
) -> list[str]:
    """Construct MERRA-2 file URLs directly for a date range.

    GES DISC S3 credentials explicitly deny s3:ListBucket, so globbing
    the bucket fails. File names are deterministic, so build the URLs
    directly — only GetObject range reads are needed downstream.
    """
    urls = []
    d = start
    while d <= end:
        urls.append(
            f"{collection_prefix}/{d.year:04d}/{d.month:02d}/"
            f"MERRA2_400.{collection}.{d.year:04d}{d.month:02d}{d.day:02d}.nc4"
        )
        d += timedelta(days=1)
    return urls


def list_source_files(
    collection_prefix: str, limit: int | None = None
) -> list[str]:
    """List NetCDF file URLs under the collection prefix.

    Uses the standard AWS credential chain by default (the GES DISC
    bucket is protected); MERRA2_S3_ANON=1 forces anonymous access.
    """
    import fsspec

    fs = fsspec.filesystem("s3", **s3_target_options())
    pattern = f"{collection_prefix}/**/*.nc4"
    urls = sorted(fs.glob(pattern))
    if limit is not None:
        urls = urls[:limit]
    return [u if u.startswith("s3://") else f"s3://{u}" for u in urls]


# Scanning one granule's HDF5 metadata costs hundreds of scattered small
# reads. fsspec's default 5 MB block means each one drags in 5 MB; 1 MB
# measured ~25% faster end to end (419s -> 319s per granule from Fly sjc)
# without becoming latency-bound. Tune via MERRA2_BLOCK_SIZE if the
# builder ever runs somewhere with very different RTT.
DEFAULT_BLOCK_SIZE = 1_048_576


def _block_size() -> int:
    try:
        return max(65_536, int(os.environ.get("MERRA2_BLOCK_SIZE", "")))
    except (TypeError, ValueError):
        return DEFAULT_BLOCK_SIZE


def reference_for_url(url: str) -> dict:
    """Build a kerchunk reference dict for one remote NetCDF file."""
    import fsspec
    from kerchunk.hdf import SingleHdf5ToZarr

    with fsspec.open(url, "rb", block_size=_block_size(), **_options_for(url)) as f:
        return SingleHdf5ToZarr(f, url).translate()


def date_of_filename(url: str) -> date | None:
    """Extract the granule date from a MERRA-2 filename (...YYYYMMDD.nc4)."""
    m = FILENAME_DATE_RE.search(url)
    if not m:
        return None
    s = m.group(1)
    return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))


def build_index(
    collection: str = DEFAULT_COLLECTION,
    out_dir: Path = Path("../data/kerchunk"),
    limit: int | None = None,
    prefix: str | None = None,
    shortname: str = DEFAULT_SHORTNAME,
    start_date: date | None = None,
    end_date: date | None = None,
    workers: int = 1,
) -> Path:
    """Build kerchunk references for one MERRA-2 collection.

    Writes ``<stem>.json`` per NetCDF file plus ``index.json`` manifest.
    Resumable: existing reference files are skipped. Returns the manifest path.
    """
    base = _collection_prefix(_s3_prefix(prefix), shortname)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "index.json"
    manifest: dict = {"collection": collection, "prefix": base, "files": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    if start_date is not None and end_date is not None:
        urls = urls_for_date_range(base, collection, start_date, end_date)
    else:
        urls = list_source_files(base, limit)
    if _access_mode() == "https":
        # Direct S3 is in-region-only (us-west-2); the HTTPS archive
        # endpoint serves the same bytes anywhere with a bearer token.
        # The references then record https:// targets, and the slicer
        # picks the matching credentials off that protocol.
        urls = [https_url_for_s3(u) for u in urls]
    if not urls:
        raise RuntimeError(
            f"no .nc4 files found under {base!r}. The GES DISC bucket is "
            "protected: export temporary Earthdata Login credentials "
            "(AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN from "
            "https://data.gesdisc.earthdata.nasa.gov/s3credentials) or check "
            "MERRA2_S3_PREFIX / S3 reachability."
        )
    print(f"indexing {len(urls)} file(s) from {base} "
          f"({workers} worker(s), block_size={_block_size()})")

    lock = threading.Lock()
    reported = [False]

    def _one(url: str) -> tuple[str, str, str]:
        """Index one granule. Returns (url, ref_name, status)."""
        ref_name = Path(url).stem + ".json"
        ref_path = out_dir / ref_name
        if ref_path.exists():
            return url, ref_name, "skip"
        refs = reference_for_url(url)
        # Write via a temp file + atomic rename: a half-written reference
        # would otherwise be indistinguishable from a complete one on the
        # next (resumable) run, and would be skipped forever.
        tmp = ref_path.with_name(ref_path.name + ".tmp")
        tmp.write_text(json.dumps(refs))
        tmp.replace(ref_path)
        with lock:
            if not reported[0]:
                _report_chunking(url, refs)
                reported[0] = True
        return url, ref_name, "built"

    def _record(url: str, ref_name: str) -> None:
        d = date_of_filename(url)
        manifest["files"][d.isoformat() if d else url] = ref_name
        manifest_path.write_text(json.dumps(manifest, indent=1))

    failures: list[tuple[str, Exception]] = []
    if workers <= 1:
        for url in urls:
            try:
                u, ref_name, status = _one(url)
            except Exception as exc:  # keep going; the build is resumable
                print(f"  FAILED {url}: {type(exc).__name__}: {exc}", flush=True)
                failures.append((url, exc))
                continue
            print(f"  {status}: {ref_name}", flush=True)
            _record(u, ref_name)
    else:
        # Threads, not processes: the work is dominated by network round
        # trips, and h5py releases the GIL around reads.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, u): u for u in urls}
            for fut in as_completed(futures):
                url = futures[fut]
                try:
                    u, ref_name, status = fut.result()
                except Exception as exc:
                    print(f"  FAILED {url}: {type(exc).__name__}: {exc}", flush=True)
                    failures.append((url, exc))
                    continue
                print(f"  {status}: {ref_name}", flush=True)
                with lock:
                    _record(u, ref_name)

    print(f"manifest: {manifest_path} ({len(manifest['files'])} entries)")
    if failures:
        print(f"WARNING: {len(failures)} granule(s) failed; re-run to retry "
              "(completed references are skipped).")
    return manifest_path


def _report_chunking(url: str, refs: dict) -> None:
    """Print per-variable zarr chunk byte sizes (latency sanity check)."""
    try:
        arrays = refs.get("refs", {})
        seen = set()
        for key, val in arrays.items():
            if not key.endswith("/.zarray") or not isinstance(val, str):
                continue
            var = key.split("/")[0]
            if var in seen or var in ("time", "lat", "lon"):
                continue
            seen.add(var)
            meta = json.loads(val)
            shape = meta.get("shape", [])
            chunks = meta.get("chunks", [])
            itemsize = {"f4": 4, "f8": 8, "i4": 4}.get(meta.get("dtype", ""), 4)
            chunk_bytes = itemsize
            for c in chunks:
                chunk_bytes *= c
            nchunks = 1
            for s, c in zip(shape, chunks):
                nchunks *= -(-s // c)
            total_mb = chunk_bytes * nchunks / 1e6
            print(f"    {var}: chunks={chunks} ~{chunk_bytes/1e3:.0f} KB/chunk, "
                  f"{nchunks} chunks/day (~{total_mb:.1f} MB/day total)")
            if len(seen) >= 8:
                break
    except Exception as exc:  # never fail the build over a report
        print(f"    (chunk report skipped: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build kerchunk indexes for MERRA-2.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--shortname", default=DEFAULT_SHORTNAME,
                        help="GES DISC short name, e.g. M2T1NXAER")
    parser.add_argument("--out", default="../data/kerchunk",
                        help="Output directory for reference JSONs")
    parser.add_argument("--prefix", default=None,
                        help="Override MERRA2_S3_PREFIX, e.g. s3://bucket/MERRA2")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only index the first N files (for testing)")
    parser.add_argument("--start-date", default=None,
                        help="First date to index (YYYY-MM-DD). With --end-date, "
                             "builds file URLs directly instead of listing the "
                             "bucket (GES DISC denies s3:ListBucket).")
    parser.add_argument("--end-date", default=None,
                        help="Last date to index (YYYY-MM-DD).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Granules to index concurrently. One granule "
                             "takes ~5 min from outside AWS, and the work is "
                             "network-bound, so 4-8 cuts wall time roughly "
                             "linearly.")
    args = parser.parse_args()
    start = date.fromisoformat(args.start_date) if args.start_date else None
    end = date.fromisoformat(args.end_date) if args.end_date else None
    build_index(args.collection, Path(args.out), args.limit, args.prefix,
                args.shortname, start, end, args.workers)


if __name__ == "__main__":
    main()
