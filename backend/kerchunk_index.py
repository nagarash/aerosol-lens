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

AUTHENTICATION (important): the bucket is *protected* — anonymous S3
reads are rejected. Access requires an Earthdata Login account. Set
EARTHDATA_TOKEN to a long-lived bearer token (generate one in the
Earthdata Login profile; server-side only, never in the repo): the
builder exchanges it for temporary AWS session credentials
automatically and refreshes them as needed. Alternatively export
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN directly
(they last ~1h; re-export before each build). Set MERRA2_S3_ANON=1 only
for a genuinely public mirror.

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
from datetime import date, timedelta
from pathlib import Path

try:  # imported as part of the backend package
    from backend.earthdata_auth import s3_target_options
except ImportError:  # run as a script: python backend/kerchunk_index.py
    from earthdata_auth import s3_target_options

DEFAULT_COLLECTION = "tavg1_2d_aer_Nx"
DEFAULT_SHORTNAME = "M2T1NXAER"
# Verified 2026-09-12 via registry.opendata.aws/nasa-merra-2 and GES DISC
# tutorials. NOTE: this bucket is *protected* — Earthdata Login temporary
# credentials are required (see module docstring); anonymous reads fail.
DEFAULT_PREFIX = "s3://gesdisc-cumulus-prod-protected/MERRA2"

FILENAME_DATE_RE = re.compile(r"\.(\d{8})\.nc4$")


def _s3_prefix(prefix: str | None) -> str:
    return (prefix or os.environ.get("MERRA2_S3_PREFIX") or DEFAULT_PREFIX).rstrip("/")


def _s3_options() -> dict:
    """s3fs options for the MERRA-2 bucket.

    The builder always reads S3, so it resolves strictly via
    backend.earthdata_auth: EARTHDATA_TOKEN exchange, the standard AWS
    credential chain (AWS_* env vars, ~/.aws, IAM role), or
    MERRA2_S3_ANON=1 for a genuinely public mirror. Missing credentials
    raise an honest error naming EARTHDATA_TOKEN instead of failing
    deep inside fsspec.
    """
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

    fs = fsspec.filesystem("s3", **_s3_options())
    pattern = f"{collection_prefix}/**/*.nc4"
    urls = sorted(fs.glob(pattern))
    if limit is not None:
        urls = urls[:limit]
    return [u if u.startswith("s3://") else f"s3://{u}" for u in urls]


def reference_for_url(url: str) -> dict:
    """Build a kerchunk reference dict for one remote NetCDF file."""
    import fsspec
    from kerchunk.hdf import SingleHdf5ToZarr

    with fsspec.open(url, "rb", **_s3_options()) as f:
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
    if not urls:
        raise RuntimeError(
            f"no .nc4 files found under {base!r}. The GES DISC bucket is "
            "protected: export temporary Earthdata Login credentials "
            "(AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY/AWS_SESSION_TOKEN from "
            "https://data.gesdisc.earthdata.nasa.gov/s3credentials) or check "
            "MERRA2_S3_PREFIX / S3 reachability."
        )
    print(f"indexing {len(urls)} file(s) from {base}")

    for url in urls:
        ref_name = Path(url).stem + ".json"
        ref_path = out_dir / ref_name
        if ref_path.exists():
            print(f"  skip (exists): {ref_name}")
        else:
            print(f"  indexing: {url}")
            refs = reference_for_url(url)
            ref_path.write_text(json.dumps(refs))
            _report_chunking(url, refs)
        d = date_of_filename(url)
        manifest["files"][d.isoformat() if d else url] = ref_name
        manifest_path.write_text(json.dumps(manifest, indent=1))

    print(f"manifest: {manifest_path} ({len(manifest['files'])} entries)")
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
    args = parser.parse_args()
    start = date.fromisoformat(args.start_date) if args.start_date else None
    end = date.fromisoformat(args.end_date) if args.end_date else None
    build_index(args.collection, Path(args.out), args.limit, args.prefix,
                args.shortname, start, end)


if __name__ == "__main__":
    main()
