"""kerchunk index builder: one-time scan of MERRA-2 NetCDF on AWS Open Data.

MERRA-2 lives as public NetCDF-4 on S3. Downloading whole files per query
would kill latency; instead we build a JSON index of byte ranges ONCE with
kerchunk, then per-query xarray does lazy HTTP range reads of only the
chunks covering (variable, bbox, time).

Usage:
    python kerchunk_index.py --collection tavg1_2d_aer_Nx --out ../data/kerchunk/

Output: one JSON reference file per NetCDF file, plus an index.json
manifest mapping (collection, date) -> reference file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# TODO(integration): implement against the real AWS Open Data bucket.
# MERRA-2 on AWS is documented in us-west-2; verify the bucket/prefix and
# the internal chunking layout of the aerosol collections before committing
# to this path (awkward chunking inflates per-query byte reads).
#
# Sketch:
#   import fsspec, kerchunk.hdf, xarray as xr
#   fs = fsspec.filesystem("s3", anon=True)
#   for url in sorted(fs.glob("s3://<merra2-bucket>/<collection>/*.nc4")):
#       with fs.open(url, "rb") as f:
#           refs = kerchunk.hdf.SingleHdf5ToZarr(f, url).translate()
#       (out_dir / (Path(url).stem + ".json")).write_text(json.dumps(refs))
#
# At query time:
#   import kerchunk.combine  # or fsspec ReferenceFileSystem
#   mapper = fsspec.get_mapper("reference://", fo=index_json)
#   ds = xr.open_dataset(mapper, engine="zarr", chunks={})
#   da = ds["DUEXTTAU"].sel(time=slice(t0, t1), lat=slice(...), lon=slice(...))


def build_index(collection: str, out_dir: Path, limit: int | None = None) -> None:
    """Build kerchunk reference JSONs for one MERRA-2 collection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "TODO(integration): implement the S3 scan sketched in the module "
        f"docstring for collection={collection!r}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build kerchunk indexes for MERRA-2.")
    parser.add_argument("--collection", default="tavg1_2d_aer_Nx",
                        help="MERRA-2 collection (default: %(default)s)")
    parser.add_argument("--out", default="../data/kerchunk",
                        help="Output directory for reference JSONs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only index the first N files (for testing)")
    args = parser.parse_args()
    build_index(args.collection, Path(args.out), args.limit)


if __name__ == "__main__":
    main()
