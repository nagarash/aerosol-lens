"""Tests for the kerchunk grid slicer (backend/grid.py).

Builds tiny synthetic MERRA-2-like NetCDF files (hourly time, lat/lon
grid, *EXTTAU variables), indexes them with the REAL kerchunk code path
(SingleHdf5ToZarr + MultiZarrToZarr), and serves them through get_grid().
Expected values are computed by reading the same NetCDF files directly,
so the tests verify the kerchunk/reference path, not the math.

Geo deps required: xarray, fsspec, kerchunk, netCDF4, dask.
Run from the repo root with:
    python backend/test_grid.py
(pytest-compatible as well: every test_* function takes no arguments.)
"""

import json
import os
import shutil
import sys
import tempfile
import traceback
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import xarray as xr
except ImportError:
    np = None
    xr = None

from backend.grid import (
    BadGridRequestError,
    IndexNotBuiltError,
    UnknownVariableError,
    collapse_time,
    downsample,
    get_grid,
    to_grid_json,
)
from backend import grid as grid_module
from backend import kerchunk_index

DAYS = ("2026-09-01", "2026-09-02")
NLAT, NLON, NTIME = 19, 37, 24  # lats -90..90 step 10, lons -180..180 step 10


# ---------------------------------------------------------------------------
# Synthetic fixtures


def make_synthetic_nc(path, day):
    """One MERRA-2-like daily granule: hourly time, lat/lon grid, EXTTAU vars.

    Like real MERRA-2, time uses a FIXED origin ("hours since 1980-01-01")
    so raw time values are unique across granules — MultiZarrToZarr
    concatenates on raw values, so per-file origins would silently collapse.
    """
    import netCDF4
    import warnings
    from datetime import datetime

    origin = datetime(1980, 1, 1)
    base = datetime.fromisoformat(day)
    # hourly steps at 00:30, 01:30, ..., 23:30 UTC, as absolute hours
    hours = (base - origin).total_seconds() / 3600.0 + 0.5 + np.arange(NTIME)
    lats = np.arange(-90, 91, 10, dtype="f4")
    lons = np.arange(-180, 181, 10, dtype="f4")
    assert len(lats) == NLAT and len(lons) == NLON
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time", NTIME)
        ds.createDimension("lat", NLAT)
        ds.createDimension("lon", NLON)
        t = ds.createVariable("time", "f8", ("time",))
        t.units = "hours since 1980-01-01 00:00:00"
        t.calendar = "standard"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # netCDF4+numpy2.5
            t[:] = hours
        la = ds.createVariable("lat", "f4", ("lat",))
        la[:] = lats
        la.units = "degrees_north"
        lo = ds.createVariable("lon", "f4", ("lon",))
        lo[:] = lons
        lo.units = "degrees_east"
        tt, yy, xx = np.meshgrid(
            np.arange(NTIME), np.arange(NLAT), np.arange(NLON), indexing="ij"
        )
        pattern = 0.01 * tt + 0.001 * yy + 0.0001 * xx
        for name, offset in (("DUEXTTAU", 0.0), ("TOTEXTTAU", 0.5)):
            v = ds.createVariable(
                name, "f4", ("time", "lat", "lon"), chunksizes=(6, NLAT, NLON)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                v[:] = pattern + offset
            v.units = "1"


def make_reference(nc_path, ref_path):
    from kerchunk.hdf import SingleHdf5ToZarr

    with open(nc_path, "rb") as f:
        refs = SingleHdf5ToZarr(f, str(nc_path)).translate()
    with open(ref_path, "w") as f:
        json.dump(refs, f)


class grid_harness:
    """Two synthetic daily granules + kerchunk manifest in a temp dir."""

    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix="gridtest-")
        files = {}
        for day in DAYS:
            nc = os.path.join(self.tmp, f"synthetic.{day}.nc")
            ref = os.path.join(self.tmp, f"synthetic.{day}.json")
            make_synthetic_nc(nc, day)
            make_reference(nc, ref)
            files[day] = os.path.basename(ref)
        self.manifest = os.path.join(self.tmp, "index.json")
        with open(self.manifest, "w") as f:
            json.dump(
                {
                    "collection": "tavg1_2d_aer_Nx",
                    "prefix": "synthetic",
                    "files": files,
                },
                f,
            )
        self._saved = os.environ.get("KERCHUNK_INDEX_PATH")
        os.environ["KERCHUNK_INDEX_PATH"] = self.manifest
        grid_module._load_manifest.cache_clear()
        return self

    def __exit__(self, *exc):
        if self._saved is None:
            os.environ.pop("KERCHUNK_INDEX_PATH", None)
        else:
            os.environ["KERCHUNK_INDEX_PATH"] = self._saved
        grid_module._load_manifest.cache_clear()
        shutil.rmtree(self.tmp, ignore_errors=True)
        return False

    def direct(self):
        """The same granules opened directly (no kerchunk) for expectations."""
        parts = [
            xr.open_dataset(os.path.join(self.tmp, f"synthetic.{d}.nc"))
            for d in DAYS
        ]
        return xr.concat(parts, dim="time")


def require_geo():
    if np is None or xr is None:
        raise AssertionError(
            "geo deps (xarray/numpy) not installed; cannot run grid tests"
        )


# ---------------------------------------------------------------------------
# Tests


def test_end_to_end_daily_matches_direct_read():
    """kerchunk path returns the same grid as a direct NetCDF read."""
    require_geo()
    with grid_harness() as h:
        resp = get_grid(
            "merra2", "DUEXTTAU", "-20,-20,20,20",
            "2026-09-01T00:00:00Z", "2026-09-02T23:59:59Z", agg="daily",
        )
        expected = (
            h.direct()["DUEXTTAU"]
            .sel(lon=slice(-20, 20), lat=slice(-20, 20))
            .resample(time="1D").mean().mean("time")
        )
    assert resp["variable"] == "DUEXTTAU"
    assert resp["units"] == "dimensionless"
    assert resp["aggregation"] == "daily"
    assert resp["source"] == "merra2"
    assert resp["time_steps_used"] == 48, resp["time_steps_used"]
    assert resp["downsampled"] is False
    assert resp["nx"] == len(resp["lons"]) == 5   # -20..20 step 10
    assert resp["ny"] == len(resp["lats"]) == 5
    assert len(resp["values"]) == 5 and len(resp["values"][0]) == 5

    assert np.allclose(resp["lats"], expected["lat"].values, atol=1e-6)
    assert np.allclose(resp["lons"], expected["lon"].values, atol=1e-6)
    assert np.allclose(np.array(resp["values"]), expected.values, atol=1e-3), (
        "kerchunk read diverged from direct read"
    )


def test_alias_canonicalization():
    require_geo()
    with grid_harness():
        resp = get_grid(
            "merra2", "DUAOD", "-20,-20,20,20",
            "2026-09-01T00:00:00Z", "2026-09-01T23:59:59Z", agg="hourly",
        )
    assert resp["variable"] == "DUEXTTAU", "legacy alias must canonicalize"


def test_hourly_aggregation_is_straight_mean():
    require_geo()
    with grid_harness() as h:
        resp = get_grid(
            "merra2", "TOTEXTTAU", "0,0,10,10",
            "2026-09-01T00:00:00Z", "2026-09-01T23:59:59Z", agg="hourly",
        )
        expected = (
            h.direct()["TOTEXTTAU"]
            .sel(lon=slice(0, 10), lat=slice(0, 10),
                 time=slice("2026-09-01", "2026-09-01T23:59:59"))
            .mean("time")
        )
    assert resp["time_steps_used"] == 24
    assert np.allclose(np.array(resp["values"]), expected.values, atol=1e-3)


def test_antimeridian_bbox_wraps():
    require_geo()
    with grid_harness() as h:
        resp = get_grid(
            "merra2", "DUEXTTAU", "170,-20,-170,20",
            "2026-09-01T00:00:00Z", "2026-09-01T23:59:59Z", agg="hourly",
        )
        # [170..180] + [-180..-170]
        assert resp["lons"] == [170.0, 180.0, -180.0, -170.0], resp["lons"]
        assert resp["nx"] == 4
        direct = h.direct()["DUEXTTAU"]
        left = direct.sel(lon=slice(170, 180), lat=slice(-20, 20)).mean("time")
        right = direct.sel(lon=slice(-180, -170), lat=slice(-20, 20)).mean("time")
        expected = xr.concat([left, right], dim="lon")
    assert np.allclose(np.array(resp["values"]), expected.values, atol=1e-3)


def test_unknown_variable_is_422():
    require_geo()
    with grid_harness():
        try:
            get_grid("merra2", "FOO", "0,0,10,10",
                     "2026-09-01T00:00:00Z", "2026-09-01T23:59:59Z")
        except UnknownVariableError:
            return
    raise AssertionError("expected UnknownVariableError")


def test_bad_requests_are_422():
    require_geo()
    cases = [
        ("merra2", "DUEXTTAU", "not-a-bbox", "2026-09-01T00:00:00Z",
         "2026-09-01T23:59:59Z", "daily"),       # malformed bbox
        ("merra2", "DUEXTTAU", "0,10,10,5", "2026-09-01T00:00:00Z",
         "2026-09-01T23:59:59Z", "daily"),       # south >= north
        ("merra2", "DUEXTTAU", "0,0,10,10", "2026-09-02T00:00:00Z",
         "2026-09-01T00:00:00Z", "daily"),       # t1 < t0
        ("merra2", "DUEXTTAU", "0,0,10,10", "2026-09-01T00:00:00Z",
         "2026-09-01T23:59:59Z", "weekly"),      # bad aggregation
        ("cams", "DUEXTTAU", "0,0,10,10", "2026-09-01T00:00:00Z",
         "2026-09-01T23:59:59Z", "daily"),       # unsupported source
    ]
    with grid_harness():
        for args in cases:
            try:
                get_grid(*args)
            except BadGridRequestError:
                continue
            raise AssertionError(f"expected BadGridRequestError for {args}")


def test_no_coverage_window_is_422():
    require_geo()
    with grid_harness():
        try:
            get_grid("merra2", "DUEXTTAU", "0,0,10,10",
                     "2026-09-05T00:00:00Z", "2026-09-06T23:59:59Z")
        except BadGridRequestError as exc:
            assert "2026-09-05" in str(exc)
            return
    raise AssertionError("expected BadGridRequestError for uncovered window")


def test_missing_index_is_501_naming_env_var():
    require_geo()
    saved = os.environ.get("KERCHUNK_INDEX_PATH")
    os.environ["KERCHUNK_INDEX_PATH"] = "/nonexistent/kerchunk-index.json"
    grid_module._load_manifest.cache_clear()
    try:
        try:
            get_grid("merra2", "DUEXTTAU", "0,0,10,10",
                     "2026-09-01T00:00:00Z", "2026-09-01T23:59:59Z")
        except IndexNotBuiltError as exc:
            assert "KERCHUNK_INDEX_PATH" in str(exc), str(exc)
            return
        raise AssertionError("expected IndexNotBuiltError")
    finally:
        if saved is None:
            os.environ.pop("KERCHUNK_INDEX_PATH", None)
        else:
            os.environ["KERCHUNK_INDEX_PATH"] = saved
        grid_module._load_manifest.cache_clear()


def test_downsample_caps_resolution():
    require_geo()
    big = xr.DataArray(
        np.zeros((361, 576)),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(-90, 90, 361),
                "lon": np.linspace(-180, 180, 576)},
    )
    out, flag = downsample(big)
    assert flag is True
    assert out.sizes["lat"] <= 180 and out.sizes["lon"] <= 360, dict(out.sizes)
    small = xr.DataArray(
        np.zeros((19, 37)),
        dims=("lat", "lon"),
        coords={"lat": np.linspace(-90, 90, 19),
                "lon": np.linspace(-180, 180, 37)},
    )
    out2, flag2 = downsample(small)
    assert flag2 is False and out2.sizes == small.sizes


def test_collapse_time_semantics():
    require_geo()
    times = xr.cftime_range("2026-09-01", periods=48, freq="h")
    da = xr.DataArray(
        np.arange(48, dtype=float).reshape(48, 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [0.0], "lon": [0.0]},
    )
    hourly = collapse_time(da, "hourly")
    assert float(hourly.values[0, 0]) == np.mean(np.arange(48))
    daily = collapse_time(da, "daily")
    assert daily.sizes == {"lat": 1, "lon": 1}
    assert float(daily.values[0, 0]) == np.mean(np.arange(48))
    monthly = collapse_time(da, "monthly_mean")
    assert float(monthly.values[0, 0]) == np.mean(np.arange(48))


def test_to_grid_json_shape_and_rounding():
    require_geo()
    da = xr.DataArray(
        np.array([[0.123456, 1.0], [2.0, 3.99999]]),
        dims=("lat", "lon"),
        coords={"lat": [10.0, 20.0], "lon": [30.0, 40.0]},
    )
    from datetime import datetime, timezone

    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 2, tzinfo=timezone.utc)
    payload = to_grid_json(da, "DUEXTTAU", t0, t1, "daily", False, 48)
    assert payload["values"] == [[0.1235, 1.0], [2.0, 4.0]]
    assert payload["lats"] == [10.0, 20.0] and payload["lons"] == [30.0, 40.0]
    assert payload["nx"] == 2 and payload["ny"] == 2


def test_to_grid_json_sorts_descending_lat():
    require_geo()
    da = xr.DataArray(
        np.array([[1.0], [2.0]]),
        dims=("lat", "lon"),
        coords={"lat": [20.0, 10.0], "lon": [0.0]},  # descending
    )
    from datetime import datetime, timezone

    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    payload = to_grid_json(da, "DUEXTTAU", t0, t0, "hourly", False, 1)
    assert payload["lats"] == [10.0, 20.0]
    assert payload["values"] == [[2.0], [1.0]], "values must follow the lat sort"


def test_kerchunk_index_helpers():
    assert kerchunk_index.date_of_filename(
        "s3://bucket/MERRA2_400.tavg1_2d_aer_Nx.20240115.nc4"
    ) == date(2024, 1, 15)
    assert kerchunk_index.date_of_filename("nope.nc") is None
    assert kerchunk_index._collection_prefix(
        "s3://bucket/MERRA2", "M2T1NXAER"
    ) == "s3://bucket/MERRA2/M2T1NXAER.5.12.4"


# ---------------------------------------------------------------------------
# Runner (works without pytest)


def main():
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
