"""Tests for the pre-aggregated daily field store and its /grid fast path.

Covers backend/fields.py (append/read/coverage/bbox slicing, all on
synthetic data -- no network), the get_grid() routing between the
fields fast path and the kerchunk path, the backfill date-range math
and idempotency (mocked granule reader), and the admin backfill
endpoints (auth + 409/404 behavior).

Run from the repo root with:
    python backend/test_fields.py
(pytest-compatible as well: every test_* function takes no arguments.)
"""

import os
import shutil
import sys
import tempfile
import traceback
from contextlib import contextmanager
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from agent.mappings import MERRA2_COLUMN_VARIABLES
from backend import backfill, fields
from backend.fields import FieldsError


# --------------------------------------------------------------------------
# Helpers


@contextmanager
def temp_env(**overrides):
    """Temporarily set env vars (None = delete), restoring afterwards."""
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextmanager
def temp_store():
    """A fresh FIELDS_DIR + GRID_CACHE_DIR pair, both in a temp dir."""
    d = tempfile.mkdtemp(prefix="fields-test-")
    try:
        with temp_env(FIELDS_DIR=os.path.join(d, "fields"),
                      GRID_CACHE_DIR=os.path.join(d, "cache"),
                      MERRA2_LATENCY_DAYS="0"):
            yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def synthetic_day(seed: int) -> np.ndarray:
    """Deterministic (361, 576) float32 field with no NaNs."""
    rng = np.random.default_rng(seed)
    base = rng.random((fields.NLAT, fields.NLON), dtype=np.float32)
    # A smooth gradient so bbox slices are distinguishable from noise.
    latw = np.linspace(0, 1, fields.NLAT, dtype=np.float32)[:, None]
    lonw = np.linspace(0, 1, fields.NLON, dtype=np.float32)[None, :]
    return (base * 0.5 + latw * 0.3 + lonw * 0.2).astype(np.float32)


def fill_store(var="DUEXTTAU", dates=("2026-08-01", "2026-08-02", "2026-08-03")):
    for i, ds in enumerate(dates):
        assert fields.append_day(var, ds, synthetic_day(1000 + i)) is True
    return list(dates)


# --------------------------------------------------------------------------
# fields.py: append / coverage / idempotency / validation


def test_append_and_coverage():
    with temp_store():
        assert fields.coverage("DUEXTTAU") == set()
        assert fields.append_day("DUEXTTAU", "2026-08-01", synthetic_day(1))
        assert fields.coverage("DUEXTTAU") == {"2026-08-01"}
        assert fields.append_day("DUEXTTAU", "2026-08-03", synthetic_day(3))
        assert fields.coverage("DUEXTTAU") == {"2026-08-01", "2026-08-03"}


def test_append_idempotent():
    with temp_store():
        assert fields.append_day("DUEXTTAU", "2026-08-01", synthetic_day(1)) is True
        # Second append of the same date: no write, returns False.
        assert fields.append_day("DUEXTTAU", "2026-08-01", synthetic_day(999)) is False
        assert fields.coverage("DUEXTTAU") == {"2026-08-01"}
        data, _, _ = fields.read("DUEXTTAU", ["2026-08-01"], (-180, -90, 180, 90))
        np.testing.assert_array_equal(data[0], synthetic_day(1))


def test_append_out_of_order_stays_sorted():
    with temp_store():
        fields.append_day("DUEXTTAU", "2026-08-03", synthetic_day(3))
        fields.append_day("DUEXTTAU", "2026-08-01", synthetic_day(1))
        fields.append_day("DUEXTTAU", "2026-08-02", synthetic_day(2))
        data, _, _ = fields.read(
            "DUEXTTAU",
            ["2026-08-01", "2026-08-02", "2026-08-03"],
            (-180, -90, 180, 90),
        )
        np.testing.assert_array_equal(data[0], synthetic_day(1))
        np.testing.assert_array_equal(data[1], synthetic_day(2))
        np.testing.assert_array_equal(data[2], synthetic_day(3))


def test_append_validates_shape_and_date():
    with temp_store():
        try:
            fields.append_day("DUEXTTAU", "2026-08-01", np.zeros((10, 10)))
        except FieldsError:
            pass
        else:
            raise AssertionError("expected FieldsError on bad shape")
        try:
            fields.append_day("DUEXTTAU", "not-a-date", synthetic_day(1))
        except FieldsError:
            pass
        else:
            raise AssertionError("expected FieldsError on bad date")
        assert fields.coverage("DUEXTTAU") == set()


def test_provenance_attrs():
    with temp_store():
        fields.append_day("DUEXTTAU", "2026-08-01", synthetic_day(1))
        import zarr

        arr = zarr.open_array(
            str(fields.fields_dir() / "DUEXTTAU.zarr"), mode="r")
        assert arr.attrs["source_collection"] == "M2T1NXAER/tavg1_2d_aer_Nx"
        assert arr.attrs["aggregation"].startswith("daily mean")
        assert arr.attrs["dates"] == ["2026-08-01"]
        assert arr.dtype == np.float32
        assert arr.chunks == (1, 361, 576)


def test_read_missing_date_raises():
    with temp_store():
        fill_store()
        try:
            fields.read("DUEXTTAU", ["2026-08-01", "2026-08-09"],
                        (-180, -90, 180, 90))
        except FieldsError as exc:
            assert "2026-08-09" in str(exc)
        else:
            raise AssertionError("expected FieldsError on missing date")


def test_read_missing_variable_raises():
    with temp_store():
        try:
            fields.read("NOPE", ["2026-08-01"], (-180, -90, 180, 90))
        except FieldsError:
            pass
        else:
            raise AssertionError("expected FieldsError on missing variable")


# --------------------------------------------------------------------------
# fields.py: bbox slicing matches grid.py's conventions exactly


def _expected_indices(w, s, e, n):
    """Brute-force reference: inclusive label slicing on the fixed grid."""
    lats = -90.0 + 0.5 * np.arange(361)
    lons = -180.0 + 0.625 * np.arange(576)
    eps = 1e-9
    lat_sel = np.where((lats >= s - eps) & (lats <= n + eps))[0]
    if w <= e:
        lon_sel = np.where((lons >= w - eps) & (lons <= e + eps))[0]
    else:
        left = np.where((lons >= w - eps) & (lons <= 179.375 + eps))[0]
        right = np.where((lons >= -180 - eps) & (lons <= e + eps))[0]
        lon_sel = np.concatenate([left, right])
    return lat_sel, lon_sel, lats[lat_sel], lons[lon_sel]


def test_read_bbox_matches_brute_force():
    with temp_store():
        dates = fill_store()
        bbox = (-10.5, 20.25, 30.0, 60.75)  # deliberately off-grid bounds
        data, lats, lons = fields.read("DUEXTTAU", dates, bbox)
        lat_sel, lon_sel, exp_lats, exp_lons = _expected_indices(*bbox)
        assert data.shape == (3, len(lat_sel), len(lon_sel))
        np.testing.assert_allclose(lats, exp_lats)
        np.testing.assert_allclose(lons, exp_lons)
        # Lats ascending in the response (to_grid_json convention).
        assert all(b >= a for a, b in zip(lats, lats[1:]))
        for i, ds in enumerate(dates):
            np.testing.assert_array_equal(
                data[i], synthetic_day(1000 + dates.index(ds))[lat_sel][:, lon_sel])


def test_read_antimeridian_wrap():
    with temp_store():
        dates = fill_store()
        bbox = (170.0, -5.0, -170.0, 10.0)
        data, lats, lons = fields.read("DUEXTTAU", dates, bbox)
        _, _, _, exp_lons = _expected_indices(*bbox)
        np.testing.assert_allclose(lons, exp_lons)
        # Wrap order: eastern piece first, then the western piece.
        assert lons[0] >= 170.0 and lons[-1] <= -170.0
        assert any(x < 0 for x in lons) and any(x > 0 for x in lons)


def test_read_matches_grid_slice_helpers():
    """fields.read must agree cell-for-cell with grid._slice_lon/_slice_lat,
    the helpers the kerchunk path uses."""
    import xarray as xr

    from backend import grid as grid_mod

    with temp_store():
        dates = fill_store()
        raw = synthetic_day(1000)
        lats = -90.0 + 0.5 * np.arange(361)
        lons = -180.0 + 0.625 * np.arange(576)
        da = xr.DataArray(raw, dims=("lat", "lon"),
                          coords={"lat": lats, "lon": lons})
        for bbox in [(-10.5, 20.25, 30.0, 60.75),
                     (170.0, -5.0, -170.0, 10.0),
                     (-180.0, -90.0, 180.0, 90.0)]:
            w, s, e, n = bbox
            ref = grid_mod._slice_lat(grid_mod._slice_lon(da, w, e), s, n)
            # to_grid_json flips descending lats to ascending; the store
            # is ascending natively, so compare against the ascending form.
            if ref["lat"].values[0] > ref["lat"].values[-1]:
                ref = ref.isel(lat=slice(None, None, -1))
            data, rlats, rlons = fields.read("DUEXTTAU", [dates[0]], bbox)
            np.testing.assert_allclose(rlats, ref["lat"].values)
            np.testing.assert_allclose(rlons, ref["lon"].values)
            np.testing.assert_allclose(data[0], ref.values, rtol=1e-6)


def test_read_empty_bbox_raises():
    with temp_store():
        fill_store()
        for bbox in [(10.0, 20.0, 10.0, 30.0)]:  # w == e selects nothing
            try:
                fields.read("DUEXTTAU", ["2026-08-01"], bbox)
            except FieldsError:
                pass
            else:
                raise AssertionError(f"expected FieldsError for {bbox}")


# --------------------------------------------------------------------------
# grid.py routing: fields fast path vs kerchunk path


def _grid_env(**extra):
    return temp_env(KERCHUNK_INDEX_PATH="/nonexistent/index.json",
                    MERRA2_LATENCY_DAYS="0", **extra)


def test_get_grid_fields_fast_path():
    from backend import grid as grid_mod

    with temp_store(), _grid_env():
        dates = fill_store()
        # KERCHUNK_INDEX_PATH points nowhere: if the fast path works,
        # the manifest is never touched.
        res = grid_mod.get_grid(
            source="merra2", variable="DUEXTTAU",
            bbox="-10.5,20.25,30.0,60.75",
            t0="2026-08-01", t1="2026-08-03", agg="daily")
        assert res["data_source"] == "fields"
        assert res["cache_hit"] is False
        assert res["source"] == "merra2"
        assert res["aggregation"] == "daily"
        # Values == mean of the three daily means over the bbox cells.
        lat_sel, lon_sel, exp_lats, exp_lons = _expected_indices(
            -10.5, 20.25, 30.0, 60.75)
        expected = np.nanmean(
            np.stack([synthetic_day(1000 + i)[lat_sel][:, lon_sel]
                      for i in range(3)], axis=0), axis=0)
        got = np.array([[v if v is not None else np.nan for v in row]
                        for row in res["values"]])
        np.testing.assert_allclose(got, np.round(expected, 4), atol=1e-3)
        np.testing.assert_allclose(res["lats"], np.round(exp_lats, 4))
        np.testing.assert_allclose(res["lons"], np.round(exp_lons, 4))
        assert res["time_steps_used"] == 72  # 24 hourly steps x 3 days


def test_get_grid_fields_monthly_mean():
    from backend import grid as grid_mod

    with temp_store(), _grid_env():
        # Span two months: 2026-07-30..2026-08-02.
        for i, ds in enumerate(["2026-07-30", "2026-07-31",
                                "2026-08-01", "2026-08-02"]):
            fields.append_day("DUEXTTAU", ds, synthetic_day(2000 + i))
        res = grid_mod.get_grid(
            source="merra2", variable="DUEXTTAU",
            bbox="-10,20,30,60", t0="2026-07-30", t1="2026-08-02",
            agg="monthly_mean")
        assert res["data_source"] == "fields"
        lat_sel, lon_sel, _, _ = _expected_indices(-10, 20, 30, 60)
        days = [synthetic_day(2000 + i)[lat_sel][:, lon_sel] for i in range(4)]
        # monthly_mean = mean of monthly means (equal weight per month).
        jul = np.nanmean(np.stack(days[:2]), axis=0)
        aug = np.nanmean(np.stack(days[2:]), axis=0)
        expected = np.nanmean(np.stack([jul, aug]), axis=0)
        got = np.array([[v if v is not None else np.nan for v in row]
                        for row in res["values"]])
        np.testing.assert_allclose(got, np.round(expected, 4), atol=1e-3)


def test_get_grid_fields_cache_roundtrip():
    from backend import grid as grid_mod

    with temp_store(), _grid_env():
        fill_store()
        kw = dict(source="merra2", variable="DUEXTTAU", bbox="-10,20,30,60",
                  t0="2026-08-01", t1="2026-08-02", agg="daily")
        first = grid_mod.get_grid(**kw)
        assert first["cache_hit"] is False
        second = grid_mod.get_grid(**kw)
        assert second["cache_hit"] is True
        assert second["data_source"] == "fields"
        assert second["values"] == first["values"]


def test_get_grid_fields_downsamples_globe():
    from backend import grid as grid_mod

    with temp_store(), _grid_env():
        fill_store()
        res = grid_mod.get_grid(
            source="merra2", variable="DUEXTTAU",
            bbox="-180,-90,180,90",
            t0="2026-08-01", t1="2026-08-01", agg="daily")
        assert res["data_source"] == "fields"
        assert res["downsampled"] is True
        assert res["nx"] <= 360 and res["ny"] <= 180


def test_get_grid_falls_back_without_fields():
    from backend import grid as grid_mod

    with temp_store(), _grid_env():
        # Empty store + nonexistent manifest -> kerchunk path raises
        # IndexNotBuiltError (not a fields error).
        try:
            grid_mod.get_grid(
                source="merra2", variable="DUEXTTAU",
                bbox="-10,20,30,60",
                t0="2026-08-01", t1="2026-08-02", agg="daily")
        except grid_mod.IndexNotBuiltError:
            pass
        else:
            raise AssertionError("expected IndexNotBuiltError")


def test_get_grid_partial_coverage_falls_back():
    from backend import grid as grid_mod

    with temp_store(), _grid_env():
        # Store covers 08-01..08-02 but the window asks for 08-03 too.
        fill_store(dates=("2026-08-01", "2026-08-02"))
        try:
            grid_mod.get_grid(
                source="merra2", variable="DUEXTTAU",
                bbox="-10,20,30,60",
                t0="2026-08-01", t1="2026-08-03", agg="daily")
        except grid_mod.IndexNotBuiltError:
            pass  # kerchunk path, manifest missing -> honest 501-class error
        else:
            raise AssertionError("expected IndexNotBuiltError")


def test_fields_response_shape_matches_kerchunk_shape():
    """The two paths must return the same response keys."""
    from backend import grid as grid_mod

    expected_keys = {"variable", "units", "lats", "lons", "values", "nx",
                     "ny", "time_start", "time_end", "time_steps_used",
                     "aggregation", "downsampled", "source", "cache_hit",
                     "data_source"}
    with temp_store(), _grid_env():
        fill_store()
        res = grid_mod.get_grid(
            source="merra2", variable="DUEXTTAU",
            bbox="-10,20,30,60",
            t0="2026-08-01", t1="2026-08-02", agg="daily")
        assert set(res.keys()) == expected_keys


# --------------------------------------------------------------------------
# backfill: date-range math


def test_default_range_is_one_year_to_archive_edge():
    with temp_env(MERRA2_LATENCY_DAYS="7", BACKFILL_START=None,
                  BACKFILL_END=None):
        start, end = backfill.default_range()
        assert end == date.today() - timedelta(days=7)
        assert start == end - timedelta(days=364)
        assert (end - start).days == 364


def test_resolve_range_env_overrides():
    with temp_env(BACKFILL_START="2025-01-01", BACKFILL_END="2025-01-10"):
        s, e = backfill.resolve_range()
        assert (s, e) == (date(2025, 1, 1), date(2025, 1, 10))


def test_resolve_range_explicit_args_win():
    with temp_env(BACKFILL_START="2025-01-01", BACKFILL_END="2025-01-10"):
        s, e = backfill.resolve_range("2025-02-01", "2025-02-03")
        assert (s, e) == (date(2025, 2, 1), date(2025, 2, 3))


def test_resolve_range_rejects_inverted():
    with temp_env(BACKFILL_START=None, BACKFILL_END=None):
        try:
            backfill.resolve_range("2025-02-03", "2025-02-01")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


def test_granule_url_shape():
    url = backfill.granule_url(date(2026, 8, 1))
    assert url == ("https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/"
                   "M2T1NXAER.5.12.4/2026/08/"
                   "MERRA2_400.tavg1_2d_aer_Nx.20260801.nc4")


# --------------------------------------------------------------------------
# backfill: idempotency + error handling (mocked reader, no network)


def _fake_reader_factory(calls, fail_on=()):
    def reader(d):
        calls.append(d.isoformat())
        if d.isoformat() in fail_on:
            raise RuntimeError("simulated granule failure")
        seed = abs(hash((d.isoformat(), "seed"))) % (2 ** 31)
        return {v: synthetic_day((seed + i) % 997)
                for i, v in enumerate(sorted(MERRA2_COLUMN_VARIABLES))}

    return reader


def test_backfill_idempotent_rerun():
    with temp_store(), temp_env(BACKFILL_START=None, BACKFILL_END=None):
        calls: list = []
        reader = _fake_reader_factory(calls)
        s1 = backfill.backfill_range("2026-08-01", "2026-08-03",
                                     reader=reader)
        assert s1["done_days"] == 3 and not s1["errors"]
        assert len(calls) == 3
        for v in MERRA2_COLUMN_VARIABLES:
            assert fields.coverage(v) == {"2026-08-01", "2026-08-02",
                                          "2026-08-03"}
        # Second run: everything already stored, reader never called.
        calls.clear()
        s2 = backfill.backfill_range("2026-08-01", "2026-08-03",
                                     reader=reader)
        assert calls == []
        assert s2["done_days"] == 3 and not s2["errors"]


def test_backfill_partial_existing_skips_only_those():
    with temp_store():
        calls: list = []
        reader = _fake_reader_factory(calls)
        # Pre-store one variable for 08-02 only.
        fields.append_day("DUEXTTAU", "2026-08-02", synthetic_day(7))
        backfill.backfill_range("2026-08-01", "2026-08-02", reader=reader)
        # Both days needed a read (08-02 still missed 5 variables).
        assert calls == ["2026-08-01", "2026-08-02"]


def test_backfill_records_errors_and_continues():
    with temp_store():
        calls: list = []
        reader = _fake_reader_factory(calls, fail_on={"2026-08-02"})
        status = backfill.backfill_range("2026-08-01", "2026-08-03",
                                         reader=reader)
        assert status["done_days"] == 3
        assert len(status["errors"]) == 1
        assert status["errors"][0]["date"] == "2026-08-02"
        assert "simulated granule failure" in status["errors"][0]["error"]
        assert not backfill.is_running()


def test_backfill_status_shape():
    st = backfill.get_status()
    assert set(st.keys()) == {"running", "total_days", "done_days",
                              "current_date", "errors"}
    assert st["running"] is False


# --------------------------------------------------------------------------
# admin endpoints


def _admin_client():
    from fastapi.testclient import TestClient

    import backend.app as app_module

    return TestClient(app_module.app)


def test_admin_404_when_token_unset():
    with temp_env(ADMIN_TOKEN=None):
        r = _admin_client().post("/admin/backfill", json={})
        assert r.status_code == 404
        r = _admin_client().get("/admin/backfill/status")
        assert r.status_code == 404


def test_admin_403_on_wrong_token():
    with temp_env(ADMIN_TOKEN="secret123"):
        r = _admin_client().post(
            "/admin/backfill", json={},
            headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 403


def test_admin_starts_backfill_and_409_while_running():
    import backend.app  # noqa: F401  (ensures backend.backfill imported)
    from backend import backfill as bf

    started = {}

    def fake_range(start=None, end=None, reader=None, **kwargs):
        started["args"] = (start, end)
        started["kwargs"] = kwargs

    orig_range, orig_running = bf.backfill_range, bf.is_running
    bf.backfill_range = fake_range
    try:
        with temp_env(ADMIN_TOKEN="secret123"):
            client = _admin_client()
            hdr = {"Authorization": "Bearer secret123"}
            r = client.post("/admin/backfill",
                            json={"start": "2026-08-01", "end": "2026-08-02"},
                            headers=hdr)
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["started"] is True
            assert body["start"] == "2026-08-01"
            assert body["end"] == "2026-08-02"
            assert body["total_days"] == 2
            # The endpoint must not block waiting for the thread; poll
            # briefly for the background thread to run the fake.
            import time as _time

            for _ in range(100):
                if "args" in started:
                    break
                _time.sleep(0.02)
            assert started["args"][0].isoformat() == "2026-08-01"
            # While a backfill is flagged running -> 409.
            bf.is_running = lambda: True
            r = client.post("/admin/backfill", json={}, headers=hdr)
            assert r.status_code == 409
            # Status endpoint mirrors backfill.get_status().
            bf.is_running = lambda: False
            r = client.get("/admin/backfill/status", headers=hdr)
            assert r.status_code == 200
            assert set(r.json().keys()) == {"running", "total_days",
                                             "done_days", "current_date",
                                             "errors"}
    finally:
        bf.backfill_range = orig_range
        bf.is_running = orig_running


# --------------------------------------------------------------------------
# Runner (mirrors the other test modules: direct execution + pytest)


_TESTS = [v for k, v in sorted(globals().items())
          if k.startswith("test_") and callable(v)]


def main() -> int:
    failures = 0
    for fn in _TESTS:
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
        else:
            print(f"ok   {fn.__name__}")
    print(f"{len(_TESTS) - failures}/{len(_TESTS)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())


def test_backfill_parallel_workers_match_sequential():
    from backend import backfill
    with temp_store():
        calls: list = []
        reader = _fake_reader_factory(calls)
        status = backfill.backfill_range("2026-08-01", "2026-08-04",
                                         reader=reader, workers=3)
        assert status["done_days"] == 4 and not status["errors"]
        assert sorted(calls) == ["2026-08-01", "2026-08-02",
                                 "2026-08-03", "2026-08-04"]
        for v in MERRA2_COLUMN_VARIABLES:
            assert fields.coverage(v) == {"2026-08-01", "2026-08-02",
                                          "2026-08-03", "2026-08-04"}
        assert not backfill.is_running()


def test_backfill_parallel_records_errors_and_continues():
    from backend import backfill
    with temp_store():
        calls: list = []
        reader = _fake_reader_factory(calls, fail_on={"2026-08-02"})
        status = backfill.backfill_range("2026-08-01", "2026-08-03",
                                         reader=reader, workers=2)
        assert status["done_days"] == 3
        assert len(status["errors"]) == 1
        assert status["errors"][0]["date"] == "2026-08-02"


def test_backfill_retry_then_gives_up_on_permanent_failure():
    from backend import backfill
    attempts: list = []
    def reader(d):
        attempts.append(d.isoformat())
        raise FileNotFoundError(f"no granule for {d}")  # permanent: no retry
    with temp_store():
        status = backfill.backfill_range("2026-08-01", "2026-08-01",
                                         reader=reader, workers=1)
        assert attempts == ["2026-08-01"]  # failed fast, no retries
        assert len(status["errors"]) == 1


def test_backfill_retries_transient_failure():
    import unittest.mock
    from backend import backfill
    attempts: list = []
    def reader(d):
        attempts.append(d.isoformat())
        if len(attempts) < 3:
            raise TimeoutError("connection timed out")  # transient
        return {v: synthetic_day(1) for v in sorted(MERRA2_COLUMN_VARIABLES)}
    with temp_store(), unittest.mock.patch("time.sleep"):
        status = backfill.backfill_range("2026-08-01", "2026-08-01",
                                         reader=reader, workers=1)
        assert len(attempts) == 3
        assert not status["errors"]
        assert fields.coverage("DUEXTTAU") == {"2026-08-01"}
