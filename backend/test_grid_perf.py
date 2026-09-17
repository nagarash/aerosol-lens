"""Tests for the /grid latency work: disk response cache + parallel fetch.

No geo deps required: grid_cache and grid_fetch are stdlib-only, and the
backend.grid import is satisfied with a stubbed agent.mappings (pydantic
is not installed in this environment). The full kerchunk path is covered
by backend/test_grid.py, which needs xarray/fsspec/kerchunk.

Run from the repo root with:
    python backend/test_grid_perf.py
(pytest-compatible as well: every test_* function takes no arguments.)
"""

import os
import shutil
import sys
import tempfile
import threading
import time
import types
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub_agent_mappings():
    """grid.py needs agent.mappings at import; stub it (no pydantic here)."""
    if "agent.mappings" in sys.modules:
        return
    pkg = types.ModuleType("agent")
    pkg.__path__ = []  # type: ignore[attr-defined]
    mod = types.ModuleType("agent.mappings")
    mod.MERRA2_COLUMN_VARIABLES = ("DUEXTTAU", "SSEXTTAU")
    mod.canonical_variable = lambda source, variable: variable.upper()
    sys.modules["agent"] = pkg
    sys.modules["agent.mappings"] = mod


_stub_agent_mappings()

from backend import grid_cache
from backend import grid_fetch


# ---------------------------------------------------------------------------
# helpers


class _EnvGuard:
    """Set env vars for a test, restore afterwards."""

    def __init__(self, **vars):
        self.vars = vars
        self.saved = {}

    def __enter__(self):
        for k, v in self.vars.items():
            self.saved[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k in self.vars:
            if self.saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = self.saved[k]
        return False


def _fresh_cache_dir():
    tmp = tempfile.mkdtemp(prefix="grid_cache_test_")
    with _EnvGuard(GRID_CACHE_DIR=tmp):
        d = grid_cache.cache_dir()
    assert d is not None
    return tmp, d


class FakeDA:
    """Stand-in for a lazy xarray DataArray: compute() sleeps, then returns."""

    _threads = []
    _lock = threading.Lock()

    def __init__(self, value, delay=0.0, fail=False, ntime=24):
        self._value = value
        self._delay = delay
        self._fail = fail
        self.sizes = {"time": ntime}

    def compute(self):
        if self._delay:
            time.sleep(self._delay)
        if self._fail:
            raise IOError("granule read failed")
        with FakeDA._lock:
            FakeDA._threads.append(threading.get_ident())
        return self._value

    @classmethod
    def reset_threads(cls):
        with cls._lock:
            cls._threads = []

    @classmethod
    def thread_count(cls):
        with cls._lock:
            return len(set(cls._threads))


# ---------------------------------------------------------------------------
# grid_cache: keys


def test_cache_key_stable_across_equivalent_inputs():
    s = datetime(2026, 7, 29, tzinfo=timezone.utc)
    e = datetime(2026, 7, 29, 23, 59, 59, tzinfo=timezone.utc)
    k1 = grid_cache.cache_key("merra2", "DUEXTTAU", "daily",
                              (-17.0, 15.0, 35.0, 32.0), s, e)
    k2 = grid_cache.cache_key("MERRA2", "duexttau", "Daily",
                              (-17, 15, 35, 32), s, e)
    assert k1 == k2, "equivalent queries must share a key"
    assert len(k1) == 32 and all(c in "0123456789abcdef" for c in k1)
    k3 = grid_cache.cache_key("merra2", "SSEXTTAU", "daily",
                              (-17, 15, 35, 32), s, e)
    assert k3 != k1, "different variable must key differently"
    k4 = grid_cache.cache_key("merra2", "DUEXTTAU", "daily",
                              (-17, 15, 35, 32),
                              s, e + timedelta(days=1))
    assert k4 != k1, "different window must key differently"


def test_normalize_bbox_snaps_to_2dp():
    assert grid_cache.normalize_bbox((1.23456, -2.34567, 3.45678, -4.56789)) == \
        (1.23, -2.35, 3.46, -4.57)


# ---------------------------------------------------------------------------
# grid_cache: read/write


def test_cache_write_read_roundtrip():
    tmp, d = _fresh_cache_dir()
    try:
        result = {"variable": "DUEXTTAU", "values": [[0.5, 0.6]], "nx": 2}
        key = grid_cache.cache_key("merra2", "DUEXTTAU", "daily",
                                   (0, 0, 10, 10),
                                   datetime(2026, 7, 29),
                                   datetime(2026, 7, 29, 23, 59))
        assert grid_cache.write(d, key, result) is True
        assert grid_cache.read(d, key, None) == result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_read_missing_is_miss():
    tmp, d = _fresh_cache_dir()
    try:
        assert grid_cache.read(d, "no-such-key", None) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_read_expired_ttl_is_miss():
    tmp, d = _fresh_cache_dir()
    try:
        result = {"variable": "DUEXTTAU"}
        assert grid_cache.write(d, "k", result) is True
        assert grid_cache.read(d, "k", 0) is None, "ttl=0 must always expire"
        assert grid_cache.read(d, "k", 3600) == result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_read_corrupt_file_is_miss():
    tmp, d = _fresh_cache_dir()
    try:
        (d / "bad.json").write_text("{not json")
        assert grid_cache.read(d, "bad", None) is None
        (d / "bad2.json").write_text(json_null := "[1,2,3]")
        assert grid_cache.read(d, "bad2", None) is None, \
            "non-dict payload must miss"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_ttl_for_window():
    edge = date(2026, 8, 1)
    assert grid_cache.ttl_for_window(date(2026, 7, 29), edge) is None, \
        "archived windows never expire"
    assert grid_cache.ttl_for_window(edge, edge) is None
    assert grid_cache.ttl_for_window(date(2026, 8, 2), edge) == 86400
    with _EnvGuard(GRID_CACHE_RECENT_TTL_SECONDS="60"):
        assert grid_cache.ttl_for_window(date(2026, 8, 2), edge) == 60.0


def test_cache_write_evicts_oldest_first():
    tmp, d = _fresh_cache_dir()
    try:
        big = {"values": [[1.5] * 100] * 50}  # ~20KB serialized
        assert grid_cache.write(d, "probe", big, max_mb=999) is True
        size = (d / "probe.json").stat().st_size
        assert size > 10_000, f"payload unexpectedly small: {size}"
        shutil.rmtree(d)
        d.mkdir()
        cap_mb = 2.5 * size / (1024 * 1024)
        t0 = time.time()
        for i, key in enumerate(("k1", "k2", "k3")):
            assert grid_cache.write(d, key, big, max_mb=cap_mb) is True
            os.utime(d / f"{key}.json", (t0 + i * 10, t0 + i * 10))
        # 3 files exceed 2.5x one file: exactly the oldest (k1) is evicted.
        assert grid_cache.read(d, "k1", None) is None, "oldest must be evicted"
        assert grid_cache.read(d, "k2", None) == big
        assert grid_cache.read(d, "k3", None) == big
        total = sum(p.stat().st_size for p in d.glob("*.json"))
        assert total <= 2.5 * size, f"over cap: {total} > {2.5 * size}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_disabled_by_env():
    with _EnvGuard(GRID_CACHE_DISABLE="1", GRID_CACHE_DIR="/tmp/whatever"):
        assert grid_cache.cache_dir() is None


def test_cache_dir_unwritable_disables_cache():
    tmp = tempfile.mkdtemp(prefix="grid_cache_test_")
    try:
        blocker = os.path.join(tmp, "blocker")
        Path(blocker).write_text("not a dir")
        with _EnvGuard(GRID_CACHE_DIR=os.path.join(blocker, "sub")):
            assert grid_cache.cache_dir() is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# grid_fetch: parallel materialization


def test_fetch_workers_default_and_env():
    with _EnvGuard():
        os.environ.pop("GRID_FETCH_WORKERS", None)
        assert grid_fetch.fetch_workers() == 8
    with _EnvGuard(GRID_FETCH_WORKERS="3"):
        assert grid_fetch.fetch_workers() == 3
    with _EnvGuard(GRID_FETCH_WORKERS="0"):
        assert grid_fetch.fetch_workers() == 1, "clamped to >= 1"
    with _EnvGuard(GRID_FETCH_WORKERS="bogus"):
        assert grid_fetch.fetch_workers() == 8, "invalid falls back to default"


def test_parallel_compute_single_is_inline():
    FakeDA.reset_threads()
    da = FakeDA("v1", delay=0.05)
    out = grid_fetch.parallel_compute([da])
    assert out == ["v1"]


def test_parallel_compute_runs_concurrently_and_preserves_order():
    FakeDA.reset_threads()
    # First granule is slowest: order must still follow input, not finish time.
    das = [FakeDA("slow", delay=0.4), FakeDA("mid", delay=0.2),
           FakeDA("fast", delay=0.0)]
    t0 = time.monotonic()
    out = grid_fetch.parallel_compute(das)
    elapsed = time.monotonic() - t0
    assert out == ["slow", "mid", "fast"], "input order must be preserved"
    assert elapsed < 0.7, f"not parallel: {elapsed:.2f}s for 0.6s of sleeps"
    assert FakeDA.thread_count() >= 2, "expected multiple worker threads"


def test_parallel_compute_failure_raises():
    das = [FakeDA("ok"), FakeDA("bad", fail=True), FakeDA("ok2")]
    try:
        grid_fetch.parallel_compute(das)
    except RuntimeError as exc:
        assert "granule slice 1" in str(exc)
    else:
        raise AssertionError("expected RuntimeError from failing granule")


# ---------------------------------------------------------------------------
# runner (mirrors test_grid.py: plain python, pytest-compatible)


def _run():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in fns:
        try:
            fn()
        except Exception:
            failed.append((name, traceback.format_exc()))
            print(f"FAIL {name}")
        else:
            passed += 1
            print(f"ok   {name}")
    print(f"\n{passed} passed, {len(failed)} failed")
    for name, tb in failed:
        print(f"\n--- {name} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
