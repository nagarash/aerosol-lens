"""Disk cache for /grid responses (stdlib only, no geo deps).

Rationale: one /grid request range-reads dozens of hourly granules from
the GES DISC archive (observed 30-70s). Identical queries are common --
demo questions, shared links, the frontend re-asking after a mode
switch -- and historical MERRA-2 granules are immutable, so caching
responses on the Fly volume turns repeats into millisecond file reads.

Key: sha256(source | variable | agg | rounded bbox | canonical t0 | t1).
The bbox is snapped to 2 decimals (~1 km) by the caller before it
reaches the key, so "-17.0,15.0" and "-17,15" share an entry.

TTL: windows ending on/before the archive edge never expire (None);
windows touching the recent edge expire after
GRID_CACHE_RECENT_TTL_SECONDS (default 24h), because fresh granules
can still be revised upstream.

Eviction: LRU by file mtime, capped at GRID_CACHE_MAX_MB (default 500).

Robustness: writes are atomic (temp file + os.replace); a read racing
an eviction is treated as a miss; if the cache dir is unavailable every
helper degrades to miss/no-op so /grid keeps serving.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import date
from pathlib import Path

log = logging.getLogger("aerosol_lens.grid_cache")

ENV_DIR = "GRID_CACHE_DIR"
ENV_MAX_MB = "GRID_CACHE_MAX_MB"
ENV_RECENT_TTL = "GRID_CACHE_RECENT_TTL_SECONDS"
ENV_DISABLE = "GRID_CACHE_DISABLE"
DEFAULT_MAX_MB = 500
DEFAULT_RECENT_TTL_SECONDS = 86400  # 24h

_ok_dirs: set[str] = set()
_hits = 0
_misses = 0


def enabled() -> bool:
    return os.environ.get(ENV_DISABLE, "").strip().lower() not in (
        "1", "true", "yes",
    )


def cache_dir() -> Path | None:
    """Resolve the cache directory, or None when caching is unavailable."""
    if not enabled():
        return None
    d = Path(os.environ.get(ENV_DIR, "/data/grid_cache"))
    key = str(d)
    if key in _ok_dirs:
        return d
    try:
        d.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=d, prefix=".probe-", delete=True):
            pass
    except OSError as exc:
        log.warning("grid_cache: disabled, cannot use %s: %s", d, exc)
        return None
    _ok_dirs.add(key)
    return d


def normalize_bbox(bbox) -> tuple[float, float, float, float]:
    """Snap a (w, s, e, n) bbox to 2 decimals (~1 km at the equator)."""
    return tuple(round(float(v), 2) for v in bbox)  # type: ignore[return-value]


def cache_key(source: str, variable: str, agg: str,
              bbox, start, end, data_source: str = "") -> str:
    """Stable 32-hex-char key for a normalized grid query.

    `data_source` ("fields" vs "kerchunk") participates in the key so a
    cached response is always labeled with the path that produced it.
    Empty (the kerchunk default) keeps pre-existing cache keys stable.
    """
    w, s, e, n = normalize_bbox(bbox)
    parts = [
        source.strip().lower(),
        variable.strip().upper(),
        agg.strip().lower(),
        f"{w:.2f},{s:.2f},{e:.2f},{n:.2f}",
        start.isoformat(),
        end.isoformat(),
    ]
    ds = data_source.strip().lower()
    if ds:
        # Only the fields fast path passes a data_source, so pre-existing
        # kerchunk-path keys are byte-identical to before this parameter
        # existed (no cache invalidation on deploy).
        parts.append(ds)
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def ttl_for_window(end_date: date, archive_edge: date) -> float | None:
    """Seconds a cached response stays valid; None = never expires."""
    if end_date <= archive_edge:
        return None
    try:
        return max(0.0, float(os.environ.get(ENV_RECENT_TTL,
                                             DEFAULT_RECENT_TTL_SECONDS)))
    except ValueError:
        return float(DEFAULT_RECENT_TTL_SECONDS)


def _path(directory: Path, key: str) -> Path:
    return directory / f"{key}.json"


def read(directory: Path, key: str, ttl_seconds: float | None):
    """Return the cached result dict, or None on miss/expiry/corruption."""
    global _hits, _misses
    try:
        with open(_path(directory, key), "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, ValueError):
        _misses += 1
        return None
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict):
        _misses += 1
        return None
    if ttl_seconds is not None:
        try:
            age = time.time() - float(payload.get("cached_at", 0))
        except (TypeError, ValueError):
            age = float("inf")
        if age > ttl_seconds:
            _misses += 1
            return None
    _hits += 1
    return result


def write(directory: Path, key: str, result: dict,
          max_mb: float | None = None) -> bool:
    """Atomically store a /grid response; evict oldest entries past the cap."""
    if max_mb is None:
        try:
            max_mb = float(os.environ.get(ENV_MAX_MB, DEFAULT_MAX_MB))
        except ValueError:
            max_mb = float(DEFAULT_MAX_MB)
    payload = {"cached_at": time.time(), "key": key, "result": result}
    tmp = directory / f".tmp-{key}-{os.getpid()}.json"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
        os.replace(tmp, _path(directory, key))
    except OSError as exc:
        log.warning("grid_cache: write failed for %s: %s", key, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    _evict_if_needed(directory, max_mb)
    return True


def _evict_if_needed(directory: Path, max_mb: float) -> None:
    cap_bytes = max_mb * 1024 * 1024
    try:
        entries = [(p.stat().st_mtime, p.stat().st_size, p)
                   for p in directory.glob("*.json")]
    except OSError:
        return
    total = sum(size for _, size, _ in entries)
    if total <= cap_bytes:
        return
    entries.sort(key=lambda t: t[0])  # oldest first
    for _, size, p in entries:
        try:
            p.unlink()
        except OSError:
            continue
        total -= size
        if total <= cap_bytes:
            break
    log.info("grid_cache: evicted down to %.1fMB (cap %.0fMB)",
             total / 1024 / 1024, max_mb)


def stats() -> dict:
    return {"hits": _hits, "misses": _misses}
