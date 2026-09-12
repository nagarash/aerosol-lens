"""Per-IP rate limiting for /grid (token-bucket, in-process, no new deps).

Every /grid request range-reads the protected MERRA-2 bucket against the
deployer's Earthdata credentials, so one abusive client burns the
deployer's quota. This tiny sliding-window limiter caps requests per
client IP; when exceeded, /grid returns 429 with a Retry-After header.

Default: GRID_RATE_LIMIT_PER_MIN requests per 60 s per IP (default 30).

v1 simplifications (documented, not hidden):
- In-process memory only: each backend replica keeps its own counters.
  Fine for a single Fly.io instance; use a shared store if you scale
  horizontally.
- Client identity is the X-Forwarded-For first entry when present
  (Fly.io and most PaaS proxies set it), else the direct peer IP. v1
  trusts the platform's XFF; don't expose the backend directly to
  clients who can spoof it.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque

DEFAULT_PER_MINUTE = 30
_MAX_TRACKED_IPS = 10_000  # bound memory: evict oldest entry past this


class RateLimiter:
    """Sliding-window rate limiter keyed by client IP."""

    def __init__(self, per_minute: int = DEFAULT_PER_MINUTE,
                 window_s: float = 60.0):
        if per_minute < 1:
            raise ValueError("per_minute must be >= 1")
        self.per_minute = per_minute
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        dq = self._hits.get(key)
        if dq is None:
            dq = self._hits[key] = deque()
        cutoff = now - self.window_s
        while dq and dq[0] <= cutoff:
            dq.popleft()
        return dq

    def allow(self, key: str) -> bool:
        """Record a hit for key; True if within the limit."""
        now = time.monotonic()
        with self._lock:
            dq = self._prune(key, now)
            if len(dq) >= self.per_minute:
                return False
            dq.append(now)
            if len(self._hits) > _MAX_TRACKED_IPS:
                # Evict an arbitrary (oldest-inserted) entry to bound memory.
                self._hits.pop(next(iter(self._hits)))
            return True

    def retry_after(self, key: str) -> int:
        """Seconds until key's oldest windowed hit expires (>= 1)."""
        now = time.monotonic()
        with self._lock:
            dq = self._prune(key, now)
            if not dq:
                return 0
            return max(1, math.ceil(dq[0] + self.window_s - now))

    def reset(self) -> None:
        """Clear all counters (tests)."""
        with self._lock:
            self._hits.clear()


def _default_limit() -> int:
    try:
        return max(1, int(os.environ.get("GRID_RATE_LIMIT_PER_MIN", "")
                         or DEFAULT_PER_MINUTE))
    except ValueError:
        return DEFAULT_PER_MINUTE


_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def get_limiter() -> RateLimiter:
    """Process-wide /grid limiter, configured from GRID_RATE_LIMIT_PER_MIN."""
    global _limiter
    with _limiter_lock:
        if _limiter is None:
            _limiter = RateLimiter(per_minute=_default_limit())
        return _limiter


def reset_limiter(per_minute: int | None = None) -> RateLimiter:
    """Rebuild the process-wide limiter (tests, config reload)."""
    global _limiter
    with _limiter_lock:
        _limiter = RateLimiter(
            per_minute=per_minute if per_minute is not None else _default_limit()
        )
        return _limiter


def client_ip(scope: dict) -> str:
    """Best-effort client IP for an ASGI scope (XFF-aware, see caveats)."""
    for name, value in scope.get("headers", []):
        if name == b"x-forwarded-for":
            first = value.decode("latin1").split(",")[0].strip()
            if first:
                return first
    peer = scope.get("client")
    return peer[0] if peer else "unknown"
