"""Tests for backend/rate_limit.py and the /grid rate-limit middleware.

The middleware test drives GridRateLimitMiddleware at the raw ASGI level
(no TestClient/httpx needed): it wraps a dummy inner app and asserts the
second request inside the window gets 429 + Retry-After.
Run from the repo root with:
    python backend/test_rate_limit.py
(pytest-compatible as well: every test_* function takes no arguments.)
"""

import asyncio
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import rate_limit
from backend.app import GridRateLimitMiddleware
from backend.rate_limit import RateLimiter, client_ip


def test_allow_up_to_limit_then_deny():
    rl = RateLimiter(per_minute=2)
    assert rl.allow("1.2.3.4") is True
    assert rl.allow("1.2.3.4") is True
    assert rl.allow("1.2.3.4") is False
    # Other IPs are unaffected.
    assert rl.allow("5.6.7.8") is True


def test_retry_after_positive_when_limited():
    rl = RateLimiter(per_minute=1)
    assert rl.retry_after("9.9.9.9") == 0, "fresh key has nothing to wait for"
    assert rl.allow("9.9.9.9") is True
    assert rl.allow("9.9.9.9") is False
    ra = rl.retry_after("9.9.9.9")
    assert 1 <= ra <= 60, ra


def test_window_expiry_restores():
    rl = RateLimiter(per_minute=1, window_s=0.05)
    assert rl.allow("1.1.1.1") is True
    assert rl.allow("1.1.1.1") is False
    time.sleep(0.06)
    assert rl.allow("1.1.1.1") is True


def test_reset_clears_counters():
    rl = RateLimiter(per_minute=1)
    assert rl.allow("2.2.2.2") is True
    assert rl.allow("2.2.2.2") is False
    rl.reset()
    assert rl.allow("2.2.2.2") is True


def test_client_ip_prefers_xff():
    scope = {
        "headers": [(b"x-forwarded-for", b"203.0.113.7, 70.41.3.18")],
        "client": ("10.0.0.1", 1234),
    }
    assert client_ip(scope) == "203.0.113.7"
    assert client_ip({"headers": [], "client": ("10.0.0.2", 80)}) == "10.0.0.2"
    assert client_ip({"headers": []}) == "unknown"


async def _call_asgi(app, scope):
    """Drive one ASGI request; return (status, headers, body)."""
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    headers = dict(
        (k.decode(), v.decode())
        for m in messages
        if m["type"] == "http.response.start"
        for k, v in m["headers"]
    )
    body = b"".join(
        m.get("body", b"") for m in messages if m["type"] == "http.response.body"
    )
    return status, headers, body


def _scope(path, ip="198.51.100.9"):
    return {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "query_string": b"",
        "headers": [],
        "client": (ip, 54321),
    }


async def _inner_app_ok(scope, receive, send):
    from starlette.responses import JSONResponse

    await JSONResponse({"ok": True})(scope, receive, send)


def test_middleware_limits_grid_only():
    rate_limit.reset_limiter(per_minute=1)
    try:
        mw = GridRateLimitMiddleware(_inner_app_ok)
        # First /grid request passes through.
        status, _, body = asyncio.run(_call_asgi(mw, _scope("/grid")))
        assert status == 200, (status, body)
        # Second /grid request from the same IP is limited.
        status, headers, body = asyncio.run(_call_asgi(mw, _scope("/grid")))
        assert status == 429, (status, body)
        assert "retry-after" in headers, headers
        assert int(headers["retry-after"]) >= 1
        assert b"rate limit" in body
        # Other paths are untouched, even when /grid is exhausted.
        status, _, _ = asyncio.run(_call_asgi(mw, _scope("/ask")))
        assert status == 200, status
        # A different IP still gets through to /grid.
        status, _, _ = asyncio.run(
            _call_asgi(mw, _scope("/grid", ip="198.51.100.10"))
        )
        assert status == 200, status
    finally:
        rate_limit.reset_limiter()  # restore the default limiter


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
