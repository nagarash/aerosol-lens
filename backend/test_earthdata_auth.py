"""Tests for backend/earthdata_auth.py (Earthdata token -> S3 credential flow).

The HTTP layer (_http_get_json) is stubbed in every test: unit tests
never touch the real s3credentials endpoint and never see a real token.
Run from the repo root with:
    python backend/test_earthdata_auth.py
(pytest-compatible as well: every test_* function takes no arguments.)
"""

import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import earthdata_auth
from backend.earthdata_auth import (
    EarthdataExchangeError,
    EarthdataTokenMissingError,
    _parse_expiration,
    get_s3_credentials,
    reset_cache,
    s3_target_options,
)

FAKE_TOKEN = "fake-earthdata-token-for-tests"
OTHER_TOKEN = "another-fake-token"


def payload(expires_in_s=3600):
    exp = datetime.now(timezone.utc) + timedelta(seconds=expires_in_s)
    return {
        "accessKeyId": "ASIAFAKEKEY",
        "secretAccessKey": "fake-secret",
        "sessionToken": "fake-session-token",
        "expiration": exp.isoformat(),
    }


@contextmanager
def env(**kwargs):
    """Temporarily set/unset env vars (None unsets)."""
    saved = {k: os.environ.get(k) for k in kwargs}
    try:
        for k, v in kwargs.items():
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
def stub_http(fn):
    """Temporarily replace earthdata_auth._http_get_json with fn(url, token)."""
    original = earthdata_auth._http_get_json
    earthdata_auth._http_get_json = fn
    try:
        yield
    finally:
        earthdata_auth._http_get_json = original


def clean_env():
    return env(
        EARTHDATA_TOKEN=None, AWS_ACCESS_KEY_ID=None,
        AWS_SECRET_ACCESS_KEY=None, AWS_SESSION_TOKEN=None,
        MERRA2_S3_ANON=None,
    )


# ---------------------------------------------------------------------------
# Tests


def test_exchange_success_and_caching():
    reset_cache()
    calls = []

    def fake(url, token):
        calls.append((url, token))
        assert token == FAKE_TOKEN, "token must reach the exchange"
        return payload()

    with clean_env(), env(EARTHDATA_TOKEN=FAKE_TOKEN), stub_http(fake):
        creds1 = get_s3_credentials()
        creds2 = get_s3_credentials()
    assert creds1 == ("ASIAFAKEKEY", "fake-secret", "fake-session-token")
    assert creds1 == creds2
    assert len(calls) == 1, f"second call must use the cache, got {len(calls)}"


def test_refresh_before_expiry():
    reset_cache()
    calls = []

    def fake(url, token):
        calls.append(token)
        return payload(expires_in_s=60)  # inside the 5-min refresh buffer

    with clean_env(), env(EARTHDATA_TOKEN=FAKE_TOKEN), stub_http(fake):
        get_s3_credentials()
        get_s3_credentials()
    assert len(calls) == 2, "near-expiry session must be re-exchanged"


def test_401_becomes_exchange_error_without_token_leak():
    reset_cache()

    def fake(url, token):
        raise EarthdataExchangeError(
            "s3credentials rejected the request (HTTP 401): bad token"
        )

    with clean_env(), env(EARTHDATA_TOKEN=FAKE_TOKEN), stub_http(fake):
        try:
            get_s3_credentials()
        except EarthdataExchangeError as exc:
            assert FAKE_TOKEN not in str(exc), "token leaked into error message"
            assert "401" in str(exc)
            return
    raise AssertionError("expected EarthdataExchangeError")


def test_missing_token_is_honest_501_error():
    reset_cache()
    with clean_env():
        try:
            get_s3_credentials()
        except EarthdataTokenMissingError as exc:
            assert "EARTHDATA_TOKEN" in str(exc)
            return
    raise AssertionError("expected EarthdataTokenMissingError")


def test_s3_target_options_resolution():
    reset_cache()

    def fake(url, token):
        return payload()

    # 1. anon escape hatch wins over everything
    with clean_env(), env(MERRA2_S3_ANON="1", EARTHDATA_TOKEN=FAKE_TOKEN):
        assert s3_target_options() == {"anon": True}

    # 2. token -> exchanged session credentials
    reset_cache()
    with clean_env(), env(EARTHDATA_TOKEN=FAKE_TOKEN), stub_http(fake):
        opts = s3_target_options()
    assert opts == {
        "key": "ASIAFAKEKEY",
        "secret": "fake-secret",
        "token": "fake-session-token",
        "anon": False,
    }, opts

    # 3. static AWS creds -> standard chain (empty options)
    reset_cache()
    with clean_env(), env(AWS_ACCESS_KEY_ID="AKIAFAKE"):
        assert s3_target_options() == {}

    # 4. nothing configured -> honest 501 error naming the token
    reset_cache()
    with clean_env():
        try:
            s3_target_options()
        except EarthdataTokenMissingError as exc:
            assert "EARTHDATA_TOKEN" in str(exc)
            return
    raise AssertionError("expected EarthdataTokenMissingError")


def test_exchange_missing_keys_is_502_error():
    reset_cache()

    def fake(url, token):
        return {"accessKeyId": "x"}  # missing the rest

    with clean_env(), env(EARTHDATA_TOKEN=FAKE_TOKEN), stub_http(fake):
        try:
            get_s3_credentials()
        except EarthdataExchangeError as exc:
            assert "missing keys" in str(exc)
            return
    raise AssertionError("expected EarthdataExchangeError")


def test_parse_expiration_variants():
    # The real endpoint returns "2026-09-12 17:23:04+00:00" (space, not T).
    dt = _parse_expiration("2026-09-12 17:23:04+00:00")
    assert (dt.year, dt.month, dt.day, dt.hour) == (2026, 9, 12, 17)
    assert dt.tzinfo is not None
    dt = _parse_expiration("2026-09-12T17:23:04Z")
    assert dt.tzinfo is not None
    dt = _parse_expiration("2026-09-12 17:23:04")  # naive -> assume UTC
    assert dt.tzinfo == timezone.utc
    dt = _parse_expiration(1780000000.0)  # epoch seconds
    assert dt.tzinfo == timezone.utc


def test_token_rotation_invalidates_cache():
    reset_cache()
    seen = []

    def fake(url, token):
        seen.append(token)
        return payload()

    with clean_env(), stub_http(fake):
        with env(EARTHDATA_TOKEN=FAKE_TOKEN):
            get_s3_credentials()
        with env(EARTHDATA_TOKEN=OTHER_TOKEN):
            get_s3_credentials()
    assert seen == [FAKE_TOKEN, OTHER_TOKEN], "rotated token must re-exchange"


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
