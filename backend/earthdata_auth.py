"""Earthdata Login token -> temporary S3 credential exchange.

The MERRA-2 bucket (``gesdisc-cumulus-prod-protected``) rejects anonymous
reads. In production the deployer sets ``EARTHDATA_TOKEN`` -- a long-lived
Earthdata Login bearer token, server-side only (Fly.io secrets in
production, never the repo or the frontend). This module exchanges it for
short-lived AWS session credentials via the GES DISC ``s3credentials``
endpoint and caches them in-process, refreshing when within 5 minutes of
expiry.

S3 access resolution (``s3_target_options()``)::

    1. MERRA2_S3_ANON=1         -> anonymous (genuinely public mirrors only)
    2. EARTHDATA_TOKEN set      -> exchange + cache (the production path)
    3. AWS_ACCESS_KEY_ID in env -> static deployer credentials via the
                                  normal AWS credential chain (fsspec default)
    4. otherwise                -> EarthdataTokenMissingError (HTTP 501)

Only stdlib (urllib) is used for the exchange, so no new dependency. The
token never appears in logs or error messages.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

S3CREDENTIALS_URL = "https://data.gesdisc.earthdata.nasa.gov/s3credentials"

# Refresh the cached session this far ahead of its expiry.
_REFRESH_BUFFER = timedelta(minutes=5)


class EarthdataAuthError(Exception):
    """Base class for Earthdata credential failures."""


class EarthdataTokenMissingError(EarthdataAuthError):
    """No EARTHDATA_TOKEN (and no other S3 credentials) configured. -> 501"""


class EarthdataExchangeError(EarthdataAuthError):
    """The s3credentials exchange failed (rejected token, unreachable). -> 502"""


# In-process cache: {token, access_key, secret_key, session_token, expires_at}.
_cache: dict = {}
_lock = threading.Lock()


def reset_cache() -> None:
    """Drop the cached session credentials (tests, token rotation)."""
    with _lock:
        _cache.clear()


def _http_get_json(url: str, token: str) -> dict:
    """GET url with `Authorization: Bearer <token>`; return the parsed JSON.

    Separated from _exchange() so unit tests can stub the HTTP layer
    without ever touching the real endpoint.
    """
    req = urllib.request.Request(url, method="GET")
    # The token lives only in this request header and process memory; it is
    # never logged and never interpolated into error messages.
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        if exc.code in (401, 403):
            raise EarthdataExchangeError(
                f"s3credentials rejected the request (HTTP {exc.code}): "
                f"{body}. The EARTHDATA_TOKEN is invalid, expired, or "
                f"revoked -- generate a new one in the Earthdata Login "
                f"profile and update the server secret."
            ) from exc
        raise EarthdataExchangeError(
            f"s3credentials returned HTTP {exc.code}: {body}"
        ) from exc
    except Exception as exc:  # URLError, timeout, ...
        raise EarthdataExchangeError(
            f"could not reach the s3credentials endpoint "
            f"({S3CREDENTIALS_URL}): {exc}"
        ) from exc


def _parse_expiration(value) -> datetime:
    """Parse the s3credentials `expiration` field to an aware UTC datetime."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _exchange(token: str) -> tuple[str, str, str, datetime]:
    """One token -> session-credential exchange. Returns (ak, sk, st, exp)."""
    payload = _http_get_json(S3CREDENTIALS_URL, token)
    missing = [
        k
        for k in ("accessKeyId", "secretAccessKey", "sessionToken", "expiration")
        if k not in payload
    ]
    if missing:
        raise EarthdataExchangeError(
            f"s3credentials response was missing keys: {missing}."
        )
    return (
        payload["accessKeyId"],
        payload["secretAccessKey"],
        payload["sessionToken"],
        _parse_expiration(payload["expiration"]),
    )


def get_s3_credentials() -> tuple[str, str, str]:
    """Return fresh (access_key, secret_key, session_token) for the S3 bucket.

    Uses the in-process cache; re-exchanges when the cached session is
    within 5 minutes of expiry. Raises EarthdataTokenMissingError when
    EARTHDATA_TOKEN is unset, EarthdataExchangeError when the endpoint
    rejects the token or is unreachable. Error messages never contain
    the token.
    """
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if not token:
        raise EarthdataTokenMissingError(
            "EARTHDATA_TOKEN is not set. Set it to an Earthdata Login bearer "
            "token (server-side only -- e.g. a Fly.io secret in production, "
            "never in the repo or the frontend) so the backend can fetch "
            "temporary S3 credentials for the protected MERRA-2 bucket. "
            "Alternatively export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY "
            "/ AWS_SESSION_TOKEN directly, or set MERRA2_S3_ANON=1 for a "
            "genuinely public mirror."
        )
    now = datetime.now(timezone.utc)
    with _lock:
        if (
            _cache
            and _cache.get("token") == token
            and _cache["expires_at"] - now > _REFRESH_BUFFER
        ):
            return (
                _cache["access_key"],
                _cache["secret_key"],
                _cache["session_token"],
            )
    access_key, secret_key, session_token, expires_at = _exchange(token)
    with _lock:
        _cache.update(
            {
                "token": token,
                "access_key": access_key,
                "secret_key": secret_key,
                "session_token": session_token,
                "expires_at": expires_at,
            }
        )
    return access_key, secret_key, session_token


def s3_target_options() -> dict:
    """fsspec target options for the protected MERRA-2 S3 bucket.

    Resolution order: MERRA2_S3_ANON=1 -> anonymous; EARTHDATA_TOKEN set ->
    exchanged session credentials; AWS_ACCESS_KEY_ID present -> standard
    AWS credential chain; otherwise EarthdataTokenMissingError.
    """
    if os.environ.get("MERRA2_S3_ANON") == "1":
        return {"anon": True}
    if os.environ.get("EARTHDATA_TOKEN", "").strip():
        access_key, secret_key, session_token = get_s3_credentials()
        return {
            "key": access_key,
            "secret": secret_key,
            "token": session_token,
            "anon": False,
        }
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        return {}  # deployer-provided static creds; fsspec uses the AWS chain
    raise EarthdataTokenMissingError(
        "no S3 credentials available for the protected MERRA-2 bucket: set "
        "EARTHDATA_TOKEN (preferred), export AWS_ACCESS_KEY_ID / "
        "AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN, or set "
        "MERRA2_S3_ANON=1 for a public mirror."
    )


# --------------------------------------------------------------------------
# HTTPS access (portable; works outside AWS us-west-2)

DATA_HTTPS_BASE = "https://data.gesdisc.earthdata.nasa.gov/data"


def https_target_options() -> dict:
    """fsspec options for HTTPS range reads of the protected MERRA-2 archive.

    Why this exists: GES DISC grants *direct S3* access only to callers
    running inside AWS us-west-2. From anywhere else (Fly.io, a laptop)
    the exchanged session credentials are valid but every GetObject comes
    back 403 Forbidden -- the denial is by request origin, not by token.
    The HTTPS archive endpoint accepts the Earthdata bearer token directly
    and serves ranged reads from any network, so it is the portable path
    and the default for this deployment.

    Returns aiohttp client kwargs carrying the bearer header. The token is
    never logged; callers pass these straight to fsspec.
    """
    token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if not token:
        raise EarthdataTokenMissingError(
            "EARTHDATA_TOKEN is not set. HTTPS access to the protected "
            "MERRA-2 archive needs an Earthdata Login bearer token "
            "(server-side only -- e.g. a Fly.io secret in production)."
        )
    return {"client_kwargs": {"headers": {"Authorization": f"Bearer {token}"}}}


def https_url_for_s3(url: str) -> str:
    """Map an s3:// MERRA-2 granule URL to its HTTPS archive equivalent.

    s3://gesdisc-cumulus-prod-protected/MERRA2/<rest>
        -> https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/<rest>
    Non-s3 URLs pass through unchanged.
    """
    if not url.startswith("s3://"):
        return url
    path = url[len("s3://"):]
    _bucket, _, rest = path.partition("/")
    return f"{DATA_HTTPS_BASE}/{rest}"
