"""
Key/value store for online room state.

Two transports, tried in this order:

  1. HTTP REST (Upstash / Vercel KV) — KV_REST_API_URL + KV_REST_API_TOKEN, or
     UPSTASH_REDIS_REST_URL + UPSTASH_REDIS_REST_TOKEN. Preferred on Vercel: a
     plain request per call, no connection pool to keep alive across a
     serverless function's short life, and no `redis` dependency.
  2. A Redis TCP URL — REDIS_URL (also KV_URL / STORAGE_REDIS_URL /
     REDIS_TLS_URL, the names the various Vercel storage integrations set).

Several names are accepted because which variables exist depends on which
provider is attached, and a room store that silently looks "not configured"
because the URL arrived under a different name is a bad failure mode.

Every failure raises StoreUnavailable so the caller can answer with one clean
status instead of leaking a driver's exception text to the browser.
"""

import json
import os
import urllib.error
import urllib.request

TIMEOUT = 5  # seconds — a serverless function must not hang on a dead host

_REST_PAIRS = (
    ("KV_REST_API_URL", "KV_REST_API_TOKEN"),
    ("UPSTASH_REDIS_REST_URL", "UPSTASH_REDIS_REST_TOKEN"),
)
_URL_VARS = ("REDIS_URL", "KV_URL", "STORAGE_REDIS_URL", "REDIS_TLS_URL")


class StoreUnavailable(Exception):
    """The store is configured but could not be reached."""


def _rest_config():
    for url_var, token_var in _REST_PAIRS:
        url = os.environ.get(url_var, "").strip().rstrip("/")
        token = os.environ.get(token_var, "").strip()
        if url and token:
            return url, token
    return None, None


def _redis_url():
    for var in _URL_VARS:
        url = os.environ.get(var, "").strip()
        if url:
            return url
    return ""


def available():
    """Is any store configured? (Says nothing about whether it responds.)"""
    url, token = _rest_config()
    return bool(url and token) or bool(_redis_url())


# --- HTTP REST transport ---------------------------------------------------


def _rest_command(url, token, command):
    """Run one Redis command through the REST API, e.g. ["GET", "room:ab12"]."""
    req = urllib.request.Request(
        url,
        data=json.dumps(command).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise StoreUnavailable(f"store returned HTTP {e.code}") from e
    except Exception as e:  # timeout, DNS, TLS, malformed body
        raise StoreUnavailable(f"store unreachable: {type(e).__name__}") from e

    if isinstance(payload, dict) and payload.get("error"):
        raise StoreUnavailable(str(payload["error"])[:200])
    return payload.get("result") if isinstance(payload, dict) else None


# --- Redis TCP transport ---------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    url = _redis_url()
    if not url:
        return None
    try:
        import redis  # imported lazily: the REST transport needs no driver
    except ImportError as e:
        raise StoreUnavailable("redis package not installed") from e
    try:
        _client = redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=TIMEOUT,
            socket_timeout=TIMEOUT,
        )
    except Exception as e:
        raise StoreUnavailable(f"bad redis url: {type(e).__name__}") from e
    return _client


# --- Public API ------------------------------------------------------------


def get(key):
    url, token = _rest_config()
    if url and token:
        raw = _rest_command(url, token, ["GET", key])
        return json.loads(raw) if raw else None

    client = _get_client()
    if client is None:
        raise StoreUnavailable("no store configured")
    try:
        raw = client.get(key)
    except Exception as e:
        raise StoreUnavailable(f"redis get failed: {type(e).__name__}") from e
    return json.loads(raw) if raw else None


def set(key, value, ex=None):
    payload = json.dumps(value)

    url, token = _rest_config()
    if url and token:
        command = ["SET", key, payload]
        if ex:
            command += ["EX", int(ex)]
        return _rest_command(url, token, command)

    client = _get_client()
    if client is None:
        raise StoreUnavailable("no store configured")
    try:
        return client.set(key, payload, ex=ex)
    except Exception as e:
        raise StoreUnavailable(f"redis set failed: {type(e).__name__}") from e
