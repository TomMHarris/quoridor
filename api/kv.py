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

The local dev server calls use_memory() instead, which keeps rooms in a dict.

Updates go through load() + swap(), a compare-and-set, so two requests editing
the same room can't silently overwrite each other.

Every failure raises StoreUnavailable so the caller can answer with one clean
status instead of leaking a driver's exception text to the browser.
"""

import json
import os
import re
import sys
import threading
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


# Every connection failure surfaces from the redis driver as the same
# ConnectionError class — a deleted database, a refused port and a TLS mismatch
# are indistinguishable by type. The message tells them apart, so it is mapped
# to a short reason that is safe to show the browser.
_REASONS = (
    ("host_not_found", ("nodename nor servname", "name or service not known",
                        "name resolution", "getaddrinfo", "no address associated")),
    ("auth_failed", ("wrongpass", "invalid password", "invalid username",
                     "noauth", "authentication")),
    ("tls_error", ("ssl", "certificate", "tls")),
    ("connection_refused", ("connection refused",)),
    ("connection_reset", ("connection reset", "closed by server", "broken pipe")),
    ("timeout", ("timed out", "timeout")),
)

_HINTS = {
    "host_not_found": "the database host no longer exists - deleted, or the URL is stale",
    "auth_failed": "the password in the URL was rejected - rotated or wrong",
    "tls_error": "TLS handshake failed",
    "connection_refused": "nothing is listening at that host and port",
    "connection_reset": "the server hung up - often redis:// used where rediss:// (TLS) is required",
    "timeout": "the server did not answer in time",
}


def _reason(exc):
    text = f"{type(exc).__name__} {exc}".lower()
    for reason, needles in _REASONS:
        if any(n in text for n in needles):
            return reason
    return type(exc).__name__


def _redact(text):
    """Strip credentials from anything that looks like a URL."""
    return re.sub(r"//[^/@\s]*@", "//***@", str(text))


def _fail(source, what, exc):
    """Log the full failure server-side and raise one with a safe summary."""
    reason = _reason(exc)
    print(f"[kv] {what} via {source} failed: {reason}: "
          f"{type(exc).__name__}: {_redact(exc)}", file=sys.stderr)
    hint = _HINTS.get(reason)
    raise StoreUnavailable(f"{what} via {source}: {reason}" +
                           (f" ({hint})" if hint else "")) from exc


def _rest_config():
    for url_var, token_var in _REST_PAIRS:
        url = os.environ.get(url_var, "").strip().rstrip("/")
        token = os.environ.get(token_var, "").strip()
        if url and token:
            return url, token, url_var
    return None, None, None


def _redis_url():
    for var in _URL_VARS:
        url = os.environ.get(var, "").strip()
        if url:
            return url, var
    return "", None


def _redis_source():
    """Which variable and scheme the TCP transport uses, e.g. 'REDIS_URL (rediss://)'."""
    url, var = _redis_url()
    scheme = url.split("://", 1)[0] if "://" in url else "?"
    return f"{var} ({scheme}://)"


# --- HTTP REST transport ---------------------------------------------------


def _rest_command(url, token, source, command):
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
        if e.code in (401, 403):
            _fail(source, command[0], Exception(f"authentication: HTTP {e.code}"))
        _fail(source, command[0], Exception(f"HTTP {e.code}"))
    except Exception as e:  # timeout, DNS, TLS, malformed body
        _fail(source, command[0], getattr(e, "reason", e))

    if isinstance(payload, dict) and payload.get("error"):
        _fail(source, command[0], Exception(str(payload["error"])[:200]))
    return payload.get("result") if isinstance(payload, dict) else None


# --- Redis TCP transport ---------------------------------------------------

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    url, _ = _redis_url()
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
        _fail(_redis_source(), "connect", e)
    return _client


# --- In-process transport (local dev server) --------------------------------

_memory = None
_memory_lock = threading.Lock()


def use_memory():
    """Keep rooms in this process instead of a real store (app.py, tests)."""
    global _memory
    _memory = {}


def _memory_command(args):
    """The handful of commands this module sends, against a dict."""
    op = args[0]
    with _memory_lock:
        if op == "GET":
            return _memory.get(args[1])
        if op == "SET":
            key, value, options = args[1], args[2], args[3:]
            if "NX" in options and key in _memory:
                return None
            _memory[key] = value
            return "OK"
        if op == "EVAL" and args[1] == _SWAP_SCRIPT:
            key, expected, value = args[3], args[4], args[5]
            if _memory.get(key) != expected:
                return 0
            _memory[key] = value
            return 1
    raise ValueError(f"in-memory store does not support {op}")


# --- Public API ------------------------------------------------------------


def available():
    """Is any store configured? (Says nothing about whether it responds.)"""
    url, token, _ = _rest_config()
    return _memory is not None or bool(url and token) or bool(_redis_url()[0])


def _command(*args):
    """Run one Redis command on whichever store is configured."""
    if _memory is not None:
        return _memory_command(args)

    url, token, var = _rest_config()
    if url and token:
        return _rest_command(url, token, f"{var} (REST)", list(args))

    client = _get_client()
    if client is None:
        raise StoreUnavailable("no store configured")
    try:
        return client.execute_command(*args)
    except Exception as e:
        _fail(_redis_source(), args[0], e)


# Compare-and-set in one round trip. Comparing the whole stored text needs no
# version field and no JSON support in the store's Lua.
_SWAP_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
  return 1
end
return 0
"""


def load(key):
    """(value, raw) for key, or (None, None). Pass raw to swap() to update it."""
    raw = _command("GET", key)
    return (json.loads(raw), raw) if raw else (None, None)


def get(key):
    return load(key)[0]


def create(key, value, ex):
    """Store value under key unless the key exists. False if it was taken."""
    return bool(_command("SET", key, json.dumps(value), "NX", "EX", int(ex)))


def swap(key, expected_raw, value, ex):
    """
    Replace key with value only if it still holds expected_raw, the text load()
    returned. False means another request wrote first: load again and retry.
    """
    result = _command("EVAL", _SWAP_SCRIPT, 1, key, expected_raw,
                      json.dumps(value), int(ex))
    return int(result or 0) == 1
