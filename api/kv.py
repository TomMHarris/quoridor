"""
Redis wrapper for online room storage.

Uses REDIS_URL environment variable (set automatically by Vercel Redis integration).
"""

import os
import json
import redis

REDIS_URL = os.environ.get("REDIS_URL", "")

_client = None


def _get_client():
    global _client
    if _client is None and REDIS_URL:
        _client = redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def available():
    return bool(REDIS_URL)


def get(key):
    r = _get_client()
    if not r:
        return None
    val = r.get(key)
    return json.loads(val) if val else None


def set(key, value, ex=None):
    r = _get_client()
    if not r:
        return None
    return r.set(key, json.dumps(value), ex=ex)
