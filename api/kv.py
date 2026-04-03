"""
Thin Upstash Redis REST API wrapper. Zero external dependencies (stdlib only).

Expects environment variables:
  KV_REST_API_URL   — Upstash REST endpoint (e.g. https://xxx.upstash.io)
  KV_REST_API_TOKEN — Upstash REST token
"""

import os
import json
import urllib.request

KV_URL = os.environ.get("KV_REST_API_URL", "")
KV_TOKEN = os.environ.get("KV_REST_API_TOKEN", "")


def available():
    return bool(KV_URL and KV_TOKEN)


def _request(command_args):
    """Send a Redis command to Upstash REST API."""
    req = urllib.request.Request(
        KV_URL,
        data=json.dumps(command_args).encode(),
        headers={
            "Authorization": f"Bearer {KV_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())["result"]


def get(key):
    val = _request(["GET", key])
    return json.loads(val) if val else None


def set(key, value, ex=None):
    args = ["SET", key, json.dumps(value)]
    if ex:
        args += ["EX", str(ex)]
    return _request(args)
