"""One small helper for talking to web APIs politely.

Three habits, applied to every request in the project:
  1. cache  - the answer to every request is saved on disk (data/raw/_cache). Asking the same thing
              twice reads the file instead of the network, so re-running a script is free and fast.
  2. pace   - never fire requests at a server faster than it asks for (arXiv: one per 3 seconds).
  3. retry  - if a server hiccups (HTTP 429/5xx, an empty body, a timeout), wait and try again;
              if the request itself is wrong (HTTP 4xx other than 429), give up at once.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from typing import Callable

import requests

from . import config as C

HEADERS = {"User-Agent": "paper-adoption-predictor/0.1 (Fontys bachelor project; python-requests)"}

_last_call: dict[str, float] = {}   # host name -> moment of the last request to it, for pacing


def _cache_path(key: str):
    """A file name that is unique for this request (a SHA-1 fingerprint of url + parameters + body)."""
    digest = hashlib.sha1(key.encode()).hexdigest()
    return C.CACHE_DIR / digest[:2] / f"{digest}.txt"


def _request(method: str, url: str, params: dict, json_body: dict | None, *, extra: dict | None,
             pace_seconds: float, retries: int, valid: Callable[[str], bool] | None, not_found_ok: bool,
             headers: dict | None = None) -> str:
    key = url + "?" + json.dumps(params, sort_keys=True) + ("|" + json.dumps(json_body, sort_keys=True) if json_body is not None else "")
    path = _cache_path(key)
    if path.exists():
        return path.read_text()

    host = url.split("/")[2]
    last_error = "no attempt made"
    for attempt in range(retries):
        wait = pace_seconds - (time.time() - _last_call.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.time()
        try:
            r = requests.request(method, url, params={**params, **(extra or {})}, json=json_body, headers={**HEADERS, **(headers or {})}, timeout=60)
            body = r.text
            if r.status_code == 404 and not_found_ok:           # "we do not know this one" is a real answer: remember it
                body = ""
            elif not (r.status_code == 200 and body.strip() and (valid is None or valid(body))):
                last_error = f"HTTP {r.status_code}, {len(body)} bytes"
                if 400 <= r.status_code < 500 and r.status_code != 429:   # our request is wrong: retrying will not help
                    break
                body = None
            if body is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body)
                return body
        except requests.RequestException as e:                  # network problem, timeout, ...
            last_error = f"{type(e).__name__}: {e}"
        if "HTTP 429" in last_error and host == "export.arxiv.org":   # arXiv throttles for minutes: wait long
            pause = min(600, 30 * 2 ** attempt)                    # 30, 60, 120, 240, 480, 600 seconds
        elif "HTTP 429" in last_error:                             # "too many requests": the pool clears in seconds
            pause = min(20, 2 ** attempt)                          # 1, 2, 4, 8, 16, 20 seconds
        else:
            pause = min(60, 5 * 2 ** attempt)                      # 5, 10, 20, 40, 60, 60 seconds
        if attempt >= 2 or "HTTP 429" not in last_error:           # do not flood the log with routine 429 retries
            print(f"  [{host}] attempt {attempt + 1}/{retries} failed ({last_error}); waiting {pause}s", file=sys.stderr, flush=True)
        time.sleep(pause)
    raise RuntimeError(f"giving up on {url} after {retries} tries ({last_error})")


def get_text(url: str, params: dict | None = None, *, extra: dict | None = None, pace_seconds: float = 1.0,
             retries: int = 8, valid: Callable[[str], bool] | None = None, not_found_ok: bool = False,
             headers: dict | None = None) -> str:
    """GET `url` with `params` and return the body as text (cached on disk).

    extra        : parameters that are sent but NOT part of the cache key (e-mail address, API key).
    valid        : optional check on the body; if it fails, the answer is treated as an error and retried.
    not_found_ok : a 404 answer is accepted and cached as an empty body instead of raising.
    headers      : extra request headers (an API key), sent but NOT part of the cache key.
    """
    return _request("GET", url, params or {}, None, extra=extra, pace_seconds=pace_seconds, retries=retries,
                    valid=valid, not_found_ok=not_found_ok, headers=headers)


def get_json(url: str, params: dict | None = None, **kw):
    """Same as get_text, but the body is parsed as JSON. Returns None for a cached 404."""
    text = get_text(url, params, valid=lambda t: t.lstrip().startswith(("{", "[")), **kw)
    return json.loads(text) if text.strip() else None


def post_json(url: str, body: dict, params: dict | None = None, *, pace_seconds: float = 1.0, retries: int = 8,
              headers: dict | None = None):
    """POST a JSON body (used by Semantic Scholar's batch endpoint) and parse the JSON answer (cached on disk)."""
    text = _request("POST", url, params or {}, body, extra=None, pace_seconds=pace_seconds, retries=retries,
                    valid=lambda t: t.lstrip().startswith(("{", "[")), not_found_ok=False, headers=headers)
    return json.loads(text)
