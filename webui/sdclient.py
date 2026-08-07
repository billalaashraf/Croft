"""
webui/sdclient.py — the web app's side of the diffusion worker link.

The worker (webui/sd_server.py) owns torch, diffusers and every pipeline. This
module is what the app uses instead, and it deliberately imports none of that:
if the app process never imports torch, it can never hold a model, which is the
whole reason the worker exists.

Both processes share a filesystem and the same LLM_MODELS_DIR, so a generated
file does not travel over HTTP. The worker writes the PNG or MP4 and returns
its path; the app records that path in the DB. Nothing is copied or encoded.

When no worker is reachable the callers fall back to generating in-process, so
a plain `uvicorn webui.app:app` with no worker still works — it just goes back
to holding the weights itself.
"""
from __future__ import annotations

import os
from typing import Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

SD_URL = os.environ.get("LLM_SD_URL", "http://127.0.0.1:7862").rstrip("/")

# Generation is slow and wildly variable (a turbo image in seconds, a clip in
# many minutes), so the read timeout is long. Connect stays short: "is there a
# worker" must be answered fast on every model-list request.
CONNECT_TIMEOUT = 1.0
READ_TIMEOUT = float(os.environ.get("LLM_SD_TIMEOUT", "3600"))


def health() -> Optional[dict]:
    """The worker's {deps, device}, or None when there's no worker."""
    if requests is None:
        return None
    try:
        r = requests.get(f"{SD_URL}/health", timeout=CONNECT_TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def available() -> bool:
    return health() is not None


def post(path: str, payload: dict) -> dict:
    """Run one job on the worker. Raises RuntimeError with the worker's reason."""
    if requests is None:
        raise RuntimeError("`requests` is required to reach the diffusion worker")
    r = requests.post(f"{SD_URL}{path}", json=payload,
                      timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    if r.status_code != 200:
        try:
            detail = r.json().get("error") or r.text
        except Exception:
            detail = r.text
        raise RuntimeError(detail.strip() or f"diffusion worker returned {r.status_code}")
    return r.json()
