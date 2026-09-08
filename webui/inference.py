# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""
webui/inference.py — model registry + chat inference for the web UI.

Chat runs out-of-process, over an OpenAI-compatible HTTP endpoint. Either:

  1. A configured endpoint (set per-model or globally in Settings), or
  2. Ollama, auto-detected on its default port and used for any installed model
     that has been imported into it.

The model used to run in-process through `llama-cpp-python`. That was the
cheapest thing to build and the most expensive thing to own: chat, image and
video weights all landed in one uvicorn process sharing one pool of unified
memory, and nothing gave memory back until the process died. A separate server
is the only arrangement where freeing actually frees. Ollama additionally drops
an idle model on its own (OLLAMA_KEEP_ALIVE, default 5m) and honours an explicit
unload, which is what `unload()` below asks for when the diffusers pipelines
need the room.

`stream_chat()` yields text deltas. Only `requests` is needed.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

from webui import gpumem

MODELS_DIR = os.environ.get("LLM_MODELS_DIR", "models")

# Common local OpenAI-compatible servers, probed for auto-detection.
CANDIDATE_ENDPOINTS = [
    ("llama.cpp", "http://127.0.0.1:8080/v1"),
    ("vllm",      "http://127.0.0.1:8000/v1"),
    ("tgi",       "http://127.0.0.1:8081/v1"),
    ("ollama",    "http://127.0.0.1:11434/v1"),
]

# Ollama speaks OpenAI on /v1 and its own dialect on /api. We use the first for
# chat and the second for the lifecycle calls OpenAI has no concept of.
OLLAMA_HOST = os.environ.get("LLM_OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

# How long unload() waits for Ollama to actually give the weights back.
UNLOAD_WAIT_S = float(os.environ.get("LLM_OLLAMA_UNLOAD_WAIT", "8"))
# Grace period after Ollama reports nothing loaded, for the runner to exit and
# its pages to come back to the OS.
UNLOAD_SETTLE_S = float(os.environ.get("LLM_OLLAMA_UNLOAD_SETTLE", "1.5"))


# ---------------------------------------------------------------------------
# Installed-model registry
# ---------------------------------------------------------------------------
def _installed() -> Dict[str, dict]:
    state = os.path.join(MODELS_DIR, "installed.json")
    if os.path.exists(state):
        try:
            with open(state, encoding="utf-8") as fh:
                return json.load(fh).get("models", {})
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def _ollama_get(path: str, timeout: float = 1.5) -> Optional[dict]:
    if requests is None:
        return None
    try:
        r = requests.get(OLLAMA_HOST + path, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def ollama_available() -> bool:
    return _ollama_get("/api/version") is not None


def ollama_tags() -> List[str]:
    """Every model name Ollama holds, e.g. ['qwen2.5-7b-q4:latest', …]."""
    data = _ollama_get("/api/tags")
    if not data:
        return []
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def ollama_name_for(model_id: str) -> Optional[str]:
    """Map an installed model id onto an Ollama tag.

    Import with `ollama create <model_id> -f Modelfile` and the names line up;
    the `:latest` suffix Ollama appends is matched here so they still do.
    """
    tags = ollama_tags()
    for tag in tags:
        if tag == model_id or tag.split(":")[0] == model_id:
            return tag
    return None


def unload() -> None:
    """Evict every model Ollama currently holds, and wait for it to take effect.

    keep_alive=0 on a prompt-less generate is Ollama's documented "drop it now".
    We ask for each loaded model by name rather than tracking what we loaded,
    so a model someone else started is freed too — on 32 GB shared with two
    diffusers pipelines, an idle 5 GB resident elsewhere is still 5 GB gone.

    The wait is not a formality. The POST returns once Ollama has *scheduled* the
    unload, not once the weights are gone, and the caller's next move is to load
    a multi-GiB pipeline. Returning early means both models are resident at the
    same moment, which is the OOM this whole path exists to avoid. So poll
    /api/ps until it drains, and give up after a few seconds rather than block a
    request forever on a model that refuses to leave.
    """
    if requests is None:
        return
    running = _ollama_get("/api/ps") or {}
    asked = False
    for m in running.get("models", []):
        name = m.get("name") or m.get("model")
        if not name:
            continue
        try:
            requests.post(OLLAMA_HOST + "/api/generate",
                          json={"model": name, "keep_alive": 0}, timeout=30)
            asked = True
        except Exception:
            continue
    if not asked:
        return
    deadline = time.monotonic() + UNLOAD_WAIT_S
    while time.monotonic() < deadline:
        still = (_ollama_get("/api/ps") or {}).get("models") or []
        if not still:
            # /api/ps goes empty when Ollama drops its bookkeeping, but the
            # runner subprocess is still exiting and its pages are still
            # charged to the machine — measured 1-2s of lag before host-free
            # memory actually rises. Returning here would hand the caller a
            # budget that has not arrived yet, which is the same
            # too-early-by-a-moment mistake this function was fixing.
            time.sleep(UNLOAD_SETTLE_S)
            return
        time.sleep(0.25)


# ---------------------------------------------------------------------------
# Endpoint validation
# ---------------------------------------------------------------------------
# An endpoint is not an innocent string. Every message in a conversation is
# POSTed to it, and LLM_CHAT_API_KEY is attached as a bearer token — so a value
# pointing at a host you do not control exfiltrates both the chat history and
# the key. It also reaches anything the machine can reach: cloud metadata on
# 169.254.169.254, a container-network neighbour, an intranet service.
#
# The default is therefore loopback-only. Reaching a model server on another
# machine is a legitimate thing to want, so it is available — but as a
# deliberate opt-in that has to be set in the environment, not as something the
# settings API can turn on by itself.
ALLOW_REMOTE = os.environ.get("LLM_ALLOW_REMOTE_ENDPOINT", "").strip() not in ("", "0")


def validate_endpoint(url: str) -> str:
    """Return `url` if it is safe to send chats and credentials to, else raise.

    Resolution happens here, at validation time, and the result is checked —
    a name that resolves into a private range is rejected on its address, not
    on how it is spelled.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    u = urlparse((url or "").strip())
    if u.scheme not in ("http", "https"):
        raise ValueError("endpoint must start with http:// or https://")
    if not u.hostname:
        raise ValueError("endpoint has no host")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or
                                   (443 if u.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"endpoint host does not resolve ({exc})") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_loopback:
            continue
        if not ALLOW_REMOTE:
            raise ValueError(
                f"endpoint resolves to {ip}, which is not loopback. Croft sends "
                "your conversation and LLM_CHAT_API_KEY to this address, so it "
                "refuses non-local endpoints unless you set "
                "LLM_ALLOW_REMOTE_ENDPOINT=1.")
        # Even opted in, the addresses that exist to be reached accidentally
        # stay refused: link-local carries cloud instance credentials.
        if ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise ValueError(f"endpoint resolves to {ip}, which is never a model server")
    return url.strip()


def _endpoint_for(model_id: str) -> Optional[str]:
    """Configured base URL for a model: per-model setting, else global default.

    Re-validated on read as well as on write. A row can predate the check, be
    edited in the database directly, or survive a downgrade — and this is the
    last point before credentials go out over the wire.
    """
    from webui import chatstore
    raw = (chatstore.get_setting(f"endpoint:{model_id}")
           or chatstore.get_setting("default_endpoint"))
    if not raw:
        return None
    try:
        return validate_endpoint(str(raw))
    except ValueError as exc:
        print(f"[inference] refusing stored endpoint {raw!r}: {exc}", file=sys.stderr)
        return None


def resolve_route(model_id: str) -> Dict[str, object]:
    """Return how a model will be run: {type, ready, detail, ...}."""
    meta = _installed().get(model_id)
    if not meta:
        return {"type": "none", "ready": False, "detail": "not installed"}
    if meta.get("kind") != "text":
        return {"type": "none", "ready": False,
                "detail": f"{meta.get('kind')} models are not chat-capable"}

    endpoint = _endpoint_for(model_id)
    if endpoint:
        return {"type": "openai", "ready": True, "endpoint": endpoint,
                "model": model_id,
                "detail": f"OpenAI-compatible endpoint {endpoint}"}

    if ollama_available():
        tag = ollama_name_for(model_id)
        if tag:
            return {"type": "openai", "ready": True, "model": tag,
                    "endpoint": OLLAMA_HOST + "/v1",
                    "detail": f"ollama · {tag}"}
        return {"type": "openai", "ready": False,
                "detail": f"not in ollama yet — run: ollama create {model_id} -f Modelfile"}
    return {"type": "openai", "ready": False,
            "detail": "start ollama, or set an OpenAI-compatible endpoint in Settings"}


def list_chat_models() -> List[Dict[str, object]]:
    out = []
    for mid, meta in _installed().items():
        route = resolve_route(mid)
        out.append({
            "id": mid,
            "kind": meta.get("kind"),
            "format": meta.get("format"),
            "chat_capable": meta.get("kind") == "text",
            "ready": bool(route.get("ready")),
            "route": route.get("type"),
            "detail": route.get("detail"),
        })
    # Chat-capable and ready first, then by id.
    out.sort(key=lambda m: (not m["chat_capable"], not m["ready"], m["id"]))
    return out


def detect_endpoints() -> List[Dict[str, str]]:
    """Probe common local servers; return the ones that answer /models."""
    found = []
    if requests is None:
        return found
    for name, base in CANDIDATE_ENDPOINTS:
        try:
            r = requests.get(f"{base}/models", timeout=1.5)
            if r.status_code < 500:
                found.append({"name": name, "endpoint": base})
        except Exception:
            continue
    return found


# The chat weights now live in Ollama's process, so this frees for real rather
# than dropping a reference and hoping the allocator agrees.
gpumem.register("chat", unload)


# ---------------------------------------------------------------------------
# OpenAI-compatible proxy (streaming SSE)
# ---------------------------------------------------------------------------
def _stream_openai(endpoint: str, model_id: str, messages: List[dict],
                   temperature: float, max_tokens: int) -> Iterator[str]:
    if requests is None:
        raise RuntimeError("`requests` is required to reach an OpenAI-compatible endpoint")
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = {"model": model_id, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "stream": True}
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("LLM_CHAT_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Redirects off. A validated loopback endpoint that answers 302 would
    # otherwise relocate this request — bearer token and full conversation
    # included — to a host the check never saw.
    with requests.post(url, json=payload, headers=headers, stream=True,
                       timeout=300, allow_redirects=False) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                piece = obj.get("choices", [{}])[0].get("delta", {}).get("content")
            except Exception:
                continue
            if piece:
                yield piece


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------
def stream_chat(model_id: str, messages: List[dict], *,
                temperature: float = 0.7, max_tokens: int = 1024) -> Iterator[str]:
    """Yield assistant text deltas for the given model + message history.

    Raises RuntimeError (with a human-readable reason) if the model can't run.
    """
    route = resolve_route(model_id)
    if not route.get("ready"):
        raise RuntimeError(route.get("detail") or "model not runnable")
    # `model` is the name the server knows, which is not always our id: Ollama
    # appends `:latest` to anything imported without an explicit tag.
    yield from _stream_openai(str(route["endpoint"]), str(route.get("model") or model_id),
                              messages, temperature, max_tokens)
