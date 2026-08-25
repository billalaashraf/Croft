"""
webui/app.py — Local LLM Chat web app.

A complete local chat interface plus the original status/manager panel:

  * `/`          chat UI — talk to your installed models, pick a model + task
                 preset per conversation, full searchable history (SQLite).
  * `/manager`   the hardware / installed-models / services status panel.
  * `/api/*`     chat + management JSON APIs (see routes below).

Chat inference runs either through an embedded GGUF (llama-cpp-python) or any
OpenAI-compatible endpoint — see webui/inference.py. History is persisted by
webui/chatstore.py. No JS build step; the frontend is one self-contained file.

Run:  uvicorn webui.app:app --host 127.0.0.1 --port 8090
"""
from __future__ import annotations

import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from installer import hardware, recommend as rec, runtime  # noqa: E402
from webui import auth, chatstore, inference, imagegen, videogen, files  # noqa: E402

try:
    from fastapi import FastAPI, Request, File, UploadFile  # type: ignore
    from fastapi.responses import (HTMLResponse, JSONResponse,  # type: ignore
                                   StreamingResponse, FileResponse)
except Exception as exc:  # pragma: no cover
    raise SystemExit("FastAPI required: pip install fastapi uvicorn") from exc


@contextlib.asynccontextmanager
async def lifespan(_app):
    """Print the one URL that opens the app, then hand over to the server.

    A lifespan handler rather than `@app.on_event("startup")`: on_event is
    deprecated and warns on the FastAPI versions this project pins.

    The token rides in the fragment, and fragments are never sent to a server —
    so it reaches the page and goes no further, even though it is in a link.
    """
    host = os.environ.get("LLM_WEBUI_HOST", "127.0.0.1")
    port = os.environ.get("LLM_WEBUI_PORT", "8090")
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "*") else host
    print(f"[webui] open: http://{shown}:{port}/#t={auth.get_token()}")
    print(f"[webui] token file: {auth.TOKEN_FILE} (mode 0600)")
    yield


app = FastAPI(title="Local LLM Chat", version="2.0.0", lifespan=lifespan)
MODELS_DIR = os.environ.get("LLM_MODELS_DIR", "models")
_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# ---------------------------------------------------------------------------
# Gate every request: Host allow-list first, then the access token on /api.
#
# One middleware rather than two, because the order is load-bearing — a
# rebinding attempt must be refused before anything looks at its credentials —
# and a single function makes that order impossible to get wrong later. See
# webui/auth.py for what each gate actually stops.
# ---------------------------------------------------------------------------
@app.middleware("http")
async def guard(request: Request, call_next):
    if not auth.host_allowed(request.headers.get("host")):
        return JSONResponse(
            {"error": "unrecognised Host header — refusing to serve this "
                      "request. Reach the app on localhost, or list the "
                      "hostname in LLM_WEBUI_ALLOWED_HOSTS."},
            status_code=421)
    if request.url.path.startswith("/api/"):
        reason = auth.authorize(request)
        if reason:
            return JSONResponse({"error": reason, "unauthorized": True},
                                status_code=401)
    return await call_next(request)


# Task presets: each sets a system prompt + sampling temperature. Choosing a
# task is how you point a conversation at "an LLM for a specific job".
TASK_PRESETS = {
    "general":  {"label": "General",  "temperature": 0.7,
                 "system": "You are a helpful, concise assistant."},
    "coding":   {"label": "Coding",   "temperature": 0.2,
                 "system": "You are an expert programming assistant. Prefer correct, "
                           "idiomatic code with short explanations, in fenced code blocks."},
    "creative": {"label": "Creative", "temperature": 0.95,
                 "system": "You are a creative writing partner with a vivid, original voice."},
    "precise":  {"label": "Precise",  "temperature": 0.1,
                 "system": "You are a precise assistant. Answer factually and briefly. "
                           "If you are unsure, say so plainly."},
}

chatstore.init_db()

# Mint the token at import, before the socket is listening, so a supervisor
# (bootstrap_install.sh, webui.sh) can read .webui_token the moment the app
# answers and hand the user a URL that already works.
auth.get_token()


# ---------------------------------------------------------------------------
# Status helpers (unchanged behaviour, now under /manager + /api)
# ---------------------------------------------------------------------------
def _installed() -> dict:
    state_file = os.path.join(MODELS_DIR, "installed.json")
    if os.path.exists(state_file):
        with open(state_file, encoding="utf-8") as fh:
            return json.load(fh)
    return {"models": {}}


def _vram_usage() -> list:
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append({"gpu": i, "name": pynvml.nvmlDeviceGetName(h),
                        "used_gb": round(mem.used / 1024**3, 1),
                        "total_gb": round(mem.total / 1024**3, 1)})
        pynvml.nvmlShutdown()
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Chat: models & presets
# ---------------------------------------------------------------------------
@app.get("/api/models")
def api_models() -> JSONResponse:
    return JSONResponse({
        "models": inference.list_chat_models(),
        "presets": [{"id": k, "label": v["label"]} for k, v in TASK_PRESETS.items()],
        "ollama": inference.ollama_available(),
        "detected_endpoints": inference.detect_endpoints(),
    })


# ---------------------------------------------------------------------------
# Chat: conversations
# ---------------------------------------------------------------------------
@app.get("/api/conversations")
def api_conversations() -> JSONResponse:
    return JSONResponse({"conversations": chatstore.list_conversations()})


@app.post("/api/conversations")
async def api_create_conversation(request: Request) -> JSONResponse:
    body = await request.json()
    task = body.get("task") or "general"
    preset = TASK_PRESETS.get(task, TASK_PRESETS["general"])
    convo = chatstore.create_conversation(
        title=body.get("title") or "New chat",
        model_id=body.get("model_id"),
        task=task,
        system_prompt=preset["system"])
    return JSONResponse(convo)


@app.get("/api/conversations/{cid}")
def api_get_conversation(cid: str) -> JSONResponse:
    convo = chatstore.get_conversation(cid)
    if not convo:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"conversation": convo, "messages": chatstore.list_messages(cid)})


@app.patch("/api/conversations/{cid}")
async def api_update_conversation(cid: str, request: Request) -> JSONResponse:
    body = await request.json()
    # Switching task swaps the system prompt too.
    if body.get("task") in TASK_PRESETS:
        body["system_prompt"] = TASK_PRESETS[body["task"]]["system"]
    convo = chatstore.update_conversation(cid, **body)
    if not convo:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(convo)


@app.delete("/api/conversations/{cid}")
def api_delete_conversation(cid: str) -> JSONResponse:
    """Delete a chat, its messages, and the files that were attached to it."""
    deleted, paths = chatstore.delete_conversation(cid)
    removed = 0
    for path in paths:
        # Only unlink inside the uploads directory. These paths come from our
        # own rows, but a delete that follows a stored path anywhere on disk is
        # one bad row away from removing something it shouldn't.
        if not files.is_within_uploads(path):
            continue
        try:
            os.remove(path)
            removed += 1
        except FileNotFoundError:
            pass
        except OSError:
            pass
    return JSONResponse({"ok": deleted, "attachments_removed": removed})


# ---------------------------------------------------------------------------
# Chat: streaming completion
# ---------------------------------------------------------------------------
def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _with_attachments(text: str, atts: list) -> str:
    """Fold each attachment's extracted text into a user turn for the model."""
    parts = [text] if text else []
    for a in atts:
        full = chatstore.get_attachment(a["id"])
        name = (full or a).get("filename", "file")
        if not full or full.get("kind") == "image":
            parts.append(f"[Attached image: {name} — you cannot view images.]")
        elif full.get("kind") in ("binary", "error") or not full.get("text"):
            parts.append(f"[Attached file: {name} — no text could be extracted.]")
        else:
            note = " (truncated)" if full.get("truncated") else ""
            parts.append(f"[Attached file: {name}{note}]\n```\n{full['text']}\n```")
    return "\n\n".join(parts)


@app.post("/api/conversations/{cid}/chat")
async def api_chat(cid: str, request: Request):
    convo = chatstore.get_conversation(cid)
    if not convo:
        return JSONResponse({"error": "conversation not found"}, status_code=404)
    body = await request.json()
    content = (body.get("content") or "").strip()
    att_ids = body.get("attachment_ids") or []
    if not content and not att_ids:
        return JSONResponse({"error": "empty message"}, status_code=400)
    if not content:
        content = "Please analyse the attached file(s)."

    model_id = body.get("model_id") or convo.get("model_id")
    task = body.get("task") or convo.get("task") or "general"
    preset = TASK_PRESETS.get(task, TASK_PRESETS["general"])
    if not model_id:
        return JSONResponse({"error": "no model selected"}, status_code=400)

    # Resolve attachments (uploaded earlier via /api/attach) for this turn.
    atts = []
    for aid in att_ids:
        a = chatstore.get_attachment(aid)
        if a:
            atts.append({"id": a["id"], "filename": a["filename"], "kind": a["kind"],
                         "chars": a["chars"], "truncated": bool(a["truncated"])})

    # Persist choice + user message; auto-title the conversation on first turn.
    updates = {"model_id": model_id, "task": task,
               "system_prompt": preset["system"]}
    if (convo.get("title") or "New chat") == "New chat":
        base = content if content != "Please analyse the attached file(s)." \
            else (atts[0]["filename"] if atts else content)
        updates["title"] = (base[:48] + "…") if len(base) > 48 else base
    chatstore.update_conversation(cid, **updates)
    chatstore.add_message(cid, "user", content, model_id, attachments=atts)

    # Build the prompt: system + full prior history (which now includes this turn),
    # with each user turn's attached file text folded into that turn's content.
    history = chatstore.list_messages(cid)
    messages = [{"role": "system", "content": preset["system"]}]
    for m in history:
        if m["role"] not in ("user", "assistant"):
            continue
        text = m["content"]
        if m["role"] == "user" and m.get("attachments"):
            text = _with_attachments(text, m["attachments"])
        messages.append({"role": m["role"], "content": text})

    def gen():
        yield _sse({"type": "start", "model": model_id, "task": task})
        acc = []
        try:
            for piece in inference.stream_chat(
                    model_id, messages,
                    temperature=float(preset["temperature"]), max_tokens=1024):
                acc.append(piece)
                yield _sse({"type": "delta", "content": piece})
            text = "".join(acc)
            chatstore.add_message(cid, "assistant", text, model_id)
            yield _sse({"type": "done"})
        except Exception as exc:
            # Save whatever streamed so the turn isn't lost, then report.
            if acc:
                chatstore.add_message(cid, "assistant", "".join(acc), model_id)
            yield _sse({"type": "error", "error": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# File attachments (upload → extract text → attach to a chat turn)
# ---------------------------------------------------------------------------
@app.post("/api/attach")
async def api_attach(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    if len(data) > files.MAX_BYTES:
        return JSONResponse(
            {"error": f"file too large (> {files.MAX_BYTES // 1024 // 1024} MB)"},
            status_code=413)
    path = files.save_upload(data, file.filename or "file")
    text, kind, truncated = files.extract_text(path, file.filename or "file")
    row = chatstore.add_attachment(None, file.filename or "file", path, kind,
                                   len(text), truncated, text)
    note = {"image": "a text model can't analyse images",
            "binary": "unsupported file — no text extracted",
            "error": text}.get(kind)
    return JSONResponse({"id": row["id"], "filename": file.filename, "kind": kind,
                         "chars": len(text) if kind not in ("image", "binary") else 0,
                         "truncated": truncated, "note": note,
                         "supported": kind in ("text", "pdf", "docx")})


@app.get("/api/attachment/{aid}/file")
def api_attachment_file(aid: str):
    a = chatstore.get_attachment(aid)
    if not a or not a.get("path") or not os.path.exists(a["path"]):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(a["path"], filename=a["filename"])


# ---------------------------------------------------------------------------
# Settings (per-model / default OpenAI-compatible endpoints)
# ---------------------------------------------------------------------------
@app.get("/api/settings")
def api_get_settings() -> JSONResponse:
    return JSONResponse(chatstore.all_settings())


@app.post("/api/settings")
async def api_set_settings(request: Request) -> JSONResponse:
    body = await request.json()
    for key, value in body.items():
        if value in (None, ""):
            continue
        chatstore.set_setting(key, value)
    return JSONResponse(chatstore.all_settings())


# ---------------------------------------------------------------------------
# Image generation (text-to-image)
# ---------------------------------------------------------------------------
@app.get("/api/image/models")
def api_image_models() -> JSONResponse:
    return JSONResponse({"models": imagegen.list_image_models(),
                         "deps": imagegen.deps_available(),
                         "device": imagegen.device()})


@app.post("/api/image/generate")
async def api_image_generate(request: Request) -> JSONResponse:
    body = await request.json()
    model_id = body.get("model_id")
    prompt = (body.get("prompt") or "").strip()
    if not model_id:
        return JSONResponse({"error": "no image model selected"}, status_code=400)
    if not prompt:
        return JSONResponse({"error": "empty prompt"}, status_code=400)
    try:
        # Blocking generation runs in FastAPI's threadpool (sync def would too,
        # but we're async here, so hand it off explicitly).
        import anyio
        res = await anyio.to_thread.run_sync(
            lambda: imagegen.generate(
                model_id, prompt,
                negative_prompt=body.get("negative_prompt"),
                steps=body.get("steps"), width=body.get("width"),
                height=body.get("height"), guidance=body.get("guidance"),
                seed=body.get("seed")))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    rec_row = chatstore.add_image(model_id, prompt, body.get("negative_prompt"),
                                  res["params"], res["path"])
    return JSONResponse({"image": _image_public(rec_row)})


@app.get("/api/image/history")
def api_image_history() -> JSONResponse:
    return JSONResponse({"images": [_image_public(i) for i in chatstore.list_images()]})


@app.get("/api/image/file/{iid}")
def api_image_file(iid: str):
    img = chatstore.get_image(iid)
    if not img or not os.path.exists(img["path"]):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(img["path"], media_type="image/png")


@app.delete("/api/image/{iid}")
def api_image_delete(iid: str) -> JSONResponse:
    path = chatstore.delete_image(iid)
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    return JSONResponse({"ok": path is not None})


def _image_public(row: dict) -> dict:
    """Shape a DB row for the client (never leak the absolute path)."""
    return {"id": row["id"], "model_id": row.get("model_id"),
            "prompt": row["prompt"], "negative_prompt": row.get("negative_prompt"),
            "params": row.get("params", {}), "created_at": row["created_at"],
            "url": f"/api/image/file/{row['id']}"}


# ---------------------------------------------------------------------------
# Video generation (text-to-video / image-to-video)
# ---------------------------------------------------------------------------
@app.get("/api/video/models")
def api_video_models() -> JSONResponse:
    return JSONResponse({"models": videogen.list_video_models(),
                         "deps": videogen.deps_available(),
                         "device": videogen.device()})


@app.post("/api/video/generate")
async def api_video_generate(request: Request) -> JSONResponse:
    body = await request.json()
    model_id = body.get("model_id")
    if not model_id:
        return JSONResponse({"error": "no video model selected"}, status_code=400)
    # img2vid takes a source image uploaded via /api/attach (we reuse its saved
    # path). The client sends an attachment id, never a path — and the path we
    # look up is confirmed to sit inside the uploads directory before it is
    # handed to the worker, so a doctored row cannot aim the pipeline at an
    # arbitrary file.
    source = None
    if body.get("source_image_id"):
        a = chatstore.get_attachment(body["source_image_id"])
        candidate = a["path"] if a else None
        if candidate and not files.is_within_uploads(candidate):
            return JSONResponse({"error": "invalid source image"}, status_code=400)
        source = candidate
    try:
        import anyio
        res = await anyio.to_thread.run_sync(
            lambda: videogen.generate(
                model_id, prompt=(body.get("prompt") or None), source_image=source,
                frames=body.get("frames"), fps=body.get("fps"), steps=body.get("steps")))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    row = chatstore.add_video(model_id, body.get("prompt"), res["params"],
                              res["path"], res["fmt"])
    return JSONResponse({"video": _video_public(row)})


@app.get("/api/video/history")
def api_video_history() -> JSONResponse:
    return JSONResponse({"videos": [_video_public(v) for v in chatstore.list_videos()]})


@app.get("/api/video/file/{vid}")
def api_video_file(vid: str):
    v = chatstore.get_video(vid)
    if not v or not os.path.exists(v["path"]):
        return JSONResponse({"error": "not found"}, status_code=404)
    mt = "video/mp4" if v.get("fmt") == "mp4" else "image/gif"
    return FileResponse(v["path"], media_type=mt)


@app.delete("/api/video/{vid}")
def api_video_delete(vid: str) -> JSONResponse:
    path = chatstore.delete_video(vid)
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    return JSONResponse({"ok": path is not None})


def _video_public(row: dict) -> dict:
    return {"id": row["id"], "model_id": row.get("model_id"), "prompt": row.get("prompt"),
            "params": row.get("params", {}), "fmt": row.get("fmt"),
            "created_at": row["created_at"], "url": f"/api/video/file/{row['id']}"}


# ---------------------------------------------------------------------------
# Management APIs (from the original panel)
# ---------------------------------------------------------------------------
@app.get("/api/hardware")
def api_hardware() -> JSONResponse:
    return JSONResponse(hardware.detect_all())


@app.get("/api/recommend")
def api_recommend() -> JSONResponse:
    return JSONResponse(rec.recommend(hardware.detect_all()))


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse({"installed": _installed()["models"],
                         "vram": _vram_usage(),
                         "docker": runtime.docker_status()})


@app.post("/api/service/{action}/{service}")
def api_service(action: str, service: str) -> JSONResponse:
    if action not in ("up", "down"):
        return JSONResponse({"error": "action must be up|down"}, status_code=400)
    rc = runtime.docker_up(service) if action == "up" else runtime.docker_down()
    return JSONResponse({"ok": rc == 0, "action": action, "service": service})


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get("/manager", response_class=HTMLResponse)
def manager() -> str:
    hw = hardware.detect_all()
    installed = _installed()["models"]
    vram = _vram_usage()
    rows = "".join(
        f"<tr><td>{mid}</td><td>{m['kind']}</td><td>{m['format']}</td>"
        f"<td><code>{m['path']}</code></td></tr>"
        for mid, m in installed.items()) or "<tr><td colspan=4>none</td></tr>"
    vrows = "".join(
        f"<tr><td>{v['name']}</td><td>{v['used_gb']}/{v['total_gb']} GB</td></tr>"
        for v in vram) or "<tr><td colspan=2>CPU-only / no NVML</td></tr>"
    gpus = ", ".join(g["name"] for g in hw["gpus"]) or "none"
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>Local LLM Chat — Manager</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:2rem;max-width:900px;color:#111}}
 a{{color:#2563eb}} h1{{font-size:1.4rem}}
 table{{border-collapse:collapse;width:100%;margin:1rem 0}}
 td,th{{border:1px solid #ddd;padding:.4rem .6rem;text-align:left;font-size:.9rem}}
 th{{background:#f5f5f5}} code{{font-size:.8rem}}
 .pill{{background:#eef;padding:.2rem .5rem;border-radius:6px;font-size:.8rem}}
</style></head><body>
<p><a href="/">← Back to chat</a></p>
<h1>Local LLM Chat — Manager</h1>
<p><span class=pill>OS: {hw['os']}</span> <span class=pill>accel: {hw['accelerator']}</span>
 <span class=pill>RAM: {hw['ram_total_gb']} GB</span>
 <span class=pill>VRAM: {hw['total_vram_gb']} GB</span>
 <span class=pill>disk free: {hw['disk_free_gb']} GB</span></p>
<p>GPUs: {gpus}</p>
<h2>GPU memory</h2><table><tr><th>GPU</th><th>used/total</th></tr>{vrows}</table>
<h2>Installed models</h2>
<table><tr><th>id</th><th>kind</th><th>format</th><th>path</th></tr>{rows}</table>
<h2>APIs</h2>
<p>JSON: <a href=/api/hardware>/api/hardware</a> ·
 <a href=/api/recommend>/api/recommend</a> · <a href=/api/status>/api/status</a></p>
</body></html>"""


if __name__ == "__main__":  # pragma: no cover
    import uvicorn  # type: ignore
    uvicorn.run(app, host="127.0.0.1", port=8090)
