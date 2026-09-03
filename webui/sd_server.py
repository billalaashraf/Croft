# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""
webui/sd_server.py — the diffusion worker: image and video, in its own process.

This exists for one reason. On a unified-memory Mac, dropping a Python
reference to a pipeline does not reliably hand the Metal buffers back to the
OS; exiting a process always does. So the app keeps the UI and the database,
and this worker keeps every model weight. If memory goes wrong, you restart
this and nothing else.

It reuses `imagegen`/`videogen` rather than reimplementing them: those modules
contain the model registry, the dtype and slicing choices, and the output
paths, and having two copies of that logic drift apart is how you end up
generating at a different precision depending on which process ran the job.
The worker calls their `_generate_local()`; the app calls their `generate()`,
which forwards here. Same code, one hop apart.

Files are not returned over HTTP. Both processes share LLM_MODELS_DIR, so the
worker writes to sd-outputs and returns the path.

Run: python -m webui.sd_server   (or ./webui.sh start, which supervises it)
Env: LLM_SD_HOST (127.0.0.1), LLM_SD_PORT (7862).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

try:
    from fastapi import FastAPI  # type: ignore
    from fastapi.responses import JSONResponse  # type: ignore
    from pydantic import BaseModel  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("Install: fastapi uvicorn pydantic") from exc

# Importing inference here is not an oversight: it registers the chat owner
# with gpumem, so a job in this process can evict the model sitting in Ollama.
# Without it the worker would only know how to free its own two pipelines.
from webui import files, gpumem, imagegen, inference, videogen  # noqa: F401

app = FastAPI(title="Local diffusion worker")

# One job at a time, and not merely to be polite about the GPU. FastAPI runs
# these sync endpoints in a threadpool, so two jobs would share `_PIPES` and
# each would call gpumem.free_except() on entry — job B evicting the very
# pipeline job A is mid-denoise on. That surfaced as an intermittent 500 under
# nothing more exotic than clicking generate twice. There is one accelerator
# here; queueing is the honest model.
_JOB = threading.Lock()


class ImageJob(BaseModel):
    model_id: str
    prompt: str
    negative_prompt: Optional[str] = None
    steps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None


class VideoJob(BaseModel):
    model_id: str
    prompt: Optional[str] = None
    source_image: Optional[str] = None
    frames: Optional[int] = None
    fps: Optional[int] = None
    steps: Optional[int] = None


@app.get("/health")
def health() -> dict:
    """Cheap enough to call on every model-list request in the app.

    The `_local` variants are mandatory here. The public `deps_available()` asks
    the worker over HTTP, which inside the worker means asking itself: every
    /health spawns a nested /health until the threadpool is full and the process
    stops answering anything. Local checks only, on this endpoint.
    """
    return {"ok": True, "deps": imagegen._deps_local(),
            "device": imagegen._device_local(), "pid": os.getpid()}


@app.post("/generate/image")
def generate_image(job: ImageJob) -> JSONResponse:
    with _JOB:
        try:
            return JSONResponse(imagegen._generate_local(
                job.model_id, job.prompt, negative_prompt=job.negative_prompt,
                steps=job.steps, width=job.width, height=job.height,
                guidance=job.guidance, seed=job.seed))
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/generate/video")
def generate_video(job: VideoJob) -> JSONResponse:
    # `source_image` is the one field here that is a filesystem path rather
    # than a number or a prompt. The app already checks it, but this worker
    # listens on its own port and will take the job from anything that can
    # reach it, so it checks for itself rather than trusting its caller.
    if job.source_image and not files.is_within_uploads(job.source_image):
        return JSONResponse(
            {"error": "source_image must be a file under the uploads directory"},
            status_code=400)
    with _JOB:
        try:
            return JSONResponse(videogen._generate_local(
                job.model_id, prompt=job.prompt, source_image=job.source_image,
                frames=job.frames, fps=job.fps, steps=job.steps))
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/unload")
def unload() -> dict:
    """Drop every pipeline. The app has no other way to make this process
    give memory back short of killing it, which loses the warm start too."""
    for owner in ("image", "video"):
        try:
            gpumem._OWNERS[owner]()
        except Exception:
            pass
    gpumem.empty_cache()
    return {"ok": True}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn  # type: ignore
    uvicorn.run(app, host=os.environ.get("LLM_SD_HOST", "127.0.0.1"),
                port=int(os.environ.get("LLM_SD_PORT", "7862")))
