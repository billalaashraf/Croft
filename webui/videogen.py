"""
webui/videogen.py — text-to-video and image-to-video generation.

Two families, chosen from the installed model:
  * AnimateDiff (txt2vid) — a motion adapter over an SD1.5 base; prompt → clip.
  * Stable Video Diffusion (img2vid) — an input image → clip.

`generate()` hands the job to the diffusion worker when one is running, so the
weights land in that process and not in the web app; `_generate_local()` is the
half that actually loads a pipeline, and is what the worker calls. With no
worker up, `generate()` runs it here instead.

Heavy deps (torch + diffusers) are optional and imported lazily; missing pieces
degrade to a clear message rather than a crash. Clips are written under
<LLM_MODELS_DIR>/sd-outputs as mp4 (via imageio-ffmpeg) or gif fallback, and
recorded in the chat DB so the gallery persists.
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Dict, List, Optional

from webui import gpumem, sdclient

MODELS_DIR = os.environ.get("LLM_MODELS_DIR", "models")
OUT_DIR = os.path.join(MODELS_DIR, "sd-outputs")
# SD1.5 base AnimateDiff rides on; overridable if you host it locally.
SD15_BASE = os.environ.get("LLM_ANIMATEDIFF_BASE", "runwayml/stable-diffusion-v1-5")

_PIPES: Dict[str, object] = {}

# Loading a clip pipeline is the heaviest thing this process does; ask the
# shared budget for room before it, and give the room back on request.
NEED_GIB = float(os.environ.get("LLM_VIDEO_NEED_GIB", "10"))
gpumem.register("video", _PIPES.clear)


# ---------------------------------------------------------------------------
# Availability + registry
# ---------------------------------------------------------------------------
def _deps_local() -> bool:
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return True
    except Exception:
        return False


def _device_local() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def deps_available() -> bool:
    """Worker's answer when there is one; see imagegen for why."""
    info = sdclient.health()
    return bool(info.get("deps")) if info is not None else _deps_local()


def device() -> str:
    info = sdclient.health()
    return str(info.get("device") or "cpu") if info is not None else _device_local()


def _installed() -> Dict[str, dict]:
    state = os.path.join(MODELS_DIR, "installed.json")
    if os.path.exists(state):
        try:
            with open(state, encoding="utf-8") as fh:
                return json.load(fh).get("models", {})
        except Exception:
            return {}
    return {}


def video_kind(model_id: str) -> str:
    """'img2vid' for Stable-Video-Diffusion-style models, else 'txt2vid'."""
    mid = model_id.lower()
    return "img2vid" if ("svd" in mid or "video-diffusion" in mid or "img2vid" in mid) else "txt2vid"


def max_frames(model_id: str) -> int:
    """Longest clip this model's weights can actually represent.

    Not a memory budget — a hard architectural limit. AnimateDiff bakes a
    temporal positional embedding of fixed length (`motion_max_seq_length`, 32
    for the SD1.5 adapter) into the motion module, so asking for more frames
    than that fails deep in the UNet with a bare shape mismatch: "The size of
    tensor a (60) must match the size of tensor b (32) at non-singleton
    dimension 1". SVD carries the same idea as `num_frames` on its UNet.

    Read from the model on disk rather than hardcoded, so a different adapter
    with a longer embedding raises the ceiling by itself.
    """
    meta = _installed().get(model_id, {})
    path = meta.get("path", os.path.join(MODELS_DIR, model_id))
    if video_kind(model_id) == "img2vid":
        cfg, key, fallback = os.path.join(path, "unet", "config.json"), "num_frames", 25
    else:
        cfg, key, fallback = os.path.join(path, "config.json"), "motion_max_seq_length", 32
    try:
        with open(cfg, encoding="utf-8") as fh:
            val = int(json.load(fh).get(key) or fallback)
            return val if val > 0 else fallback
    except Exception:
        return fallback


def default_params(model_id: str) -> Dict[str, object]:
    cap = max_frames(model_id)
    if video_kind(model_id) == "img2vid":
        return {"frames": min(14, cap), "fps": 7, "steps": 20, "max_frames": cap}
    return {"frames": min(16, cap), "fps": 8, "steps": 20, "max_frames": cap}


def list_video_models() -> List[Dict[str, object]]:
    have = deps_available()
    out = []
    for mid, meta in _installed().items():
        if meta.get("kind") != "video":
            continue
        path = meta.get("path", os.path.join(MODELS_DIR, mid))
        on_disk = os.path.isdir(path)
        ready = bool(have and on_disk)
        vk = video_kind(mid)
        if not have:
            detail = "install torch + diffusers to enable video generation"
        elif not on_disk:
            detail = "model files missing on disk"
        else:
            detail = f"diffusers · {vk} · {device()}"
        out.append({"id": mid, "format": meta.get("format"), "ready": ready,
                    "detail": detail, "video_kind": vk, "defaults": default_params(mid)})
    out.sort(key=lambda m: (not m["ready"], m["id"]))
    return out


# ---------------------------------------------------------------------------
# Export frames → mp4 (imageio-ffmpeg) or gif fallback
# ---------------------------------------------------------------------------
def _export(frames, fps: int) -> tuple:
    os.makedirs(OUT_DIR, exist_ok=True)
    import numpy as np
    arr = [np.asarray(f) for f in frames]
    base = os.path.join(OUT_DIR, uuid.uuid4().hex)
    try:
        import imageio
        path = base + ".mp4"
        imageio.mimsave(path, arr, fps=fps, codec="libx264",
                        macro_block_size=None)   # needs imageio-ffmpeg
        return path, "mp4"
    except Exception:
        import imageio
        path = base + ".gif"
        imageio.mimsave(path, arr, duration=1.0 / max(fps, 1))
        return path, "gif"


# ---------------------------------------------------------------------------
# Pipelines (lazy, cached)
# ---------------------------------------------------------------------------
def _dtype():
    """half on any accelerator, full on CPU. Override with LLM_SD_DTYPE.

    MPS handles float16 fine, and half precision here is worth more than the
    weights it saves: the frame batch multiplies every activation, so full
    precision on a shared-memory device is what tips a clip over the watermark.
    """
    import torch
    want = os.environ.get("LLM_SD_DTYPE", "float16" if _device_local() != "cpu" else "float32")
    return getattr(torch, want, torch.float32)


def _txt2vid_pipe(model_id: str):
    key = "adiff:" + model_id
    if key in _PIPES:
        return _PIPES[key]
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    meta = _installed().get(model_id, {})
    adapter_path = meta.get("path", os.path.join(MODELS_DIR, model_id))
    dev = _device_local()
    dtype = _dtype()
    adapter = MotionAdapter.from_pretrained(adapter_path, torch_dtype=dtype)
    pipe = AnimateDiffPipeline.from_pretrained(SD15_BASE, motion_adapter=adapter,
                                               torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config,
                                               beta_schedule="linear")
    pipe = pipe.to(dev)
    gpumem.tune_pipeline(pipe, slice_attention=True)
    _PIPES[key] = pipe
    return pipe


def _img2vid_pipe(model_id: str):
    key = "svd:" + model_id
    if key in _PIPES:
        return _PIPES[key]
    from diffusers import StableVideoDiffusionPipeline
    meta = _installed().get(model_id, {})
    path = meta.get("path", os.path.join(MODELS_DIR, model_id))
    dev = _device_local()
    # `variant` stays cuda-only: a local checkout may hold only fp32 weight
    # files, and from_pretrained casts them to `dtype` on load either way.
    pipe = StableVideoDiffusionPipeline.from_pretrained(path, torch_dtype=_dtype(),
                                                        variant="fp16" if dev == "cuda" else None)
    pipe = pipe.to(dev)
    gpumem.tune_pipeline(pipe, slice_attention=True)
    _PIPES[key] = pipe
    return pipe


def generate(model_id: str, *, prompt: Optional[str] = None,
             source_image: Optional[str] = None, frames: Optional[int] = None,
             fps: Optional[int] = None, steps: Optional[int] = None) -> Dict[str, object]:
    """Generate one clip; return {path, fmt, params}.

    Forwards to the worker when one is up, otherwise runs here. See imagegen.
    """
    if sdclient.available():
        return sdclient.post("/generate/video", {
            "model_id": model_id, "prompt": prompt, "source_image": source_image,
            "frames": frames, "fps": fps, "steps": steps})
    return _generate_local(model_id, prompt=prompt, source_image=source_image,
                           frames=frames, fps=fps, steps=steps)


def _generate_local(model_id: str, *, prompt: Optional[str] = None,
                    source_image: Optional[str] = None, frames: Optional[int] = None,
                    fps: Optional[int] = None, steps: Optional[int] = None) -> Dict[str, object]:
    """Generate one clip in THIS process; return {path, fmt, params}."""
    if not _deps_local():
        raise RuntimeError("torch + diffusers not installed "
                           "(pip install torch diffusers transformers accelerate "
                           "imageio imageio-ffmpeg)")
    meta = _installed().get(model_id)
    if not meta or meta.get("kind") != "video":
        raise RuntimeError(f"'{model_id}' is not an installed video model")

    d = default_params(model_id)
    frames = int(frames or d["frames"])
    fps = int(fps or d["fps"])
    steps = int(steps or d["steps"])
    vk = video_kind(model_id)

    # Check before loading anything: the alternative is a bare tensor-shape
    # error raised deep inside the UNet, minutes into a job, naming neither
    # "frames" nor the model.
    cap = max_frames(model_id)
    if frames > cap:
        raise RuntimeError(
            f"'{model_id}' supports at most {cap} frames ({cap / max(fps, 1):.1f}s "
            f"at {fps} fps); asked for {frames}")
    if frames < 1:
        raise RuntimeError(f"frames must be at least 1, got {frames}")

    # Before the pipeline *and* before the run: a cached pipe still needs room
    # for its activations, which are the larger half of a clip's footprint.
    gpumem.free_except("video", need_gib=NEED_GIB)

    if vk == "img2vid":
        if not source_image or not os.path.exists(source_image):
            raise RuntimeError("image-to-video needs a source image")
        from diffusers.utils import load_image
        image = load_image(source_image).resize((1024, 576))
        pipe = _img2vid_pipe(model_id)
        result = pipe(image, num_frames=frames, num_inference_steps=steps, fps=fps)
        clip = result.frames[0]
    else:
        if not prompt:
            raise RuntimeError("text-to-video needs a prompt")
        pipe = _txt2vid_pipe(model_id)
        result = pipe(prompt=prompt, num_frames=frames, num_inference_steps=steps)
        clip = result.frames[0]

    path, fmt = _export(clip, fps)
    return {"path": path, "fmt": fmt,
            "params": {"frames": frames, "fps": fps, "steps": steps, "kind": vk}}
