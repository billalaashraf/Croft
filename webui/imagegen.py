# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""
webui/imagegen.py — text-to-image generation for the web app.

Loads an installed diffusers image model (e.g. sdxl-turbo) locally and runs
text→image, entirely on-device. Heavy deps (torch + diffusers) are optional and
imported lazily; when they're absent the app degrades gracefully and the UI
shows a one-line "install these" hint instead of failing.

`generate()` hands the job to the diffusion worker when one is running, so the
weights land in that process and not in the web app; `_generate_local()` is the
half that actually loads a pipeline, and is what the worker calls. With no
worker up, `generate()` runs it here instead.

Output PNGs are written to <LLM_MODELS_DIR>/sd-outputs and recorded in the
chat DB (webui/chatstore.py) so the gallery persists across restarts.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from webui import gpumem, sdclient

MODELS_DIR = os.environ.get("LLM_MODELS_DIR", "models")
OUT_DIR = os.path.join(MODELS_DIR, "sd-outputs")

_PIPES: Dict[str, object] = {}

NEED_GIB = float(os.environ.get("LLM_IMAGE_NEED_GIB", "8"))
# fp32 holds the same weights at twice the width, so a job that upcasts needs a
# correspondingly bigger budget asked of gpumem before it loads.
NEED_GIB_FP32 = float(os.environ.get("LLM_IMAGE_NEED_GIB_FP32", str(NEED_GIB * 2)))

# Largest edge float16 survives on MPS. Above this the SDXL UNet overflows
# mid-denoise and every latent comes out NaN — see _dtype_for() for the detail.
FP16_MAX_EDGE = int(os.environ.get("LLM_SD_FP16_MAX_EDGE", "512"))

gpumem.register("image", _PIPES.clear)


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
    """Answered by the worker when there is one, so the app needn't import torch.

    Importing torch into the web app would pull a few hundred MB of framework
    into the one process meant to stay light, purely to answer a yes/no question
    for the model list. The `_local` variants below are what the worker itself
    uses, so it never asks itself over HTTP.
    """
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


def is_turbo(model_id: str) -> bool:
    mid = model_id.lower()
    return "turbo" in mid or "lightning" in mid or "lcm" in mid


def default_params(model_id: str) -> Dict[str, object]:
    """Sensible defaults per model family (turbo = few steps, no guidance)."""
    if is_turbo(model_id):
        return {"steps": 4, "guidance": 0.0, "width": 512, "height": 512}
    return {"steps": 25, "guidance": 6.0, "width": 768, "height": 768}


def list_image_models() -> List[Dict[str, object]]:
    have = deps_available()
    out = []
    for mid, meta in _installed().items():
        if meta.get("kind") != "image":
            continue
        path = meta.get("path", os.path.join(MODELS_DIR, mid))
        on_disk = os.path.isdir(path)
        ready = bool(have and on_disk)
        if not have:
            detail = "install torch + diffusers to enable image generation"
        elif not on_disk:
            detail = "model files missing on disk"
        else:
            detail = f"diffusers · {device()}"
        out.append({"id": mid, "format": meta.get("format"), "ready": ready,
                    "detail": detail, "defaults": default_params(mid)})
    out.sort(key=lambda m: (not m["ready"], m["id"]))
    return out


# ---------------------------------------------------------------------------
# Pipeline (lazy, cached — a model is loaded once and kept resident)
# ---------------------------------------------------------------------------
def _dtype_for(width: int, height: int):
    """Precision to load at for this output size.

    float16 is the right default on an accelerator, but on MPS it silently
    breaks above 512px: the SDXL UNet overflows during denoising and returns
    latents that are 100% NaN, which the image processor casts to uint8 as a
    fully black PNG. Measured on this machine at seed 3 — 512 clean, 768 and
    1024 NaN in every pixel, while the identical run in float32 came back clean
    with a healthy latent absmax of 3.2.

    This is *not* the well-known SDXL fp16-VAE bug. The VAE config already sets
    `force_upcast`, so the decode runs in float32; pulling the latents before
    the VAE shows them already NaN. The overflow is upstream, in the UNet.

    So above FP16_MAX_EDGE on MPS, load float32 instead. CUDA is left alone —
    fp16 SDXL at 1024 is the standard there and does not overflow. An explicit
    LLM_SD_DTYPE always wins, since someone setting it is making the call
    themselves; the non-finite guard in _generate_local still catches the fallout.
    """
    import torch
    dev = _device_local()
    want = os.environ.get("LLM_SD_DTYPE")
    if want is None:
        want = "float32" if dev == "cpu" else "float16"
        if dev == "mps" and max(width, height) > FP16_MAX_EDGE:
            want = "float32"
    return getattr(torch, want, torch.float32)


def _pipe(model_id: str, dtype=None):
    """Load (and cache) the pipeline for `model_id` at `dtype`.

    Keyed by dtype because the same model at two precisions is two different
    resident copies. Only one is kept: holding an fp16 and an fp32 SDXL at once
    is ~15 GB, so switching precision drops the other rather than stacking them.
    Alternating between 512 and 768 therefore pays a reload, which is the honest
    trade against a black image or an OOM.
    """
    import torch
    from diffusers import AutoPipelineForText2Image

    if dtype is None:
        dtype = _dtype_for(FP16_MAX_EDGE, FP16_MAX_EDGE)
    key = f"{model_id}:{str(dtype).rsplit('.', 1)[-1]}"
    if key in _PIPES:
        return _PIPES[key]
    for stale in [k for k in _PIPES if k.split(":")[0] == model_id]:
        _PIPES.pop(stale, None)
    gpumem.empty_cache()

    meta = _installed().get(model_id, {})
    path = meta.get("path", os.path.join(MODELS_DIR, model_id))
    dev = _device_local()
    pipe = AutoPipelineForText2Image.from_pretrained(
        path, torch_dtype=dtype, use_safetensors=True)
    pipe = pipe.to(dev)
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    # SDXL upcasts its VAE to float32 to decode; slicing keeps that peak down.
    gpumem.tune_pipeline(pipe)
    _PIPES[key] = pipe
    return pipe


def generate(model_id: str, prompt: str, *, negative_prompt: Optional[str] = None,
             steps: Optional[int] = None, width: Optional[int] = None,
             height: Optional[int] = None, guidance: Optional[float] = None,
             seed: Optional[int] = None) -> Dict[str, object]:
    """Generate one image and return {path, params}.

    Runs on the worker when one is up, in-process when it isn't. Both paths end
    at `_generate_local` and return the same shape, so the caller can't tell.
    """
    if sdclient.available():
        return sdclient.post("/generate/image", {
            "model_id": model_id, "prompt": prompt,
            "negative_prompt": negative_prompt, "steps": steps,
            "width": width, "height": height, "guidance": guidance, "seed": seed})
    return _generate_local(model_id, prompt, negative_prompt=negative_prompt,
                           steps=steps, width=width, height=height,
                           guidance=guidance, seed=seed)


def _generate_local(model_id: str, prompt: str, *, negative_prompt: Optional[str] = None,
                    steps: Optional[int] = None, width: Optional[int] = None,
                    height: Optional[int] = None, guidance: Optional[float] = None,
                    seed: Optional[int] = None) -> Dict[str, object]:
    """Generate one image in THIS process, save it as PNG, return {path, params}."""
    if not _deps_local():
        raise RuntimeError("torch + diffusers not installed "
                           "(pip install torch diffusers safetensors pillow accelerate)")
    meta = _installed().get(model_id)
    if not meta or meta.get("kind") != "image":
        raise RuntimeError(f"'{model_id}' is not an installed image model")

    d = default_params(model_id)
    steps = int(steps or d["steps"])
    width = int(width or d["width"])
    height = int(height or d["height"])
    guidance = float(guidance if guidance is not None else d["guidance"])

    import numpy as np
    import torch

    def _run(dtype):
        need = NEED_GIB_FP32 if dtype == torch.float32 else NEED_GIB
        gpumem.free_except("image", need_gib=need)
        pipe = _pipe(model_id, dtype)
        gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None
        out = pipe(prompt=prompt, negative_prompt=negative_prompt or None,
                   num_inference_steps=steps, guidance_scale=guidance,
                   width=width, height=height, generator=gen,
                   output_type="np")
        return out.images[0]

    # Never write a corrupt image. A NaN run is not a slightly-worse picture: the
    # processor casts NaN to uint8 and the result is a uniformly black PNG that
    # used to be saved, recorded in the gallery and returned as 200 OK, so the
    # only evidence anything failed was a RuntimeWarning in the worker log.
    #
    # float16 on MPS goes non-finite *intermittently*. Size raises the odds
    # sharply — above 512 it is the common case, which is why _dtype_for()
    # upcasts there — but 512 is not immune: the one blank PNG this bug left
    # behind on disk is 512x512 with every pixel exactly 0. Since float32 has
    # never reproduced it, a single retry at full precision turns an
    # intermittent black image into a slower correct one, and anything still
    # non-finite after that is raised rather than saved.
    dtype = _dtype_for(width, height)
    arr = _run(dtype)
    if not np.isfinite(arr).all() and dtype != torch.float32:
        arr = _run(torch.float32)
        dtype = torch.float32
    if not np.isfinite(arr).all():
        raise RuntimeError(
            f"generation produced non-finite pixels at {width}x{height} even in "
            "float32 — the image would have been blank; try a smaller size.")
    from PIL import Image
    image = Image.fromarray((arr * 255).round().astype("uint8"))

    os.makedirs(OUT_DIR, exist_ok=True)
    import uuid
    fname = f"{uuid.uuid4().hex}.png"
    fpath = os.path.join(OUT_DIR, fname)
    image.save(fpath, format="PNG")
    return {"path": fpath,
            "params": {"steps": steps, "width": width, "height": height,
                       "guidance": guidance, "seed": seed}}
