"""
webui/gpumem.py — shared accelerator budget for the three model families.

Chat (llama.cpp GGUF), image and video (diffusers) all live in the same uvicorn
process and all allocate on the same device, but each keeps its own permanent
cache and none of them knows the others exist. On Apple Silicon that device is
unified memory shared with the OS, so a resident SDXL pipeline plus a resident
7B GGUF plus an AnimateDiff load is enough to pass the MPS high-watermark and
push the machine into swap.

Each module registers an `unload` callback under an owner name. Before a heavy
load, it calls `free_except(<its own name>, need_gib=...)`; if the device is
short of that much headroom, every *other* owner drops its cached models first.
Freeing the Python reference is only half the job — the Metal/CUDA buffers come
back after a gc pass plus an explicit cache flush, which `empty_cache()` does.

Set LLM_GPU_EVICT=0 to disable eviction (keeps everything resident, and is how
you get the old behaviour back on a machine with memory to spare).
"""
from __future__ import annotations

import gc
import os
from typing import Callable, Dict, Optional

GIB = 1024 ** 3

# Rows of the attention score matrix computed per chunk when slicing. See
# tune_pipeline() for why 32; lower it if a longer clip still OOMs.
ATTN_SLICE_ROWS = int(os.environ.get("LLM_ATTN_SLICE_ROWS", "32"))

_OWNERS: Dict[str, Callable[[], None]] = {}


def register(owner: str, unload: Callable[[], None]) -> None:
    """Record how to drop `owner`'s cached models. Called once, at import."""
    _OWNERS[owner] = unload


def _torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def _device() -> str:
    """Accelerator this process would use: 'cuda', 'mps' or 'cpu'."""
    torch = _torch()
    if torch is None:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def empty_cache() -> None:
    """Collect garbage, then hand the freed device buffers back to the driver."""
    gc.collect()
    torch = _torch()
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def os_available_bytes() -> Optional[int]:
    """Physical memory the OS could hand out right now, or None.

    psutil reports this per-machine, which is the whole point: it sees the model
    Ollama is holding in another process, and the MPS counters do not.
    """
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def headroom_bytes() -> Optional[int]:
    """Memory still available for a load, or None when it can't be determined.

    On MPS this is the recommended working set minus what the driver has handed
    out — *not* the high-watermark ceiling, which sits at 1.7x the recommended
    set and is already deep into swap territory on a shared-memory machine.

    That figure alone is not enough, and trusting it is what made eviction dead
    code. `driver_allocated_memory()` counts only *this process's* allocations,
    so a freshly-started worker reports the full ~25 GiB free on a 32 GB machine
    no matter how much Ollama and everything else are holding — the check was
    structurally blind to the cross-process contention it exists to prevent. On
    unified memory the device pool *is* system RAM, so bound it by what the OS
    actually has free and take the smaller number. Ollama's own scheduler makes
    the same call for the same reason.

    CUDA is deliberately exempt: discrete VRAM is not host RAM, and clamping it
    to system free memory would evict pipelines a dedicated GPU has room for.
    """
    torch = _torch()
    if torch is None:
        return None
    try:
        if torch.cuda.is_available():
            return int(torch.cuda.mem_get_info()[0])
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_free = max(0, int(torch.mps.recommended_max_memory()
                                     - torch.mps.driver_allocated_memory()))
            host_free = os_available_bytes()
            return device_free if host_free is None else min(device_free, host_free)
    except Exception:
        pass
    return None


def free_except(owner: str, need_gib: float = 0.0) -> None:
    """Drop every other owner's models if the device lacks `need_gib` of room.

    A no-op when there's headroom, so chatting doesn't evict a pipeline you're
    about to use again. When headroom can't be read we evict anyway: an
    unnecessary reload costs seconds, an OOM costs the request.
    """
    if os.environ.get("LLM_GPU_EVICT", "1") == "0":
        return
    free = headroom_bytes()
    if free is not None and free >= need_gib * GIB:
        return
    for name, unload in _OWNERS.items():
        if name == owner:
            continue
        try:
            unload()
        except Exception:
            pass
    empty_cache()


def _force_sliced_attention(pipe) -> bool:
    """Install SlicedAttnProcessor when the pipeline's own slicing didn't take.

    `enable_attention_slicing()` walks the model for children exposing
    `set_attention_slice`. `UNetMotionModel` exposes no such hook, so the call
    returns cleanly and changes nothing — every processor stays `AttnProcessor2_0`.
    That silent no-op is why video OOM'd for weeks while the code looked correct.

    Setting the processor directly is the part that actually works. Returns
    whether the pipeline is now sliced, so callers can stop guessing.
    """
    unet = getattr(pipe, "unet", None)
    if unet is None or not hasattr(unet, "set_attn_processor"):
        return False
    try:
        from diffusers.models.attention_processor import SlicedAttnProcessor
        # Rows of the (batch*heads, tokens, tokens) score matrix per chunk. At
        # 512x512 each row is ~32 MB, so 32 caps a chunk near 1 GiB — the peak
        # drops from 8 GiB to under 4, without the 256-chunk crawl slice_size=1
        # would impose.
        unet.set_attn_processor(SlicedAttnProcessor(slice_size=ATTN_SLICE_ROWS))
        return True
    except Exception:
        return False


def tune_pipeline(pipe, *, slice_attention: bool = False) -> None:
    """Trade a little speed for a lot of peak memory on a diffusers pipeline.

    `slice_attention` is for video only, and only pays off on MPS. AnimateDiff
    inflates the UNet batch to `num_frames` and classifier-free guidance doubles
    it again, so spatial self-attention allocates
    `frames*2 * heads * tokens^2` — at 16 frames and 512x512 that is exactly
    8 GiB in one tensor, the single largest allocation in the run and the one
    that OOMs. Slicing computes it in chunks instead.

    Diffusers warns against slicing when SDPA is active, because SDPA is
    normally memory-efficient. That advice does not hold on MPS: the traceback
    puts the 8 GiB allocation *inside* `F.scaled_dot_product_attention`, so the
    backend is materializing the full matrix rather than using a fused kernel.
    On CUDA the warning stands and flash attention does the job, which is why
    this is gated on the device.

    Image pipelines take the `enable_attention_slicing()` path, which on a plain
    `UNet2DConditionModel` genuinely works — so they end up sliced too. At batch
    1-2 that buys little and costs some speed; it is longstanding behaviour and
    left alone deliberately rather than changed while chasing the video bug.
    VAE slicing/tiling caps the decode peak for both.
    """
    if slice_attention and _device() == "mps":
        _force_sliced_attention(pipe)
    else:
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    # diffusers 0.40 moves the VAE toggles onto the VAE itself and deprecates
    # the pipeline-level aliases, so prefer the new spelling and fall back.
    vae = getattr(pipe, "vae", None)
    for new, old in (("enable_slicing", "enable_vae_slicing"),
                     ("enable_tiling", "enable_vae_tiling")):
        if vae is not None and hasattr(vae, new):
            try:
                getattr(vae, new)()
                continue
            except Exception:
                pass
        try:
            getattr(pipe, old)()
        except Exception:
            pass
