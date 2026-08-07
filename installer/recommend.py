"""
recommend.py — Model compatibility engine.

Maps a hardware capability report (from hardware.py) to a ranked list of
models that will actually run, using a param-count -> memory formula:

    weights_bytes  = params * bytes_per_param(precision)
    runtime_bytes  = weights_bytes * OVERHEAD_MULT      # KV cache + activations
    required_gb    = runtime_bytes / 1024**3 + HEADROOM_GB

bytes_per_param:  fp32=4, fp16/bf16=2, int8≈1, int4≈0.5

The catalog below is a *default* set. The authoritative, richer list lives in
models_manifest.json; `load_manifest()` merges it in when present so users can
extend recommendations without editing code.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --- Estimation constants ----------------------------------------------------
BYTES_PER_PARAM = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5}
OVERHEAD_MULT = 1.5      # KV-cache + activation working set (×). Use 2.0 for long ctx.
HEADROOM_GB = 1.0        # fixed cushion for framework/driver allocations
CPU_RAM_OVERHEAD_MULT = 1.25   # CPU inference is lighter on transient buffers


def bytes_per_param(precision: str) -> float:
    return BYTES_PER_PARAM.get(precision.lower(), 2.0)


def estimate_required_gb(params_billion: float, precision: str,
                         overhead_mult: float = OVERHEAD_MULT,
                         headroom_gb: float = HEADROOM_GB) -> float:
    """Return estimated VRAM (or RAM for CPU) in GiB for a model."""
    params = params_billion * 1e9
    weights = params * bytes_per_param(precision)
    runtime = weights * overhead_mult
    return round(runtime / (1024 ** 3) + headroom_gb, 1)


# --- Model catalog -----------------------------------------------------------
@dataclass
class ModelEntry:
    id: str
    hf_repo: str
    kind: str                 # "text" | "image" | "video"
    params_billion: float
    precision: str            # native precision of the listed weights
    format: str               # "gguf" | "gptq" | "safetensors" | "diffusers"
    license: str
    gated: bool = False       # requires HF license acceptance / token
    approx_disk_gb: float = 0.0
    recommended_mode: str = "native"   # "docker" | "native" | "cpu-only"
    notes: str = ""

    def required_gb(self) -> float:
        mult = OVERHEAD_MULT
        return estimate_required_gb(self.params_billion, self.precision, mult)


# Curated, permissive-first defaults. HF repo IDs are real, widely-used repos.
DEFAULT_CATALOG: List[ModelEntry] = [
    # ---- TEXT (CPU-friendly GGUF quantized) ----
    ModelEntry("qwen2.5-3b-q4", "Qwen/Qwen2.5-3B-Instruct-GGUF", "text", 3.0,
               "int4", "gguf", "Apache-2.0", False, 2.1, "cpu-only",
               "Great small CPU/edge model"),
    ModelEntry("llama3.2-3b-q4", "bartowski/Llama-3.2-3B-Instruct-GGUF", "text",
               3.2, "int4", "gguf", "Llama-3.2-Community", True, 2.2, "cpu-only",
               "Gated: accept Meta license on HF first"),
    ModelEntry("mistral-7b-q4", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "text",
               7.0, "int4", "gguf", "Apache-2.0", False, 4.1, "cpu-only",
               "Strong 7B, runs on CPU or small GPU"),
    ModelEntry("qwen2.5-7b-q4", "Qwen/Qwen2.5-7B-Instruct-GGUF", "text", 7.6,
               "int4", "gguf", "Apache-2.0", False, 4.7, "native"),
    # ---- TEXT (GPU GPTQ / fp16) ----
    ModelEntry("mistral-7b-gptq", "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
               "text", 7.0, "int4", "gptq", "Apache-2.0", False, 4.2, "native",
               "4-bit GPTQ for CUDA"),
    ModelEntry("llama3.1-8b-fp16", "meta-llama/Llama-3.1-8B-Instruct", "text",
               8.0, "fp16", "safetensors", "Llama-3.1-Community", True, 16.1,
               "docker", "Gated: accept Meta license + HF token"),
    ModelEntry("mixtral-8x7b-gptq", "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
               "text", 46.7, "int4", "gptq", "Apache-2.0", False, 24.0, "docker",
               "MoE; ~13B active params but full weights must be resident"),
    ModelEntry("qwen2.5-32b-gptq", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", "text",
               32.0, "int4", "gptq", "Apache-2.0", False, 19.0, "docker"),
    ModelEntry("llama3.3-70b-q4", "bartowski/Llama-3.3-70B-Instruct-GGUF",
               "text", 70.0, "int4", "gguf", "Llama-3.3-Community", True, 40.0,
               "docker", "Gated; multi-GPU or CPU offload recommended"),
    # ---- IMAGE ----
    ModelEntry("sd-1.5", "runwayml/stable-diffusion-v1-5", "image", 0.98, "fp16",
               "diffusers", "CreativeML-OpenRAIL-M", False, 4.3, "native",
               "512px baseline, ~4GB VRAM"),
    ModelEntry("sdxl-1.0", "stabilityai/stable-diffusion-xl-base-1.0", "image",
               3.5, "fp16", "diffusers", "CreativeML-OpenRAIL++-M", False, 13.0,
               "native", "1024px; refiner optional; ~8-10GB VRAM"),
    ModelEntry("sdxl-turbo", "stabilityai/sdxl-turbo", "image", 3.5, "fp16",
               "diffusers", "STAI-Community", False, 13.0, "native",
               "1-4 step fast generation"),
    ModelEntry("controlnet-sdxl", "diffusers/controlnet-canny-sdxl-1.0", "image",
               1.3, "fp16", "diffusers", "OpenRAIL", False, 5.0, "native",
               "Add-on; VRAM stacks on top of base SDXL"),
    # ---- VIDEO ----
    ModelEntry("svd-xt", "stabilityai/stable-video-diffusion-img2vid-xt",
               "video", 1.5, "fp16", "diffusers", "STAI-NC-Community", True, 9.5,
               "docker", "img2vid, 25 frames; non-commercial license"),
    ModelEntry("animatediff-sd15", "guoyww/animatediff-motion-adapter-v1-5-2",
               "video", 1.4, "fp16", "diffusers", "Apache-2.0", False, 3.0,
               "native", "Motion module for SD1.5 text2video"),
]


def load_manifest(path: str = "models_manifest.json") -> List[ModelEntry]:
    """Merge models_manifest.json entries into the default catalog (by id)."""
    catalog = {m.id: m for m in DEFAULT_CATALOG}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for e in data.get("models", []):
                catalog[e["id"]] = ModelEntry(
                    id=e["id"], hf_repo=e.get("source_url", "").replace(
                        "https://huggingface.co/", ""),
                    kind=e.get("kind", "text"),
                    params_billion=float(e.get("params", 0)) / 1e9
                    if e.get("params", 0) > 1e6 else float(e.get("params", 0)),
                    precision=(e.get("quantized_formats_available") or ["fp16"])[0]
                    if isinstance(e.get("quantized_formats_available"), list) else "fp16",
                    format=e.get("format", "safetensors"),
                    license=e.get("license", "unknown"),
                    gated=e.get("gated", False),
                    approx_disk_gb=float(e.get("size_bytes", 0)) / (1024 ** 3),
                    recommended_mode=e.get("recommended_mode", "native"),
                    notes=e.get("notes", ""),
                )
        except Exception:
            pass  # malformed manifest -> silently fall back to defaults
    return list(catalog.values())


# --- Tier classification -----------------------------------------------------
def classify_tier(report: Dict[str, Any]) -> str:
    """Bucket the machine into a coarse capability tier."""
    vram = report.get("total_vram_gb") or 0.0
    ram = report.get("ram_available_gb") or report.get("ram_total_gb") or 0.0
    accel = report.get("accelerator", "cpu")
    ngpu = len(report.get("gpus", []))
    if ngpu >= 2 and vram >= 40:
        return "multi-gpu"
    if vram >= 24:
        return "high-end"
    if vram >= 8:
        return "mid-range"
    if accel == "mps":
        return "apple-silicon"
    return "cpu-only"


def recommend(report: Dict[str, Any],
              manifest_path: str = "models_manifest.json",
              kinds: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Return a structured recommendation: tier, per-kind ranked compatible models,
    and per-model fit metrics. A model is "fits" if its required memory is
    within the available VRAM budget (or RAM budget on CPU/MPS).
    """
    catalog = load_manifest(manifest_path)
    tier = classify_tier(report)
    vram = report.get("total_vram_gb") or 0.0
    ram = report.get("ram_available_gb") or report.get("ram_total_gb") or 0.0
    disk = report.get("disk_free_gb") or 0.0
    accel = report.get("accelerator", "cpu")
    cpu_only = accel in ("cpu", "mps") and vram < 4

    results: Dict[str, List[Dict[str, Any]]] = {"text": [], "image": [], "video": []}
    for m in catalog:
        if kinds and m.kind not in kinds:
            continue
        # Budget rules:
        #   * GGUF runs anywhere; when VRAM can't hold it (or there's no GPU)
        #     it falls back to CPU/MPS and is measured against system RAM.
        #   * GPTQ / safetensors / diffusers need a real accelerator: they are
        #     measured against VRAM only. On a GPU-less host (vram==0) they
        #     cannot fit, no matter how much RAM is free.
        on_cpu = m.format == "gguf" and (cpu_only or vram < m.required_gb())
        if on_cpu:
            required = estimate_required_gb(
                m.params_billion, m.precision, CPU_RAM_OVERHEAD_MULT)
            budget = ram
            device = "mps" if accel == "mps" else "cpu"
        else:
            required = m.required_gb()
            # MPS exposes a VRAM budget (unified memory); pure-CPU exposes none.
            budget = vram
            device = accel if accel != "cpu" else "cpu"

        fits_mem = required <= budget + 0.001
        fits_disk = m.approx_disk_gb <= disk or disk == 0
        headroom = round(budget - required, 1)
        entry = {
            "id": m.id,
            "hf_repo": m.hf_repo,
            "params_billion": m.params_billion,
            "precision": m.precision,
            "format": m.format,
            "license": m.license,
            "gated": m.gated,
            "device": device,
            "required_memory_gb": required,
            "approx_disk_gb": m.approx_disk_gb,
            "fits_memory": fits_mem,
            "fits_disk": fits_disk,
            "fits": bool(fits_mem and fits_disk),
            "memory_headroom_gb": headroom,
            "recommended_mode": "cpu-only" if on_cpu else m.recommended_mode,
            "notes": m.notes,
        }
        results.setdefault(m.kind, []).append(entry)

    # Rank: fitting models first, then by capability (params) desc, then headroom.
    for kind in results:
        results[kind].sort(
            key=lambda e: (e["fits"], e["params_billion"], e["memory_headroom_gb"]),
            reverse=True,
        )

    default_text = next((e["id"] for e in results["text"] if e["fits"]), None)
    return {
        "tier": tier,
        "accelerator": accel,
        "vram_gb": vram,
        "ram_gb": ram,
        "disk_free_gb": disk,
        "default_text_model": default_text,
        "recommendations": results,
        "formula": {
            "bytes_per_param": BYTES_PER_PARAM,
            "overhead_mult": OVERHEAD_MULT,
            "headroom_gb": HEADROOM_GB,
            "example": "7B int4 => 7e9*0.5*1.5/1024^3 + 1 ≈ 5.9 GB",
        },
    }


if __name__ == "__main__":  # pragma: no cover
    import sys
    from installer.hardware import detect_all
    rep = detect_all()
    print(json.dumps(recommend(rep), indent=2))
