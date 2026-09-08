# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
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
import sys
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
    # Direct single-file URL. When set, the download is fetched over plain HTTP
    # with resume and verified against `integrity_sha256` — the only path that
    # actually checks a hash we published rather than trusting the hub.
    download_url: Optional[str] = None
    integrity_sha256: Optional[str] = None
    # Glob filter for snapshot downloads. A GGUF repo holds every quantisation,
    # so pulling one unfiltered fetches many times the advertised size.
    hf_allow_patterns: Optional[List[str]] = None
    # Commit SHA to download. `main` moves and can be force-pushed, so an
    # unpinned snapshot means two people installing the same model id weeks
    # apart can get different weights. None falls back to "main".
    revision: Optional[str] = None

    def required_gb(self) -> float:
        mult = OVERHEAD_MULT
        return estimate_required_gb(self.params_billion, self.precision, mult)


# GGUF repos publish every quantisation side by side, so a snapshot has to be
# filtered down to the one we advertise. Matching is case-sensitive and the
# repos disagree on case (`Q4_K_M` vs `q4_k_m`), hence both spellings.
GGUF_Q4_PATTERNS = ["*Q4_K_M*.gguf", "*q4_k_m*.gguf", "*.json"]

# Curated, permissive-first defaults. HF repo IDs are real, widely-used repos.
DEFAULT_CATALOG: List[ModelEntry] = [
    # ---- TEXT (CPU-friendly GGUF quantized) ----
    ModelEntry("qwen2.5-3b-q4", "Qwen/Qwen2.5-3B-Instruct-GGUF", "text", 3.0,
               "int4", "gguf", "Apache-2.0", False, 2.1, "cpu-only",
               "Great small CPU/edge model",
               hf_allow_patterns=GGUF_Q4_PATTERNS),
    ModelEntry("llama3.2-3b-q4", "bartowski/Llama-3.2-3B-Instruct-GGUF", "text",
               3.2, "int4", "gguf", "Llama-3.2-Community", True, 2.2, "cpu-only",
               "Gated: accept Meta license on HF first",
               hf_allow_patterns=GGUF_Q4_PATTERNS),
    ModelEntry("mistral-7b-q4", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "text",
               7.0, "int4", "gguf", "Apache-2.0", False, 4.1, "cpu-only",
               "Strong 7B, runs on CPU or small GPU",
               hf_allow_patterns=GGUF_Q4_PATTERNS),
    ModelEntry("qwen2.5-7b-q4", "Qwen/Qwen2.5-7B-Instruct-GGUF", "text", 7.6,
               "int4", "gguf", "Apache-2.0", False, 4.7, "native",
               hf_allow_patterns=GGUF_Q4_PATTERNS),
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
               "docker", "Gated; multi-GPU or CPU offload recommended",
               hf_allow_patterns=GGUF_Q4_PATTERNS),
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

# Immutable commit SHAs for every entry above.
#
# models_manifest.json pins its own entries, but the manifest does not name
# every id in this catalog — `qwen2.5-3b-q4`, `llama3.2-3b-q4`,
# `mistral-7b-gptq`, `qwen2.5-32b-gptq` and `controlnet-sdxl` exist only here.
# Without a pin those fell through to `main`, which moves and can be
# force-pushed, so repeated installs of the same id could fetch different
# weights with nothing to indicate it.
#
# Kept as a table rather than a constructor argument for two reasons: the
# entries above are already dense with positional fields, and a table can be
# checked for completeness — `test_recommend_catalog.py` fails if any snapshot
# entry is missing from it, so a new model cannot be added unpinned.
DEFAULT_REVISIONS: Dict[str, str] = {
    "qwen2.5-3b-q4":       "7dabda4d13d513e3e842b20f0d435c732f172cbe",
    "llama3.2-3b-q4":      "5ab33fa94d1d04e903623ae72c95d1696f09f9e8",
    "mistral-7b-q4":       "3a6fbf4a41a1d52e415a4958cde6856d34b2db93",
    "qwen2.5-7b-q4":       "bb5d59e06d9551d752d08b292a50eb208b07ab1f",
    "mistral-7b-gptq":     "7532d6bc89ef9300fb39d2d94ed4414ec534b72a",
    "llama3.1-8b-fp16":    "0e9e39f249a16976918f6564b8830bc894c89659",
    "mixtral-8x7b-gptq":   "0f81ba4680ccd2bce163334b93305d40b9e27b09",
    "qwen2.5-32b-gptq":    "c83e67dfb2664f5039fd4cd99e206799e27dd800",
    "llama3.3-70b-q4":     "b6c5c9f176f3279204034e1d16d393105e95cb88",
    "sd-1.5":              "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    "sdxl-1.0":            "462165984030d82259a11f4367a4eed129e94a7b",
    "sdxl-turbo":          "71153311d3dbb46851df1931d3ca6e939de83304",
    "controlnet-sdxl":     "eb115a19a10d14909256db740ed109532ab1483c",
    "svd-xt":              "9e43909513c6714f1bc78bcb44d96e733cd242aa",
    "animatediff-sd15":    "6167b88ffe39b4441fdf2113e77b99a6f56b7906",
}

for _entry in DEFAULT_CATALOG:
    _entry.revision = DEFAULT_REVISIONS.get(_entry.id)
del _entry


def load_manifest(path: str = "models_manifest.json") -> List[ModelEntry]:
    """Merge models_manifest.json entries into the default catalog (by id)."""
    catalog = {m.id: m for m in DEFAULT_CATALOG}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for e in data.get("models", []):
                fmt = e.get("format", "safetensors")
                patterns = e.get("hf_allow_patterns")
                # A GGUF entry with no filter would snapshot every quant in the
                # repo. Fall back to the standard Q4 filter rather than quietly
                # downloading ten times what the entry advertises.
                if patterns is None and fmt == "gguf":
                    patterns = list(GGUF_Q4_PATTERNS)
                catalog[e["id"]] = ModelEntry(
                    id=e["id"], hf_repo=e.get("hf_repo") or e.get(
                        "source_url", "").replace("https://huggingface.co/", ""),
                    kind=e.get("kind", "text"),
                    params_billion=float(e.get("params", 0)) / 1e9
                    if e.get("params", 0) > 1e6 else float(e.get("params", 0)),
                    precision=(e.get("quantized_formats_available") or ["fp16"])[0]
                    if isinstance(e.get("quantized_formats_available"), list) else "fp16",
                    format=fmt,
                    license=e.get("license", "unknown"),
                    gated=e.get("gated", False),
                    approx_disk_gb=float(e.get("size_bytes", 0)) / (1024 ** 3),
                    recommended_mode=e.get("recommended_mode", "native"),
                    notes=e.get("notes", ""),
                    download_url=e.get("download_url"),
                    integrity_sha256=e.get("integrity_sha256"),
                    hf_allow_patterns=patterns,
                    # Falling back to the built-in pin matters: a manifest entry
                    # that omits `revision` would otherwise replace a pinned
                    # default with None and quietly reopen the `main` hole.
                    revision=(e.get("revision")
                              or DEFAULT_REVISIONS.get(e["id"])),
                )
        except Exception as exc:
            # Falling back to the built-in catalog is right — a bad manifest
            # should not brick the installer. Doing it silently is not: adding
            # a custom model is a documented workflow, and a typo there used to
            # look exactly like an entry that simply never appeared.
            print(f"[recommend] warning: could not read {path} ({exc}); "
                  f"using the built-in catalog. Custom models in that file "
                  f"will not be listed until it parses.", file=sys.stderr)
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
