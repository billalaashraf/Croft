# Hardware → Model Compatibility

## The estimation formula

Memory needed to run a model is dominated by resident weights plus a transient
working set (KV cache + activations). The installer uses:

```
weights_bytes = params × bytes_per_param(precision)
runtime_bytes = weights_bytes × OVERHEAD_MULT      # KV cache + activations
required_GB   = runtime_bytes / 1024³ + HEADROOM_GB
```

| precision | bytes/param |
|-----------|-------------|
| FP32      | 4.0         |
| FP16/BF16 | 2.0         |
| INT8      | ~1.0        |
| INT4      | ~0.5        |

Defaults: `OVERHEAD_MULT = 1.5` (use `2.0` for long context / large batch),
`HEADROOM_GB = 1.0` (framework + driver allocations). CPU inference uses a
lighter `1.25` multiplier because llama.cpp streams weights and keeps a smaller
transient buffer.

### Worked examples

| Model | Precision | Raw weights | ×1.5 overhead | + headroom | **Require** |
|-------|-----------|-------------|---------------|------------|-------------|
| 7B    | FP16      | 13.0 GB     | 19.6 GB       | +1         | **~20.6 GB** |
| 7B    | INT4      | 3.3 GB      | 4.9 GB        | +1         | **~5.9 GB**  |
| 13B   | INT4      | 6.1 GB      | 9.1 GB        | +1         | **~10.1 GB** |
| 13B   | FP16      | 24.2 GB     | 36.3 GB       | +1         | **~37.3 GB** |
| 70B   | INT4      | 32.6 GB     | 48.9 GB       | +1         | **~49.9 GB** |
| 70B   | INT8      | 65.2 GB     | 97.8 GB       | +1         | **~98.8 GB** |

(1 billion params INT4 ≈ 0.47 GiB of weights; multiply and add overhead.)

## Tiered text-model recommendations

| Tier | Detected | Recommended text models | Format | Min VRAM | Disk |
|------|----------|-------------------------|--------|----------|------|
| **CPU-only** | no GPU, ≥8 GB RAM | Qwen2.5-3B, Mistral-7B, Llama-3.2-3B | GGUF Q4_K_M | 0 (uses RAM: 6–8 GB) | 2–5 GB |
| **Apple Silicon** | M-series, MPS | Mistral-7B, Qwen2.5-7B (Metal) | GGUF Q4/Q5 | unified (~8 GB) | 4–5 GB |
| **Mid-range GPU** | 8–16 GB VRAM | Mistral-7B GPTQ, Qwen2.5-7B, Llama-3.1-8B (4-bit) | GPTQ/GGUF int4 | 6–10 GB | 4–8 GB |
| **High-end GPU** | 24 GB VRAM | Qwen2.5-32B int4, Mixtral-8x7B GPTQ | GPTQ int4 | 19–28 GB | 19–24 GB |
| **Multi-GPU / 40 GB+** | 2× GPU or 48 GB+ | Llama-3.3-70B Q4, Mixtral fp16 | GGUF/safetensors | 40–80 GB | 40–140 GB |

### Small / medium / large minimums (quick reference)

| Class | Example | 4-bit VRAM | FP16 VRAM | Disk (4-bit) |
|-------|---------|-----------|-----------|--------------|
| Small (7B) | Mistral-7B | ~6 GB | ~20 GB | ~4 GB |
| Medium (13–16B) | Qwen2.5-14B | ~10 GB | ~34 GB | ~8 GB |
| Large (30–70B) | Llama-3.3-70B | ~40–50 GB | ~150 GB | ~40 GB |
| CPU GGUF (3–7B) | Qwen2.5-3B/7B | uses 4–8 GB **RAM** | — | 2–5 GB |

## Image-model recommendations

| Model | Resolution | VRAM (fp16) | Disk | Notes |
|-------|-----------|-------------|------|-------|
| SD 1.5 | 512² | ~4 GB | ~4.3 GB | Runs on 4 GB cards / MPS |
| SDXL 1.0 | 1024² | ~8–10 GB | ~13 GB | + refiner optional (+6 GB) |
| SDXL-Turbo | 1024², 1–4 steps | ~8 GB | ~13 GB | Fast preview generation |
| ControlNet-SDXL | +add-on | **+4–5 GB on top of base** | ~5 GB | Canny/depth/pose guidance |
| LoRA | +add-on | +0.1–0.5 GB | 20–400 MB | Style/subject fine-tunes |

With `enable_model_cpu_offload()` SDXL fits in ~6 GB VRAM at a speed cost.

## Video-generation recommendations

| Approach | Model | VRAM | Notes |
|----------|-------|------|-------|
| Text→video (light) | AnimateDiff on SD1.5 | ~8 GB | 16 frames, Apache-2.0 motion adapter |
| Image→video | Stable Video Diffusion (SVD-XT) | ~16 GB | 25 frames, **non-commercial** license, gated |
| **Fallback pipeline** | SD frame-by-frame + RIFE + FFmpeg | ~6 GB (SD1.5) | Works on modest GPUs; see below |

### Fallback frame-by-frame pipeline (works on 6–8 GB GPUs)

1. Generate keyframes with SD 1.5 (optionally ControlNet for consistency),
   varying prompt/seed/latent per frame (Deforum-style).
2. Interpolate to smooth motion with **RIFE**:
   `rife-ncnn-vulkan -i frames/ -o interp/ -n 2` (2× frame count).
3. Assemble with FFmpeg: `ffmpeg -r 24 -i interp/%08d.png -c:v libx264 -pix_fmt yuv420p out.mp4`.

Resource envelope: SD1.5 keyframes ~4 GB VRAM; RIFE ncnn ~1–2 GB; FFmpeg is
CPU-bound. A 4-second 24 fps clip ≈ 96 frames ≈ a few minutes on a mid GPU.

## GPU can't run the chosen pipeline?

The recommendation engine automatically demotes: if a diffusion video model
exceeds VRAM, it recommends the frame-by-frame fallback; if a text model
exceeds VRAM, it drops to the GGUF/CPU variant (measured against system RAM
instead of VRAM). This is what `installer/recommend.py::recommend()` encodes in
the `on_cpu` branch and the `fits` flag.
