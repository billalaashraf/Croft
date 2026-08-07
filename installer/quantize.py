"""
quantize.py — Conversion / quantization hooks.

These functions *build the command lines* and (optionally) run them via
subprocess. Every runner supports `dry_run=True` to print the exact commands
without executing — the safe default for the installer's preview mode.

Supported paths:
  * llama.cpp GGUF conversion + quantization (CPU / Metal / CUDA)
  * AutoGPTQ 4-bit quantization (GPU)
  * bitsandbytes 4-bit load recipe (runtime, no separate conversion step)
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from typing import List, Optional


def _run(cmd: List[str], dry_run: bool) -> int:
    printable = " ".join(shlex.quote(c) for c in cmd)
    if dry_run:
        print(f"[dry-run] {printable}")
        return 0
    print(f"[run] {printable}")
    return subprocess.run(cmd, check=False).returncode


# ---------------------------------------------------------------------------
# llama.cpp — HF safetensors -> GGUF -> quantized GGUF
# ---------------------------------------------------------------------------
def llamacpp_convert_to_gguf(hf_model_dir: str, out_gguf: str, *,
                             llamacpp_dir: str = "third_party/llama.cpp",
                             outtype: str = "f16",
                             dry_run: bool = False) -> int:
    """Convert an HF model directory to an f16 GGUF file."""
    cmd = [
        sys.executable,
        f"{llamacpp_dir}/convert_hf_to_gguf.py",
        hf_model_dir,
        "--outfile", out_gguf,
        "--outtype", outtype,
    ]
    return _run(cmd, dry_run)


def llamacpp_quantize(in_gguf: str, out_gguf: str, *,
                      quant: str = "Q4_K_M",
                      llamacpp_dir: str = "third_party/llama.cpp",
                      dry_run: bool = False) -> int:
    """Quantize an f16 GGUF to e.g. Q4_K_M / Q5_K_M / Q8_0 using llama-quantize."""
    cmd = [f"{llamacpp_dir}/build/bin/llama-quantize", in_gguf, out_gguf, quant]
    return _run(cmd, dry_run)


def build_llamacpp(llamacpp_dir: str = "third_party/llama.cpp", *,
                   cuda: bool = False, metal: bool = False,
                   dry_run: bool = False) -> int:
    """Clone + build llama.cpp with the appropriate accelerator backend."""
    clone = ["git", "clone", "--depth", "1",
             "https://github.com/ggerganov/llama.cpp", llamacpp_dir]
    _run(clone, dry_run)
    flags = ["cmake", "-S", llamacpp_dir, "-B", f"{llamacpp_dir}/build"]
    if cuda:
        flags.append("-DGGML_CUDA=ON")
    if metal:
        flags.append("-DGGML_METAL=ON")
    _run(flags, dry_run)
    return _run(["cmake", "--build", f"{llamacpp_dir}/build",
                 "--config", "Release", "-j"], dry_run)


# ---------------------------------------------------------------------------
# AutoGPTQ — 4-bit GPTQ quantization
# ---------------------------------------------------------------------------
def autogptq_quantize(hf_model: str, out_dir: str, *,
                      bits: int = 4, group_size: int = 128,
                      dataset: str = "c4", dry_run: bool = False) -> int:
    """
    Emit an AutoGPTQ quantization invocation. Requires `pip install auto-gptq
    optimum` and a CUDA GPU. Uses a small calibration dataset.
    """
    script = (
        "from transformers import AutoTokenizer; "
        "from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig; "
        f"tok=AutoTokenizer.from_pretrained('{hf_model}', use_fast=True); "
        f"cfg=BaseQuantizeConfig(bits={bits}, group_size={group_size}, desc_act=False); "
        f"m=AutoGPTQForCausalLM.from_pretrained('{hf_model}', cfg); "
        "examples=[tok('The quick brown fox jumps over the lazy dog.', return_tensors='pt')]; "
        "m.quantize(examples); "
        f"m.save_quantized('{out_dir}', use_safetensors=True); "
        f"tok.save_pretrained('{out_dir}')"
    )
    return _run([sys.executable, "-c", script], dry_run)


# ---------------------------------------------------------------------------
# bitsandbytes — runtime 4-bit (no offline conversion)
# ---------------------------------------------------------------------------
def bitsandbytes_recipe(hf_model: str) -> str:
    """Return a copy-pasteable snippet for 4-bit (nf4) runtime loading."""
    return (
        "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n"
        "import torch\n"
        "bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',\n"
        "                         bnb_4bit_compute_dtype=torch.bfloat16,\n"
        "                         bnb_4bit_use_double_quant=True)\n"
        f"model = AutoModelForCausalLM.from_pretrained('{hf_model}',\n"
        "                                              quantization_config=bnb,\n"
        "                                              device_map='auto')\n"
    )


if __name__ == "__main__":  # pragma: no cover
    print(bitsandbytes_recipe("mistralai/Mistral-7B-Instruct-v0.2"))
    llamacpp_quantize("model-f16.gguf", "model-q4.gguf", dry_run=True)
    autogptq_quantize("mistralai/Mistral-7B-Instruct-v0.2", "out-gptq", dry_run=True)
