# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""
hardware.py — Cross-platform hardware capability probe.

Produces a JSON "capability report" describing CPU, RAM, disk, swap and every
detected GPU (NVIDIA / AMD / Apple Silicon). The module degrades gracefully:
if a vendor tool (nvidia-smi, rocm-smi, system_profiler) or an optional Python
dependency (psutil, pynvml, py-cpuinfo) is missing, the corresponding fields are
filled with best-effort fallbacks and a note is added to `report["warnings"]`.

Usage:
    python -m installer.hardware              # pretty JSON to stdout
    python -m installer.hardware --raw        # compact JSON
    from installer.hardware import detect_all  # -> dict
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# ---- Optional dependencies (all imports are soft) ---------------------------
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    psutil = None

try:
    import cpuinfo  # py-cpuinfo  # type: ignore
except Exception:  # pragma: no cover
    cpuinfo = None

_BYTES_PER_GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run(cmd: List[str], timeout: int = 15) -> Optional[str]:
    """Run a command, returning stripped stdout or None on any failure."""
    if not cmd or shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _round(n: Optional[float], digits: int = 1) -> Optional[float]:
    return None if n is None else round(float(n), digits)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class GPU:
    vendor: str                       # "nvidia" | "amd" | "apple" | "unknown"
    name: str = "unknown"
    vram_gb: Optional[float] = None
    driver_version: Optional[str] = None
    compute_capability: Optional[str] = None   # e.g. "8.6" (NVIDIA)
    cuda_version: Optional[str] = None
    backend: Optional[str] = None              # "cuda" | "rocm" | "mps"


@dataclass
class Report:
    os: str = platform.system().lower()
    os_release: str = ""
    arch: str = platform.machine()
    kernel: str = platform.release()
    python: str = platform.python_version()
    cpu_model: str = "unknown"
    cpu_cores_physical: Optional[int] = None
    cpu_cores_logical: Optional[int] = None
    ram_total_gb: Optional[float] = None
    ram_available_gb: Optional[float] = None
    swap_total_gb: Optional[float] = None
    disk_free_gb: Optional[float] = None
    disk_path: str = "."
    gpus: List[GPU] = field(default_factory=list)
    total_vram_gb: float = 0.0
    accelerator: str = "cpu"          # "cuda" | "rocm" | "mps" | "cpu"
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CPU / RAM / disk
# ---------------------------------------------------------------------------
def detect_cpu(report: Report) -> None:
    if cpuinfo is not None:
        try:
            info = cpuinfo.get_cpu_info()
            report.cpu_model = info.get("brand_raw") or report.cpu_model
        except Exception:
            report.warnings.append("py-cpuinfo failed; CPU model may be generic")
    if report.cpu_model == "unknown":
        # Fallbacks by platform
        if report.os == "linux":
            txt = _read_file("/proc/cpuinfo") or ""
            m = re.search(r"model name\s*:\s*(.+)", txt)
            if m:
                report.cpu_model = m.group(1).strip()
        elif report.os == "darwin":
            report.cpu_model = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or report.cpu_model

    if psutil is not None:
        report.cpu_cores_physical = psutil.cpu_count(logical=False)
        report.cpu_cores_logical = psutil.cpu_count(logical=True)
    else:
        report.cpu_cores_logical = os.cpu_count()
        report.warnings.append("psutil missing; physical core count unavailable")


def detect_memory(report: Report) -> None:
    if psutil is not None:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        report.ram_total_gb = _round(vm.total / _BYTES_PER_GB)
        report.ram_available_gb = _round(vm.available / _BYTES_PER_GB)
        report.swap_total_gb = _round(sm.total / _BYTES_PER_GB)
        return
    # Fallback: Linux /proc/meminfo
    txt = _read_file("/proc/meminfo")
    if txt:
        def _kb(key: str) -> Optional[float]:
            m = re.search(rf"{key}:\s+(\d+) kB", txt)
            return int(m.group(1)) / (1024 ** 2) if m else None
        report.ram_total_gb = _round(_kb("MemTotal"))
        report.ram_available_gb = _round(_kb("MemAvailable"))
        report.swap_total_gb = _round(_kb("SwapTotal"))
    else:
        report.warnings.append("Could not determine system RAM")


def detect_disk(report: Report, path: str = ".") -> None:
    report.disk_path = os.path.abspath(path)
    try:
        usage = shutil.disk_usage(report.disk_path)
        report.disk_free_gb = _round(usage.free / _BYTES_PER_GB)
    except Exception:
        report.warnings.append(f"Could not read disk usage for {path}")


def _read_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def detect_nvidia() -> List[GPU]:
    """Detect NVIDIA GPUs via nvidia-smi (CSV query). VRAM in MiB -> GiB."""
    gpus: List[GPU] = []
    query = "name,memory.total,driver_version,compute_cap"
    out = _run([
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return gpus
    # CUDA runtime version appears in `nvidia-smi` header
    cuda_ver = None
    header = _run(["nvidia-smi"]) or ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", header)
    if m:
        cuda_ver = m.group(1)
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            vram = float(parts[1]) / 1024.0  # MiB -> GiB
        except ValueError:
            vram = None
        driver = parts[2] if len(parts) > 2 else None
        cc = parts[3] if len(parts) > 3 else None
        gpus.append(GPU(
            vendor="nvidia", name=name, vram_gb=_round(vram),
            driver_version=driver, compute_capability=cc,
            cuda_version=cuda_ver, backend="cuda",
        ))
    return gpus


def detect_amd() -> List[GPU]:
    """Detect AMD GPUs via rocm-smi. VRAM parsing is best-effort."""
    gpus: List[GPU] = []
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if out:
        try:
            data = json.loads(out)
            for key, dev in data.items():
                if not key.lower().startswith("card"):
                    continue
                name = dev.get("Card series") or dev.get("Card model") or "AMD GPU"
                vram_bytes = dev.get("VRAM Total Memory (B)") or dev.get("vram_total")
                vram = float(vram_bytes) / _BYTES_PER_GB if vram_bytes else None
                gpus.append(GPU(vendor="amd", name=str(name),
                                vram_gb=_round(vram), backend="rocm"))
            if gpus:
                return gpus
        except Exception:
            pass
    # Fallback: plain rocm-smi presence => report an unknown-VRAM AMD card
    if _run(["rocm-smi"]) is not None:
        gpus.append(GPU(vendor="amd", name="AMD GPU (ROCm)", backend="rocm"))
    return gpus


def detect_apple() -> List[GPU]:
    """Detect Apple Silicon GPU. Unified memory => VRAM≈system RAM budget."""
    if platform.system().lower() != "darwin" or platform.machine() != "arm64":
        return []
    name = "Apple Silicon GPU"
    out = _run(["system_profiler", "SPDisplaysDataType"])
    if out:
        m = re.search(r"Chipset Model:\s*(.+)", out)
        if m:
            name = m.group(1).strip()
    # Unified memory: usable GPU budget ~= 65-75% of system RAM.
    ram_bytes = None
    mem = _run(["sysctl", "-n", "hw.memsize"])
    if mem and mem.isdigit():
        ram_bytes = int(mem)
    vram = _round((ram_bytes * 0.70) / _BYTES_PER_GB) if ram_bytes else None
    return [GPU(vendor="apple", name=name, vram_gb=vram, backend="mps")]


def detect_gpus(report: Report) -> None:
    gpus: List[GPU] = []
    gpus += detect_nvidia()
    gpus += detect_amd()
    gpus += detect_apple()
    report.gpus = gpus
    report.total_vram_gb = _round(sum(g.vram_gb or 0.0 for g in gpus)) or 0.0
    if gpus:
        # Priority: cuda > rocm > mps for backend selection
        backends = {g.backend for g in gpus if g.backend}
        for pref in ("cuda", "rocm", "mps"):
            if pref in backends:
                report.accelerator = pref
                break
    else:
        report.accelerator = "cpu"
        report.warnings.append(
            "No GPU detected; only CPU / ggml-gguf models will be recommended"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_all(disk_path: str = ".") -> Dict[str, Any]:
    report = Report()
    try:
        report.os_release = platform.platform()
    except Exception:
        report.os_release = report.os
    detect_cpu(report)
    detect_memory(report)
    detect_disk(report, disk_path)
    detect_gpus(report)
    out = asdict(report)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    raw = "--raw" in argv
    path = "."
    for i, a in enumerate(argv):
        if a in ("--path", "-p") and i + 1 < len(argv):
            path = argv[i + 1]
    report = detect_all(path)
    print(json.dumps(report, indent=None if raw else 2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
