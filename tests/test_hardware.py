# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""
pytest stubs for hardware detection + recommendation.

Run:  pytest -q
These tests avoid real GPU/network access by feeding synthetic reports and by
monkeypatching subprocess-backed probes.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from installer import hardware, recommend as rec


# --- hardware.detect_all shape ----------------------------------------------
def test_detect_all_returns_expected_keys():
    report = hardware.detect_all()
    for key in ("os", "arch", "cpu_cores_logical", "ram_total_gb",
                "disk_free_gb", "gpus", "total_vram_gb", "accelerator"):
        assert key in report, f"missing key: {key}"
    assert report["accelerator"] in ("cuda", "rocm", "mps", "cpu")
    assert isinstance(report["gpus"], list)


def test_detect_all_is_json_serializable():
    json.dumps(hardware.detect_all())  # must not raise


def test_nvidia_parser_handles_missing_tool(monkeypatch):
    # When nvidia-smi is absent, _run returns None -> no GPUs.
    monkeypatch.setattr(hardware, "_run", lambda *a, **k: None)
    assert hardware.detect_nvidia() == []


def test_nvidia_parser_parses_csv(monkeypatch):
    csv = "NVIDIA GeForce RTX 4090, 24564, 550.90, 8.9"
    def fake_run(cmd, timeout=15):
        if "--query-gpu" in " ".join(cmd):
            return csv
        return "CUDA Version: 12.4"
    monkeypatch.setattr(hardware, "_run", fake_run)
    gpus = hardware.detect_nvidia()
    assert len(gpus) == 1
    assert gpus[0].vendor == "nvidia"
    assert gpus[0].vram_gb == pytest.approx(24.0, abs=0.1)
    assert gpus[0].compute_capability == "8.9"
    assert gpus[0].cuda_version == "12.4"


# --- recommend formula ------------------------------------------------------
@pytest.mark.parametrize("params,precision,expected", [
    (7.0, "fp16", 20.6),   # 7e9*2*1.5/1024^3 + 1
    (7.0, "int4", 5.9),    # 7e9*0.5*1.5/1024^3 + 1
    (13.0, "int8", 19.2),
])
def test_estimate_required_gb(params, precision, expected):
    got = rec.estimate_required_gb(params, precision)
    assert got == pytest.approx(expected, abs=0.3)


def test_classify_tier():
    assert rec.classify_tier({"total_vram_gb": 24, "gpus": [1]}) == "high-end"
    assert rec.classify_tier({"total_vram_gb": 12, "gpus": [1]}) == "mid-range"
    assert rec.classify_tier({"total_vram_gb": 0, "accelerator": "cpu",
                              "gpus": []}) == "cpu-only"
    assert rec.classify_tier({"total_vram_gb": 48, "gpus": [1, 2]}) == "multi-gpu"


def test_recommend_cpu_only_prefers_gguf():
    report = {"total_vram_gb": 0, "ram_available_gb": 16, "ram_total_gb": 16,
              "disk_free_gb": 200, "accelerator": "cpu", "gpus": []}
    reco = rec.recommend(report)
    assert reco["tier"] == "cpu-only"
    fitting = [e for e in reco["recommendations"]["text"] if e["fits"]]
    assert fitting, "expected at least one CPU-fitting text model"
    assert all(e["format"] == "gguf" for e in fitting)


def test_recommend_high_end_fits_large_models():
    report = {"total_vram_gb": 80, "ram_available_gb": 128, "ram_total_gb": 128,
              "disk_free_gb": 500, "accelerator": "cuda", "gpus": [1, 2]}
    reco = rec.recommend(report, kinds=["text"])
    ids = {e["id"] for e in reco["recommendations"]["text"] if e["fits"]}
    assert "llama3.3-70b-q4" in ids or "mixtral-8x7b-gptq" in ids
