"""
pytest stubs for manifest parsing + downloader integrity logic.
"""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from installer import downloader, recommend as rec

MANIFEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models_manifest.json")

REQUIRED_FIELDS = {"id", "kind", "source_url", "license", "gated", "params",
                   "size_bytes", "quantized_formats_available", "format",
                   "recommended_min_vram_GB", "recommended_mode",
                   "integrity_sha256"}


def test_manifest_is_valid_json():
    with open(MANIFEST, encoding="utf-8") as fh:
        data = json.load(fh)
    assert "models" in data and isinstance(data["models"], list)
    assert len(data["models"]) >= 5


def test_every_model_has_required_fields():
    with open(MANIFEST, encoding="utf-8") as fh:
        data = json.load(fh)
    ids = set()
    for m in data["models"]:
        missing = REQUIRED_FIELDS - set(m)
        assert not missing, f"{m.get('id')} missing {missing}"
        assert m["id"] not in ids, f"duplicate id {m['id']}"
        ids.add(m["id"])
        assert m["kind"] in ("text", "image", "video")
        assert m["recommended_mode"] in ("docker", "native", "cpu-only")


def test_load_manifest_merges_into_catalog():
    catalog = rec.load_manifest(MANIFEST)
    ids = {m.id for m in catalog}
    assert "mistral-7b-q4" in ids
    assert "sdxl-1.0" in ids


# --- downloader integrity ---------------------------------------------------
def test_sha256_and_verify(tmp_path):
    f = tmp_path / "blob.bin"
    payload = b"local-llm-chat integrity test"
    f.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert downloader.sha256_file(str(f)) == digest
    assert downloader.verify_sha256(str(f), digest) is True
    assert downloader.verify_sha256(str(f), "deadbeef") is False
    # None expected => skip verification (returns True)
    assert downloader.verify_sha256(str(f), None) is True


def test_verify_missing_file_returns_false(tmp_path):
    assert downloader.verify_sha256(str(tmp_path / "nope.bin"), "abc") is False


def test_http_download_dry_run_does_not_touch_network(tmp_path):
    dest = str(tmp_path / "model.gguf")
    out = downloader.http_download("https://example.com/model.gguf", dest,
                                   dry_run=True)
    assert out == dest
    assert not os.path.exists(dest)  # dry-run wrote nothing


def test_gated_models_flagged():
    catalog = {m.id: m for m in rec.load_manifest(MANIFEST)}
    assert catalog["llama3.1-8b-fp16"].gated is True
    assert catalog["mistral-7b-q4"].gated is False
