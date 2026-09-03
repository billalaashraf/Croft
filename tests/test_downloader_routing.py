# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""
Which download path a model takes, and what it carries.

The integrity check is only worth having if something actually calls it. This
pins down the routing decision in `cmd_pull`:

  * an entry with a `download_url` goes through `http_download` WITH the
    manifest's SHA256 — otherwise `integrity_sha256` is decoration, which is
    what it was;
  * an entry without one goes through `hf_download` WITH `allow_patterns` —
    otherwise pulling one 4 GB quant snapshots every quant in the repo.
"""
import argparse
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from installer import downloader, main as installer_main, recommend as rec  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "models_manifest.json")


class Recorder:
    """Stands in for a downloader function and remembers how it was called."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "/dev/null"


@pytest.fixture()
def pull(tmp_path, monkeypatch):
    """Run cmd_pull against a temp models dir with both backends stubbed."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(installer_main, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(installer_main, "STATE_FILE",
                        str(tmp_path / "installed.json"))
    http, hf = Recorder(), Recorder()
    monkeypatch.setattr(downloader, "http_download", http)
    monkeypatch.setattr(downloader, "hf_download", hf)

    def run(model_id):
        args = argparse.Namespace(model=model_id, token=None, yes=True,
                                  dry_run=False)
        return installer_main.cmd_pull(args), http, hf
    return run


def manifest_entry(model_id):
    with open(MANIFEST, encoding="utf-8") as fh:
        for entry in json.load(fh)["models"]:
            if entry["id"] == model_id:
                return entry
    raise AssertionError(f"{model_id} is not in the manifest")


# ---------------------------------------------------------------------------
# Direct single-file downloads
# ---------------------------------------------------------------------------
def test_a_direct_url_routes_through_http_download_with_the_hash(pull):
    entry = manifest_entry("mistral-7b-q4")
    rc, http, hf = pull("mistral-7b-q4")

    assert rc == 0
    assert hf.calls == [], "a single-file model must not snapshot the repo"
    assert len(http.calls) == 1
    args, kwargs = http.calls[0]
    assert args[0] == entry["download_url"]
    assert kwargs["expected_sha256"] == entry["integrity_sha256"]
    assert args[1].endswith("mistral-7b-instruct-v0.2.Q4_K_M.gguf")


def test_the_shipped_hash_is_a_real_sha256(pull):
    """A truncated placeholder ('3e0039fd0273fcbe...REPLACE_ME') guarantees a
    verification failure on every download, so the shape is worth asserting."""
    for entry in json.load(open(MANIFEST, encoding="utf-8"))["models"]:
        digest = entry["integrity_sha256"]
        if digest is None:
            continue
        assert len(digest) == 64, f"{entry['id']}: not a full SHA256"
        assert all(c in "0123456789abcdef" for c in digest.lower()), \
            f"{entry['id']}: not hexadecimal"


def test_every_direct_url_entry_carries_a_hash():
    """A `download_url` with no hash is an unverifiable download."""
    for entry in json.load(open(MANIFEST, encoding="utf-8"))["models"]:
        if entry.get("download_url"):
            assert entry["integrity_sha256"], \
                f"{entry['id']} has a download_url but no integrity_sha256"


# ---------------------------------------------------------------------------
# Snapshot downloads
# ---------------------------------------------------------------------------
def test_a_snapshot_model_is_scoped_by_allow_patterns(pull):
    rc, http, hf = pull("qwen2.5-7b-q4")

    assert rc == 0
    assert http.calls == []
    assert len(hf.calls) == 1
    patterns = hf.calls[0][1]["allow_patterns"]
    assert patterns, "a GGUF snapshot without patterns pulls every quant"
    assert any("gguf" in p.lower() for p in patterns)


def test_a_diffusers_model_pulls_the_whole_repo(pull):
    """Not every model can be filtered: a diffusers pipeline needs all of its
    components, so `None` here is the correct answer, not a missing one."""
    rc, http, hf = pull("sdxl-1.0")
    assert rc == 0
    assert hf.calls[0][1]["allow_patterns"] is None


def test_every_gguf_entry_is_filtered():
    for model in rec.load_manifest(MANIFEST):
        if model.format == "gguf":
            assert model.hf_allow_patterns, \
                f"{model.id}: GGUF snapshot with no allow_patterns"


def test_patterns_cover_both_spellings_of_the_quant():
    """Repos disagree on case (`Q4_K_M` vs `q4_k_m`) and fnmatch is
    case-sensitive on POSIX, so one spelling matches nothing on half of them."""
    for model in rec.load_manifest(MANIFEST):
        if model.format != "gguf":
            continue
        joined = " ".join(model.hf_allow_patterns)
        assert "Q4_K_M" in joined and "q4_k_m" in joined, \
            f"{model.id}: only one case spelling"


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------
def test_a_successful_pull_records_provenance(pull, tmp_path):
    rc, _http, _hf = pull("mistral-7b-q4")
    assert rc == 0
    state = json.load(open(tmp_path / "installed.json", encoding="utf-8"))
    recorded = state["models"]["mistral-7b-q4"]
    assert recorded["license"] == "Apache-2.0"
    assert recorded["source_url"] == manifest_entry("mistral-7b-q4")["download_url"]
    assert recorded["integrity_sha256"] == manifest_entry("mistral-7b-q4")["integrity_sha256"]


def test_an_unknown_model_id_fails_without_downloading(pull):
    rc, http, hf = pull("no-such-model")
    assert rc == 2
    assert http.calls == [] and hf.calls == []


# ---------------------------------------------------------------------------
# Reproducibility: pinned revisions
# ---------------------------------------------------------------------------
# `main` is a moving target and can be force-pushed. Without a pinned commit,
# two people installing the same model id weeks apart can get different weights
# and neither has any way to notice.
def test_every_snapshot_entry_pins_a_revision():
    for entry in json.load(open(MANIFEST, encoding="utf-8"))["models"]:
        if entry.get("download_url"):
            continue                       # pinned by integrity_sha256 instead
        rev = entry.get("revision")
        assert rev, f"{entry['id']}: hf_repo entry with no pinned revision"
        assert len(rev) == 40, f"{entry['id']}: {rev!r} is not a full commit SHA"
        assert all(c in "0123456789abcdef" for c in rev.lower()), \
            f"{entry['id']}: revision is not hexadecimal"


def test_the_pinned_revision_is_what_gets_downloaded(pull):
    """A revision in the manifest that never reaches snapshot_download would be
    documentation, not a pin — which is exactly what integrity_sha256 was."""
    rc, http, hf = pull("sdxl-turbo")

    assert rc == 0
    assert http.calls == [], "a snapshot model must not take the direct-URL path"
    assert len(hf.calls) == 1
    _args, kwargs = hf.calls[0]
    assert kwargs["revision"] == manifest_entry("sdxl-turbo")["revision"]


def test_a_pull_records_the_revision_it_installed(pull, tmp_path):
    rc, _http, _hf = pull("sdxl-turbo")
    assert rc == 0
    state = json.load(open(tmp_path / "installed.json", encoding="utf-8"))
    assert state["models"]["sdxl-turbo"]["revision"] == \
        manifest_entry("sdxl-turbo")["revision"]


# ---------------------------------------------------------------------------
# The docs and the manifest have to agree
# ---------------------------------------------------------------------------
def test_every_model_id_named_in_the_readme_exists():
    """`--model sdxl-turbo` sat in the README for a whole release while the
    manifest had no such entry, so every documented Images example failed."""
    import re
    ids = {e["id"] for e in json.load(open(MANIFEST, encoding="utf-8"))["models"]}
    readme = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    # `--model <id>` only. The trailing lookahead skips the examples that pass a
    # path or a repo id (`--model models/qwen2.5-7b-q4/*.gguf`), which are not
    # manifest ids and are not meant to be.
    cited = set(re.findall(r"--model\s+([a-z0-9][a-z0-9.\-]*)(?![\w/])", readme))
    unknown = {c for c in cited if c not in ids and not c.startswith("<")}
    assert not unknown, f"README names models that are not in the manifest: {sorted(unknown)}"
