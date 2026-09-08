# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""
The built-in catalog must be as pinned as the manifest.

`models_manifest.json` gets its own reproducibility tests, but the manifest
does not name every model id Croft can install: five entries live only in
`recommend.DEFAULT_CATALOG`. Those fell through to `revision=None`, which
`cmd_pull` turns into `main` — a moving target that can be force-pushed, so
repeated installs of the same id could fetch different weights with nothing to
indicate it had happened.

These tests close that gap permanently: a new catalog entry without a pin fails
here rather than shipping.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from installer import recommend as rec  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "models_manifest.json")


def _snapshot_entries():
    """Catalog entries that download a repo snapshot rather than one file.

    A `download_url` entry is pinned by `integrity_sha256` instead, so it is not
    this file's problem.
    """
    return [m for m in rec.DEFAULT_CATALOG if not m.download_url]


def test_every_builtin_snapshot_entry_is_pinned():
    unpinned = [m.id for m in _snapshot_entries() if not m.revision]
    assert not unpinned, (
        "these catalog entries would download the moving `main` branch: "
        f"{sorted(unpinned)}. Add a commit SHA to recommend.DEFAULT_REVISIONS."
    )


def test_pins_are_full_commit_shas():
    for m in _snapshot_entries():
        assert len(m.revision) == 40, f"{m.id}: {m.revision!r} is not a commit SHA"
        assert all(c in "0123456789abcdef" for c in m.revision.lower()), \
            f"{m.id}: revision is not hexadecimal"


def test_the_revision_table_has_no_stale_entries():
    """A pin for an id that no longer exists is dead weight that reads as
    coverage. Catch it here rather than letting the table drift."""
    ids = {m.id for m in rec.DEFAULT_CATALOG}
    stale = set(rec.DEFAULT_REVISIONS) - ids
    assert not stale, f"DEFAULT_REVISIONS pins ids not in the catalog: {sorted(stale)}"


def test_the_manifest_agrees_with_the_builtin_pins():
    """Where both name a model, they must name the same commit — otherwise which
    weights you get depends on whether the manifest happens to be readable."""
    manifest = {e["id"]: e for e in json.load(open(MANIFEST, encoding="utf-8"))["models"]}
    for mid, rev in rec.DEFAULT_REVISIONS.items():
        if mid in manifest and manifest[mid].get("revision"):
            assert manifest[mid]["revision"] == rev, (
                f"{mid}: manifest pins {manifest[mid]['revision']}, "
                f"DEFAULT_REVISIONS pins {rev}")


def test_a_manifest_entry_without_a_revision_keeps_the_builtin_pin(tmp_path):
    """The merge must not let an unpinned manifest entry erase a good default.

    Overriding a model to change its size or notes is a documented workflow;
    doing so should not silently reopen the `main` hole.
    """
    partial = tmp_path / "models_manifest.json"
    partial.write_text(json.dumps({"models": [{
        "id": "sdxl-turbo",
        "kind": "image",
        "hf_repo": "stabilityai/sdxl-turbo",
        # deliberately no "revision"
        "license": "StabilityAI-NC-Community",
        "params": 3500000000,
        "size_bytes": 7000000000,
        "format": "diffusers",
    }]}), encoding="utf-8")

    merged = {m.id: m for m in rec.load_manifest(str(partial))}
    assert merged["sdxl-turbo"].revision == rec.DEFAULT_REVISIONS["sdxl-turbo"]
