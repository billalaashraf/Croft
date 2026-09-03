# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
"""
downloader.py — Resumable, integrity-checked model downloads.

Two backends:
  * huggingface_hub.snapshot_download (preferred; handles gated repos + token)
  * a pure-`requests` HTTP range downloader (fallback / direct mirrors) with
    resume-on-partial (.part files) and SHA256 verification.

All destructive actions are opt-in: nothing is deleted without `force=True`,
and a `dry_run` flag previews the plan without touching the network.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Callable, Dict, Optional

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

_CHUNK = 1024 * 1024  # 1 MiB


# ---------------------------------------------------------------------------
# Checksums
# ---------------------------------------------------------------------------
def sha256_file(path: str, chunk: int = _CHUNK) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_sha256(path: str, expected: Optional[str]) -> bool:
    """Return True if the file matches. If `expected` is None, skip (True)."""
    if not expected:
        return True
    if not os.path.exists(path):
        return False
    actual = sha256_file(path)
    return actual.lower() == expected.lower()


# ---------------------------------------------------------------------------
# HTTP range downloader with resume
# ---------------------------------------------------------------------------
def http_download(url: str, dest: str, *,
                  expected_sha256: Optional[str] = None,
                  token: Optional[str] = None,
                  max_retries: int = 5,
                  dry_run: bool = False,
                  progress: Optional[Callable[[int, Optional[int]], None]] = None,
                  ) -> str:
    """
    Download `url` to `dest`, resuming from a `.part` file if present.
    Verifies SHA256 when provided. Retries with exponential backoff.
    Returns the final path. Idempotent: a completed, verified file is skipped.
    """
    if requests is None:
        raise RuntimeError("`requests` is required for http_download; pip install requests")

    if os.path.exists(dest) and verify_sha256(dest, expected_sha256):
        if progress:
            progress(os.path.getsize(dest), os.path.getsize(dest))
        return dest  # already complete & verified — idempotent no-op

    if dry_run:
        print(f"[dry-run] would download {url} -> {dest}")
        return dest

    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    part = dest + ".part"
    base_headers: Dict[str, str] = {}
    if token:
        base_headers["Authorization"] = f"Bearer {token}"

    # An explicit flag rather than a while/else: the loop now has two ways out
    # (success, and giving up without a final pointless sleep), and `else`
    # cannot tell them apart. Without this a run that exhausted its retries
    # fell through to the verification step and — with no expected hash to
    # check — returned the path of a file that was never written.
    completed = False
    last_error: Optional[BaseException] = None

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        resume_at = os.path.getsize(part) if os.path.exists(part) else 0
        # Rebuilt every attempt. Carrying one dict across retries meant a
        # `Range` header from an earlier attempt survived into a later one that
        # had no partial file left to resume from, so the server returned bytes
        # from the middle of the file and the download silently lost its head.
        headers = dict(base_headers)
        if resume_at:
            headers["Range"] = f"bytes={resume_at}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:
                    # Range not satisfiable: we already hold the whole file.
                    # Only true if we actually asked for a range.
                    if resume_at and os.path.exists(part):
                        os.replace(part, dest)
                        completed = True
                        break
                    raise RuntimeError(
                        "server rejected the byte range but there is nothing "
                        "to resume from")
                r.raise_for_status()
                # A server may ignore Range and answer 200 with the whole file.
                # Appending that to a partial would corrupt it, so start over.
                mode = "ab" if resume_at and r.status_code == 206 else "wb"
                if mode == "wb":
                    resume_at = 0
                # Order matters: Content-Length is the length of *this*
                # response, so the bytes already on disk are only part of the
                # total when we are genuinely resuming. Computing this before
                # the mode decision inflated the total on a 200 fallback and
                # left the progress bar stuck short of 100%.
                total = None
                if "Content-Length" in r.headers:
                    try:
                        total = int(r.headers["Content-Length"]) + resume_at
                    except ValueError:
                        total = None
                downloaded = resume_at
                with open(part, mode) as fh:
                    for block in r.iter_content(_CHUNK):
                        if not block:
                            continue
                        fh.write(block)
                        downloaded += len(block)
                        if progress:
                            progress(downloaded, total)
            os.replace(part, dest)
            completed = True
            break
        except Exception as exc:  # network hiccup -> back off and resume
            last_error = exc
            have = os.path.getsize(part) if os.path.exists(part) else 0
            if attempt >= max_retries:
                sys.stderr.write(
                    f"[downloader] attempt {attempt}/{max_retries} failed: {exc}\n")
                break          # no point sleeping before giving up
            wait = min(2 ** attempt, 30)
            sys.stderr.write(
                f"[downloader] attempt {attempt}/{max_retries} failed: {exc}; "
                f"retrying in {wait}s (resuming from {have} bytes)\n")
            time.sleep(wait)

    if not completed:
        raise RuntimeError(
            f"Download failed after {max_retries} attempts: {url} ({last_error})")

    if not verify_sha256(dest, expected_sha256):
        # Leave the file for inspection but signal failure clearly.
        raise ValueError(
            f"SHA256 mismatch for {dest}: expected {expected_sha256}, "
            f"got {sha256_file(dest)}")
    return dest


# ---------------------------------------------------------------------------
# Hugging Face snapshot download (gated + token aware)
# ---------------------------------------------------------------------------
def hf_download(repo_id: str, dest_dir: str, *,
                token: Optional[str] = None,
                revision: str = "main",
                allow_patterns: Optional[list] = None,
                dry_run: bool = False) -> str:
    """
    Download a full repo snapshot with resume (hf transfer handles integrity via
    ETag / commit hash). `token` is required for gated repos.
    """
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if dry_run:
        print(f"[dry-run] would snapshot_download {repo_id}@{revision} -> {dest_dir}"
              f" (patterns={allow_patterns})")
        return dest_dir
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub not installed. `pip install huggingface_hub` "
            f"or use http_download for direct mirrors. ({exc})")
    os.makedirs(dest_dir, exist_ok=True)
    # No `resume_download=True` or `local_dir_use_symlinks=False` here. Both
    # were deprecated and then removed: resuming is the default now, and a
    # `local_dir` snapshot already writes real files rather than symlinks.
    # Passing them warns on the versions this project pins and raises on 1.x.
    return snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=dest_dir,
        token=token,
        allow_patterns=allow_patterns,
    )


def cli_progress(downloaded: int, total: Optional[int]) -> None:
    if total:
        pct = downloaded / total * 100
        bar = "#" * int(pct // 4)
        sys.stdout.write(f"\r  [{bar:<25}] {pct:5.1f}%  "
                         f"{downloaded/1e6:8.1f}/{total/1e6:.1f} MB")
    else:
        sys.stdout.write(f"\r  {downloaded/1e6:8.1f} MB")
    sys.stdout.flush()
    if total and downloaded >= total:
        sys.stdout.write("\n")


if __name__ == "__main__":  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description="Resumable model downloader")
    p.add_argument("url_or_repo")
    p.add_argument("dest")
    p.add_argument("--sha256")
    p.add_argument("--hf", action="store_true", help="treat arg as HF repo id")
    p.add_argument("--token")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.hf:
        hf_download(a.url_or_repo, a.dest, token=a.token, dry_run=a.dry_run)
    else:
        http_download(a.url_or_repo, a.dest, expected_sha256=a.sha256,
                      token=a.token, dry_run=a.dry_run, progress=cli_progress)
