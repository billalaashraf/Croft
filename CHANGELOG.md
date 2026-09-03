# Changelog

All notable changes to Croft are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The version lives in one place — `installer/__version__` — and `pyproject.toml`
and the web app both read it from there.

## [Unreleased]

## [0.1.0] — 2026-09-03

First public release. Croft has been usable for a while; this is the version
that made it publishable — a licence, a stable name, pinned model revisions, and
CI that covers the parts of the install that actually break.

### Added

- **Apache-2.0 licence** (`LICENSE`, `NOTICE`, SPDX headers on every module).
  Croft bundles no model weights; each model's licence is recorded in
  `models_manifest.json` and in `models/installed.json` once installed.
- **`pyproject.toml`** — `pip install -e .` for contributors, and a `croft`
  console entry point as a shorter alias for `python3 -m installer.main`.
- **`sdxl-turbo` in the model manifest.** It was referenced throughout the docs
  and in `webui/imagegen.py` but never shipped, so every documented Images
  example failed. It is now the fast default for the Images tab.
- **Pinned model revisions.** Every `hf_repo` entry carries a `revision` commit
  SHA, so two people installing weeks apart get identical weights and an
  upstream force-push cannot change what Croft downloads. Enforced by
  `tests/test_downloader_routing.py`.
- **`.env.example`** documenting every `LLM_*` environment variable in one
  place.
- **Contributor scaffolding** — `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue
  and pull-request templates, and private vulnerability reporting.
- **CI coverage for the engine install.** A nightly job installs the real
  `torch`/`diffusers`/`transformers` stack and asserts it imports; a per-PR
  resolver check catches conflicting pins in seconds. Docker Compose services
  gained healthchecks.

### Changed

- **The project is called Croft everywhere.** The README, the layout diagram,
  the FastAPI title and the bootstrap's fallback install directory previously
  disagreed, naming it "Local LLM Chat" or `local-llm-chat`.
- **CI tests Python 3.10 through 3.14.** The matrix stopped at 3.12 while
  `requirements.lock` was resolved on 3.14, so the interpreter most likely to be
  in use was the one interpreter never tested.
- **Docker base images are pinned by digest** instead of floating on `:latest`
  and `:server`.
- **`docs/SECURITY.md`** no longer claims snapshot downloads are verified by
  commit hash in a way the manifest did not support — now that revisions are
  pinned, the claim is accurate — and directs vulnerability reports to GitHub's
  private reporting instead of a public issue.
- **The README's file-by-file layout diagram** is now directory-level. The
  file-level version had already drifted: it listed two of six test files.

### Fixed

- **`pip` and `uvicorn` are invoked through the venv interpreter** (`python -m
  pip`) rather than through `.venv/bin/pip`. A console script hardcodes the
  absolute path of the venv that created it into its shebang, so renaming or
  copying the project directory left every one of them dead with
  `bad interpreter` — and the installer reported it as a network failure.
  `bootstrap_install.sh` and `webui.sh` now also detect a venv whose base
  interpreter has gone and rebuild it instead of failing every install.
- **`transformers` is constrained to `<5`** in the engine install. Unpinned, pip
  resolved `transformers` 5.x, which requires `huggingface_hub>=1.5` and so
  collided with the deliberate `<1.0` pin — pip then downgraded the hub to
  satisfy the lock and left `transformers` unimportable on an install that
  reported success.
- **A malformed `models_manifest.json` now warns** instead of silently falling
  back to the built-in catalogue, which made a bad hand-edit look like a model
  that simply never appeared.
- **The `text-llamacpp` compose service pointed at an image that no longer
  exists.** llama.cpp moved to the `ggml-org` organisation, so
  `ghcr.io/ggerganov/llama.cpp:server` returns 404 and the service could not
  start. Now `ghcr.io/ggml-org/llama.cpp:server`, pinned by digest.
- **The systemd unit and launchd agent invoked `.venv/bin/uvicorn`** and so had
  the same latent breakage as the shell scripts — a moved checkout produced
  `203/EXEC` with nothing explaining why. Both now go through `python -m
  uvicorn`.
- **`LLM_CHAT_CTX` and `LLM_CHAT_GGUF` were documented but never read.** The
  README described two knobs that did nothing; they have been removed from it
  rather than invented in the code.

[Unreleased]: https://github.com/billalaashraf/Croft/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/billalaashraf/Croft/releases/tag/v0.1.0
