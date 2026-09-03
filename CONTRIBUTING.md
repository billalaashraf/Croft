# Contributing to Croft

Thanks for taking the time. This document is what CI already enforces, written
down — so you find out what's expected before a pull request goes red rather
than after.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting set up

```bash
git clone https://github.com/billalaashraf/Croft.git croft && cd croft
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,webui]'
pytest -q
```

`pip install -e` is for contributors. `bootstrap_install.sh` deliberately does
not use it — the bootstrap has to work on a machine where nothing is installed
yet, so it installs from `requirements.txt` / `requirements.lock` instead.

One rule that has already bitten this project: **never invoke `.venv/bin/pip` or
`.venv/bin/uvicorn` directly** in scripts. A console script bakes the absolute
path of the venv that created it into its shebang, so renaming the project
directory leaves every one of them dead with `bad interpreter`. Go through the
interpreter — `"$VENV_PY" -m pip` — which survives a move.

## What CI checks

Everything below runs on every pull request. Running it locally first is faster
than waiting for the matrix.

| Check | Command |
|---|---|
| Unit tests, Python 3.10–3.14, Ubuntu + macOS | `pytest -q` |
| Shell test suite | `./tests/test_bootstrap.sh` |
| Shell linting | `shellcheck --severity=warning bootstrap_install.sh webui.sh tests/test_bootstrap.sh` |
| Every module byte-compiles | `python -m compileall -q installer webui tests` |
| Bootstrap makes no changes in `--dry-run` | `./bootstrap_install.sh --dry-run --yes --mode native` |
| The lockfile still installs | `pip install -r requirements.lock` |
| Engine dependency resolution | `pip install --dry-run torch diffusers 'transformers>=4.46,<5' …` |

A **nightly** job additionally installs the real inference engines and asserts
they import. It is nightly rather than per-PR because it downloads several
gigabytes of wheels.

## Conventions

**Comments explain why, not what.** This codebase's comments carry a lot of
weight — they record the reasoning behind a non-obvious choice, and often the
bug that motivated it. Match that. A comment that restates the code is noise; a
comment that says "WSL reports `uname -s` as Linux, so it can only be told apart
*after* the Linux match" is the reason the next person doesn't reintroduce the
bug.

**Every change to install or download behaviour needs a test.** The pattern to
follow is `tests/test_downloader_routing.py`: it asserts the *policy* ("a
`download_url` with no hash is an unverifiable download"), not just the
mechanics. A policy that is only written in a comment is a policy that will be
broken.

**Keep `--dry-run` honest.** Every command has a dry-run path, and its output
should be exactly what the real run would execute. If you add a step, add it to
the dry-run branch in the same commit — a dry-run that under-reports is worse
than no dry-run.

**Consent before anything expensive or irreversible.** No download, system
change, or install happens without a prompt or an explicit `--yes`.

**Adding a model?** See [Adding custom models](README.md#adding-custom-models).
`integrity_sha256` is required for direct downloads and `revision` is required
for Hugging Face repos; the tests will reject an entry missing either.

## Commit messages

A short imperative subject line, then a body explaining *why* if it isn't
obvious. Reference an issue when there is one.

```
Pin manifest revisions to commit SHAs

`main` moves and can be force-pushed, so two people installing the same
model id weeks apart could get different weights with no way to notice.
```

## Pull requests

- Branch from `main`.
- Keep it focused — one concern per PR reviews far better than a large mixed one.
- Update `CHANGELOG.md` under `## [Unreleased]` for anything user-visible.
- Say what you tested. "Ran the bootstrap on a clean Ubuntu 24.04 VM" is worth
  more than a green checkmark on the parts CI can reach.

## Reporting bugs

Open an issue with the template. The most useful thing you can include is the
output of:

```bash
python3 -m installer.main detect        # hardware report
python3 -c 'import installer; print(installer.__version__)'
```

## Security

Do **not** open a public issue for a vulnerability. Use
[GitHub's private vulnerability reporting](https://github.com/billalaashraf/Croft/security/advisories/new).
See [docs/SECURITY.md](docs/SECURITY.md) for the threat model and what is
explicitly out of scope.

## Licence

Croft is Apache-2.0. Contributions are accepted under the same licence, and new
source files should carry the SPDX header the existing ones do:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Bilal Ashraf
```
