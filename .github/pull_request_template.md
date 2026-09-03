<!--
Thanks for the pull request. CONTRIBUTING.md has the dev setup and the full list
of what CI checks; the boxes below are the things reviewers most often have to
ask about.
-->

## What this changes

<!-- One or two sentences. If it fixes an issue, "Fixes #123". -->

## Why

<!-- The reasoning, especially for a non-obvious choice. This is the part that
     tends to become a comment in the code. -->

## How it was tested

<!-- Beyond CI. "Ran the bootstrap on a clean Ubuntu 24.04 VM" is worth more
     than a green check on the parts CI can reach. Say if you could not test a
     platform — that is useful, not disqualifying. -->

## Checklist

- [ ] `pytest -q` passes
- [ ] `shellcheck --severity=warning bootstrap_install.sh webui.sh tests/test_bootstrap.sh` is clean (if shell changed)
- [ ] `--dry-run` output matches what a real run would do (if install behaviour changed)
- [ ] A test covers the new behaviour, asserting the policy rather than the mechanics
- [ ] `CHANGELOG.md` updated under `## [Unreleased]` (if user-visible)
- [ ] New files carry the SPDX header
- [ ] New manifest entries have `revision`, and `integrity_sha256` for direct downloads
