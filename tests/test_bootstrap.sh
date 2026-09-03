#!/usr/bin/env bash
# tests/test_bootstrap.sh — checks on bootstrap_install.sh that pytest cannot make.
#
# Two things are being pinned down here:
#
#   1. `sha256_check` works with whichever hashing tool the machine has. macOS
#      ships `shasum` and not `sha256sum`, so the GNU-only form used before
#      failed the tarball verification on every Mac.
#   2. `--dry-run` changes nothing on disk. That is the promise the flag makes,
#      and it is only as good as the last edit to the script.
#
# Run:  ./tests/test_bootstrap.sh        (plain bash, no bats needed)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BOOTSTRAP="$PROJECT_DIR/bootstrap_install.sh"

PASS=0
FAIL=0

ok()   { PASS=$((PASS + 1)); printf '  \033[0;32mok\033[0m   %s\n' "$1"; }
bad()  { FAIL=$((FAIL + 1)); printf '  \033[0;31mFAIL\033[0m %s\n' "$1"; }
check() { if [ "$1" -eq 0 ]; then ok "$2"; else bad "$2"; fi; }

WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "bootstrap_install.sh checks"

# --- source just the helpers -------------------------------------------------
# The script runs an install when sourced, so pull out the block above the
# argument parser — which is where the helpers live — and source that instead.
sed -n '1,/^# ---- args/p' "$BOOTSTRAP" > "$WORK/helpers.sh"
DRY_RUN=0
# shellcheck disable=SC1090
source "$WORK/helpers.sh"
# That block carries the script's own `set -euo pipefail`, which would abort
# this harness on the first deliberately-failing case. A test file needs to run
# past a failure to report it.
set +eu +o pipefail

# --- sha256_check ------------------------------------------------------------
printf 'croft integrity test' > "$WORK/blob.bin"
# Computed independently of the helper, so a broken helper cannot agree with
# itself. Falls back the same way the helper does, for the same portability
# reason.
if command -v sha256sum >/dev/null 2>&1; then
  EXPECTED="$(sha256sum "$WORK/blob.bin" | awk '{print $1}')"
else
  EXPECTED="$(shasum -a 256 "$WORK/blob.bin" | awk '{print $1}')"
fi

sha256_check "$EXPECTED" "$WORK/blob.bin"
check $? "sha256_check accepts a matching hash"

sha256_check "$(printf '%s' "$EXPECTED" | tr 'a-f' 'A-F')" "$WORK/blob.bin"
check $? "sha256_check is case-insensitive (hashes get pasted in uppercase)"

if sha256_check "$(printf '0%.0s' {1..64})" "$WORK/blob.bin"; then
  bad "sha256_check rejects a mismatching hash"
else
  ok "sha256_check rejects a mismatching hash"
fi

# The bug this replaced: `sha256sum -c` is GNU-only and absent on stock macOS.
if grep -q 'sha256sum -c' "$BOOTSTRAP"; then
  bad "no bare 'sha256sum -c' (fails on macOS)"
else
  ok "no bare 'sha256sum -c' (fails on macOS)"
fi

# --- no shell re-parsing of arguments ---------------------------------------
if grep -qE '^\s*run\(\)\s*\{.*eval' "$BOOTSTRAP"; then
  bad "run() executes directly rather than through eval"
else
  ok "run() executes directly rather than through eval"
fi

# An argument that would be catastrophic under eval must survive as one word.
marker="$WORK/pwned"
# Read by the run() sourced from bootstrap_install.sh, which shellcheck cannot
# see from this file.
# shellcheck disable=SC2034
DRY_RUN=0
run printf '%s\n' "; touch $marker" > "$WORK/out.txt"
if [ -e "$marker" ]; then
  bad "run does not let an argument become a second command"
else
  ok "run does not let an argument become a second command"
fi
grep -qF "; touch $marker" "$WORK/out.txt"
check $? "run passes an argument through intact"

# --- WSL detection is reachable ---------------------------------------------
# WSL reports `uname -s` as Linux, so a WSL branch in the fallback arm of the
# case can never run. It has to sit inside the Linux arm.
if awk '/^case "\$OS" in/,/^esac/' "$BOOTSTRAP" | grep -A3 'Linux)' | grep -q 'microsoft'; then
  ok "WSL is detected inside the Linux branch (where it is reachable)"
else
  bad "WSL is detected inside the Linux branch (where it is reachable)"
fi

# --- --dry-run writes nothing -----------------------------------------------
DRY_DIR="$WORK/dry"
mkdir -p "$DRY_DIR"
before="$(find "$DRY_DIR" | sort)"

env -i PATH="$PATH" HOME="$WORK/home" \
    LLM_INSTALL_DIR="$DRY_DIR" LLM_NO_WEBUI=1 LLM_NO_ENGINES=1 \
    bash "$BOOTSTRAP" --dry-run --yes --mode native >"$WORK/dry.log" 2>&1
dry_rc=$?

after="$(find "$DRY_DIR" | sort)"
if [ "$before" = "$after" ]; then
  ok "--dry-run leaves the install directory untouched"
else
  bad "--dry-run leaves the install directory untouched"
  diff <(printf '%s\n' "$before") <(printf '%s\n' "$after") | head -10
fi

[ ! -d "$DRY_DIR/.venv" ]; check $? "--dry-run creates no virtualenv"

grep -q '\[dry-run\]' "$WORK/dry.log"
check $? "--dry-run says what it would have done"

if [ "$dry_rc" -ne 0 ]; then
  printf '  \033[0;33mnote\033[0m --dry-run exited %s; tail of its log:\n' "$dry_rc"
  tail -5 "$WORK/dry.log" | sed 's/^/         /'
fi

# --- syntax ------------------------------------------------------------------
for f in "$BOOTSTRAP" "$PROJECT_DIR/webui.sh"; do
  bash -n "$f"
  check $? "$(basename "$f") parses"
done

echo
printf '%s passed, %s failed\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
