#!/usr/bin/env bash
# bootstrap_install.sh — cross-platform bootstrap for Local LLM Chat.
#
#   * Detects OS/arch and available accelerators.
#   * Prompts for install mode (docker | native) unless --mode is given.
#   * Installs or *instructs on* prerequisites (Docker, NVIDIA toolkit, python3).
#   * Fetches/uses the installer package and verifies its SHA256 checksum.
#   * Idempotent, non-destructive, supports --dry-run and --yes.
#
# Usage:
#   ./bootstrap_install.sh                      # interactive
#   ./bootstrap_install.sh --mode native --yes  # unattended
#   ./bootstrap_install.sh --dry-run            # preview only
set -euo pipefail

# ---- defaults ---------------------------------------------------------------
# Resolve the directory this script actually lives in, so a checkout that
# already ships the installer is used in place rather than re-fetched.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

MODE=""
ASSUME_YES=0
DRY_RUN=0
REPO_URL="${LLM_REPO_URL:-https://github.com/example/local-llm-chat}"
PKG_URL="${LLM_PKG_URL:-}"                 # optional tarball URL
PKG_SHA256="${LLM_PKG_SHA256:-}"           # expected checksum of the tarball

# Where to install. Precedence:
#   1. $LLM_INSTALL_DIR when set explicitly (env override).
#   2. The script's own directory when it already contains the installer —
#      i.e. it's being run from inside a real checkout, so install in place
#      and skip any fetch/clone.
#   3. $HOME/local-llm-chat as the remote-bootstrap fallback.
# A --dir argument (parsed below) still overrides all of these.
if [ -n "${LLM_INSTALL_DIR:-}" ]; then
  INSTALL_DIR="$LLM_INSTALL_DIR"
elif [ -f "$SCRIPT_DIR/installer/main.py" ]; then
  INSTALL_DIR="$SCRIPT_DIR"
else
  INSTALL_DIR="$HOME/local-llm-chat"
fi

log()  { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }
run()  { if [ "$DRY_RUN" -eq 1 ]; then echo "[dry-run] $*"; else eval "$@"; fi; }

confirm() {
  [ "$ASSUME_YES" -eq 1 ] && return 0
  read -r -p "$1 [y/N] " ans
  case "$ans" in y|Y|yes|Yes) return 0 ;; *) return 1 ;; esac
}

# ---- args -------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --dir) INSTALL_DIR="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) err "Unknown arg: $1"; exit 2 ;;
  esac
done

# Project-local virtualenv (resolved after --dir is applied). The installer and
# host-side model download run through this venv's Python.
VENV_DIR="$INSTALL_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

# Where models + the installer's state file live, mirroring the Python side
# (installer/main.py: LLM_MODELS_DIR, default "models"). Absolute paths are used
# as-is; relative ones are under INSTALL_DIR (the installer's working dir).
case "${LLM_MODELS_DIR:-models}" in
  /*) MODELS_DIR="${LLM_MODELS_DIR}" ;;
  *)  MODELS_DIR="$INSTALL_DIR/${LLM_MODELS_DIR:-models}" ;;
esac
INSTALLED_JSON="$MODELS_DIR/installed.json"      # written by the installer on success
SESSION_FILE="$INSTALL_DIR/.bootstrap_session"   # our picker/progress journal

# ---- OS / arch detection ----------------------------------------------------
OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS" in
  Linux)  PLATFORM="linux" ;;
  Darwin) PLATFORM="macos" ;;
  *) if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then PLATFORM="wsl";
     else err "Unsupported OS: $OS (use WSL2 on Windows)"; exit 1; fi ;;
esac
log "Detected platform: $PLATFORM ($ARCH)"

# Accelerator hints
ACCEL="cpu"
if command -v nvidia-smi >/dev/null 2>&1; then ACCEL="cuda";
elif command -v rocm-smi >/dev/null 2>&1; then ACCEL="rocm";
elif [ "$PLATFORM" = "macos" ] && [ "$ARCH" = "arm64" ]; then ACCEL="mps"; fi
log "Detected accelerator: $ACCEL"

# ---- pick mode --------------------------------------------------------------
if [ -z "$MODE" ]; then
  if [ "$ASSUME_YES" -eq 1 ]; then
    MODE=$(command -v docker >/dev/null 2>&1 && echo docker || echo native)
  else
    echo "Choose install mode:"
    echo "  1) docker  (recommended: isolated, reproducible)"
    echo "  2) native  (Python virtualenv on this host)"
    read -r -p "Selection [1/2]: " sel
    case "$sel" in 2) MODE="native" ;; *) MODE="docker" ;; esac
  fi
fi
log "Install mode: $MODE"

# ---- prerequisites ----------------------------------------------------------
need_cmd() { command -v "$1" >/dev/null 2>&1; }

ensure_python() {
  if ! need_cmd python3; then
    warn "python3 not found."
    case "$PLATFORM" in
      linux|wsl) confirm "Install python3 + venv via apt?" &&
        run "sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip" ;;
      macos) confirm "Install python via Homebrew?" && run "brew install python" ;;
    esac
  fi
  need_cmd python3 || { err "python3 is required."; exit 1; }
  log "python3: $(python3 --version)"
}

ensure_docker() {
  if ! need_cmd docker; then
    warn "Docker not found."
    case "$PLATFORM" in
      linux|wsl)
        if confirm "Install Docker Engine via the official convenience script?"; then
          run "curl -fsSL https://get.docker.com | sh"
          run "sudo usermod -aG docker \"$USER\" || true"
          warn "Log out/in (or 'newgrp docker') for group changes to take effect."
        else
          err "Docker required for docker mode. Re-run with --mode native to skip."
          exit 1
        fi ;;
      macos)
        err "Install Docker Desktop for Mac: https://www.docker.com/products/docker-desktop/"
        exit 1 ;;
    esac
  fi
  log "docker: $(docker --version 2>/dev/null || echo 'pending group refresh')"

  # NVIDIA Container Toolkit (only when CUDA present and on Linux)
  if [ "$ACCEL" = "cuda" ] && [ "$PLATFORM" != "macos" ]; then
    if ! docker info 2>/dev/null | grep -qi nvidia; then
      warn "NVIDIA Container Toolkit not detected — GPUs won't be visible to containers."
      if confirm "Install nvidia-container-toolkit via apt?"; then
        run "distribution=\$(. /etc/os-release; echo \$ID\$VERSION_ID)"
        run "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        run "curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        run "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        run "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
      fi
    else
      log "NVIDIA Container Toolkit: present"
    fi
  fi
}

# The installer and the host-side model download both run on Python in BOTH
# modes (models are fetched to ./models on the host, then mounted into any
# containers), so Python is always required. Docker is additionally required
# only for docker mode.
ensure_python
case "$MODE" in
  docker) ensure_docker ;;
  native) : ;;
  *) err "Invalid mode: $MODE"; exit 2 ;;
esac

# ---- fetch installer package + verify checksum ------------------------------
fetch_package() {
  if [ -f "$INSTALL_DIR/installer/main.py" ]; then
    log "Installer already present at $INSTALL_DIR (idempotent skip)."
    return 0
  fi
  mkdir -p "$INSTALL_DIR"
  if [ -n "$PKG_URL" ]; then
    log "Downloading installer package: $PKG_URL"
    run "curl -fL --retry 5 -C - -o /tmp/local-llm.tar.gz \"$PKG_URL\""
    if [ -n "$PKG_SHA256" ] && [ "$DRY_RUN" -eq 0 ]; then
      echo "$PKG_SHA256  /tmp/local-llm.tar.gz" | sha256sum -c - \
        || { err "Checksum verification FAILED — aborting."; exit 1; }
      log "Checksum verified."
    else
      warn "No PKG_SHA256 provided — skipping integrity check (not recommended)."
    fi
    run "tar -xzf /tmp/local-llm.tar.gz -C \"$INSTALL_DIR\" --strip-components=1"
  else
    log "No package URL set; cloning repo: $REPO_URL"
    run "git clone --depth 1 \"$REPO_URL\" \"$INSTALL_DIR\""
  fi
}
fetch_package

# ---- install core Python dependencies BEFORE any download ------------------
# The installer imports huggingface_hub / requests / rich to detect hardware and
# to download models. Without them the download aborts with
# "huggingface_hub not installed". So we create a project-local venv and install
# the core requirements up front, then run the installer through that venv. This
# also provides the uvicorn used to launch the web UI later.
ensure_venv_deps() {
  local pip="$VENV_DIR/bin/pip" req="$INSTALL_DIR/requirements.txt"

  if [ "$DRY_RUN" -eq 1 ]; then
    log "[dry-run] would set up venv + install dependencies:"
    echo "[dry-run] python3 -m venv \"$VENV_DIR\""
    echo "[dry-run] \"$pip\" install --upgrade pip"
    echo "[dry-run] \"$pip\" install -r \"$req\""
    return 0
  fi

  if [ ! -x "$VENV_PY" ]; then
    log "Creating Python virtualenv at $VENV_DIR ..."
    if ! python3 -m venv "$VENV_DIR"; then
      err "Could not create a virtualenv. On Debian/Ubuntu install python3-venv:"
      err "  sudo apt-get install -y python3-venv"
      exit 1
    fi
  fi

  log "Installing core dependencies (huggingface_hub, requests, rich, uvicorn ...)"
  "$pip" install --upgrade pip >/dev/null 2>&1 || warn "pip self-upgrade skipped."
  if [ -f "$req" ]; then
    if ! "$pip" install -r "$req"; then
      err "Dependency install failed. The machine may be offline or pip is blocked."
      err "Model download needs these packages — fix connectivity and re-run."
      exit 1
    fi
  else
    warn "requirements.txt missing; installing the download-critical packages only."
    "$pip" install "huggingface_hub>=0.25" "requests>=2.31" rich || {
      err "Failed to install huggingface_hub/requests — cannot download models."; exit 1; }
  fi

  # Prove the download backend is importable before we promise a working download.
  if ! "$VENV_PY" -c "import huggingface_hub, requests" 2>/dev/null; then
    err "huggingface_hub is still not importable in the venv — aborting before download."
    exit 1
  fi
  log "✓ Dependencies ready — the machine can download models."
}
ensure_venv_deps

# ---- hand off to the Python installer --------------------------------------
log "Launching Python installer (mode=$MODE, accel=$ACCEL)"
EXTRA=""
[ "$ASSUME_YES" -eq 1 ] && EXTRA="$EXTRA --yes"
[ "$DRY_RUN" -eq 1 ] && EXTRA="$EXTRA --dry-run"

# Run the installer through the venv's Python when available (so huggingface_hub
# and friends are importable), falling back to system python3 otherwise.
pyinst() {
  local py="python3"
  [ -x "$VENV_PY" ] && py="$VENV_PY"
  (cd "$INSTALL_DIR" && "$py" -m installer.main "$@")
}

# Show the human-readable recommendation report. Run directly (not via `run`)
# so it prints even under --dry-run — reading hardware never changes anything.
pyinst $EXTRA recommend || warn "recommend step reported an issue."

# Capture a machine-readable view of the ranked models for the picker.
RECO_JSON="$(pyinst recommend --json 2>/dev/null || true)"

# Turn the JSON into tab-separated rows: idx, id, kind, fits(1/0), params,
# disk, license, gated. First line carries the recommended default id.
MENU="$(printf '%s' "$RECO_JSON" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
print("DEFAULT\t%s" % (d.get("default_text_model") or ""))
i = 0
for kind in ("text", "image", "video"):
    for e in d.get("recommendations", {}).get(kind, []):
        i += 1
        print("%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
            i, e["id"], kind, "1" if e["fits"] else "0",
            ("%g" % e["params_billion"]), ("%.1f" % e["approx_disk_gb"]),
            e["license"], "gated" if e["gated"] else ""))
' 2>/dev/null || true)"

DEFAULT_MODEL="$(printf '%s\n' "$MENU" | awk -F'\t' '$1=="DEFAULT"{print $2}')"
ROWS="$(printf '%s\n' "$MENU" | awk -F'\t' '$1!="DEFAULT" && NF>=8')"

if [ -z "$ROWS" ]; then
  warn "Could not build a model list from the recommender."
  log  "Install manually later: python3 -m installer.main install --mode $MODE"
  log  "Bootstrap complete."
  exit 0
fi

# ---- session / on-disk state (resume support) ------------------------------
# A model is INSTALLED if it is recorded in models/installed.json (written by the
# installer only after a fully successful download). It is PARTIAL if its model
# directory exists on disk but it is not yet recorded — i.e. a download that was
# interrupted; huggingface_hub/.part resume will continue it on re-install.
get_installed_ids() {
  [ -f "$INSTALLED_JSON" ] || return 0
  local py="python3"; [ -x "$VENV_PY" ] && py="$VENV_PY"
  "$py" - "$INSTALLED_JSON" <<'PY' 2>/dev/null || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
print("\n".join(d.get("models", {}).keys()))
PY
}
INSTALLED_LIST="$(get_installed_ids)"

is_installed() { printf '%s\n' "$INSTALLED_LIST" | grep -qxF "$1"; }
is_partial()   { [ -d "$MODELS_DIR/$1" ] && ! is_installed "$1"; }
# Status tag for a model id, given its fit flag (1/0).
status_tag() {
  if   is_installed "$1"; then echo "✓ installed"
  elif is_partial   "$1"; then echo "◐ resume"
  elif [ "$2" = "0" ];    then echo "won't fit"
  else                          echo "fits"
  fi
}

# Session journal: one "id<TAB>pending|done" line per selected model.
session_write_pending() {   # args: model ids -> (re)write file, all pending
  : > "$SESSION_FILE"
  local id
  for id in "$@"; do printf '%s\tpending\n' "$id" >> "$SESSION_FILE"; done
}
session_mark_done() {       # $1 = id -> flip its line to done
  [ -f "$SESSION_FILE" ] || return 0
  awk -F'\t' -v id="$1" 'BEGIN{OFS="\t"} $1==id{$2="done"} {print}' \
      "$SESSION_FILE" > "$SESSION_FILE.tmp" 2>/dev/null && mv "$SESSION_FILE.tmp" "$SESSION_FILE"
}
session_pending_ids() {     # print ids not yet done
  [ -f "$SESSION_FILE" ] || return 0
  awk -F'\t' '$2!="done"{print $1}' "$SESSION_FILE"
}

# pick_kind <kind> <default-id-for-kind>
# Renders an interactive picker for ONE model kind and writes the chosen model
# ids to stdout (one per line). All menu text and prompts go to stderr, so the
# caller captures only the ids via  sel="$(pick_kind ...)".
#   numbers (space/comma) -> those models    d -> the recommended default
#   a -> every model of this kind that fits   s -> skip this kind (emit nothing)
pick_kind() {
  local kind="$1" kdefault="$2"
  local krows i n k sel tok pid hint mark g
  local idx id kk fits params disk lic gated
  local -a KIDS KFIT

  krows="$(printf '%s\n' "$ROWS" | awk -F'\t' -v k="$kind" '$3==k')"
  if [ -z "$krows" ]; then
    printf '\n[bootstrap] No %s models available for this machine — skipping.\n' \
           "$kind" >&2
    return 0
  fi

  # Local, per-kind numbering (1..n) -> id / fits.
  n=0
  while IFS=$'\t' read -r idx id kk fits params disk lic gated; do
    n=$((n + 1)); KIDS[$n]="$id"; KFIT[$n]="$fits"
  done <<EOF
$krows
EOF

  {
    printf '\n===== %s models =====\n' \
           "$(printf '%s' "$kind" | tr '[:lower:]' '[:upper:]')"
    i=0
    while IFS=$'\t' read -r idx id kk fits params disk lic gated; do
      i=$((i + 1))
      mark="$(status_tag "$id" "$fits")"
      g="";        [ -n "$gated" ] && g="  [gated]"
      printf "  %2s) %-24s %6sB  %6sGB  %-24s %-11s%s\n" \
             "$i" "$id" "$params" "$disk" "$lic" "$mark" "$g"
    done <<EOF
$krows
EOF
    echo
    [ -n "$kdefault" ] && echo "   d) recommended default  ->  $kdefault"
    echo "   a) all $kind models that fit (already-installed skipped, partials resumed)"
    echo "   s) skip $kind"
  } >&2

  # Unattended (--yes): take the default when there is one, else skip.
  if [ "$ASSUME_YES" -eq 1 ]; then
    [ -n "$kdefault" ] && printf '%s\n' "$kdefault"
    return 0
  fi

  hint="numbers"
  [ -n "$kdefault" ] && hint="$hint, d=default"
  hint="$hint, a=all-fit, s=skip"
  read -r -p "Select $kind model(s) [$hint]: " sel || sel="s"

  case "$sel" in
    s|S|"") : ;;                                   # skip -> nothing
    d|D) [ -n "$kdefault" ] && printf '%s\n' "$kdefault" ;;
    a|A)
      # All that fit, minus already-installed; plus any partials to resume.
      k=1
      while [ "$k" -le "$n" ]; do
        pid="${KIDS[$k]}"
        if is_installed "$pid"; then :   # skip completed
        elif [ "${KFIT[$k]}" = "1" ] || is_partial "$pid"; then printf '%s\n' "$pid"
        fi
        k=$((k + 1))
      done ;;
    *)
      for tok in $(printf '%s' "$sel" | tr ',' ' '); do
        case "$tok" in
          ''|*[!0-9]*) printf '[warn] ignoring invalid choice: %s\n' "$tok" >&2 ;;
          *)
            pid="${KIDS[$tok]:-}"
            if [ -z "$pid" ]; then
              printf '[warn] no %s model numbered %s\n' "$kind" "$tok" >&2
            elif is_installed "$pid"; then
              printf '[note] %s already installed — re-selecting to re-verify.\n' "$pid" >&2
              printf '%s\n' "$pid"
            else
              printf '%s\n' "$pid"
            fi ;;
        esac
      done ;;
  esac
}

# ---- resume a previous session if one is unfinished ------------------------
# If a prior run selected models but did not finish them all (interrupted, a
# failed download, Ctrl-C), offer to resume exactly those. Choosing "fresh"
# discards the journal and falls through to the normal pickers.
SELECTED=()
RESUMING=0
PENDING="$(session_pending_ids)"
if [ -n "$PENDING" ]; then
  echo
  warn "A previous session has unfinished models:"
  while IFS= read -r pid; do
    [ -z "$pid" ] && continue
    tag="pending"; is_partial "$pid" && tag="partial download — will resume"
    is_installed "$pid" && tag="now installed"
    printf '   - %-24s (%s)\n' "$pid" "$tag" >&2
  done <<EOF
$PENDING
EOF
  ans="r"
  [ "$ASSUME_YES" -eq 0 ] && { read -r -p "Resume this session? [R=resume / f=start fresh]: " ans || ans="r"; }
  case "$ans" in
    f|F|fresh|Fresh)
      log "Starting fresh — discarding the previous session."
      [ "$DRY_RUN" -eq 0 ] && rm -f "$SESSION_FILE" ;;
    *)
      RESUMING=1
      while IFS= read -r pid; do
        [ -n "$pid" ] && ! is_installed "$pid" && SELECTED+=("$pid")
      done <<EOF
$PENDING
EOF
      # Anything already completed since last time needs no reinstall.
      if [ "${#SELECTED[@]}" -eq 0 ]; then
        log "Everything from the previous session is already installed."
        [ "$DRY_RUN" -eq 0 ] && rm -f "$SESSION_FILE"
        RESUMING=0
      else
        log "Resuming previous session: ${SELECTED[*]}"
      fi ;;
  esac
fi

# ---- run the three pickers in order: text -> image -> video ----------------
# Skipped entirely when resuming. Loops so that skipping ALL of them offers a
# restart (or quit).
while [ "$RESUMING" -eq 0 ]; do
  SELECTED=()
  for id in \
      $(pick_kind text  "$DEFAULT_MODEL") \
      $(pick_kind image "") \
      $(pick_kind video ""); do
    SELECTED+=("$id")
  done

  # De-duplicate while preserving order (a model could be picked twice).
  UNIQ=()
  for id in "${SELECTED[@]:-}"; do
    [ -z "$id" ] && continue
    seen=0
    if [ "${#UNIQ[@]}" -gt 0 ]; then
      for u in "${UNIQ[@]}"; do [ "$u" = "$id" ] && { seen=1; break; }; done
    fi
    [ "$seen" -eq 0 ] && UNIQ+=("$id")
  done
  SELECTED=("${UNIQ[@]:-}")

  [ "${#SELECTED[@]}" -gt 0 ] && [ -n "${SELECTED[0]:-}" ] && break

  # Nothing chosen across all three kinds.
  if [ "$ASSUME_YES" -eq 1 ]; then
    warn "Unattended: nothing selected — skipping install."
    log "Bootstrap complete."
    exit 0
  fi
  echo
  warn "You skipped text, image and video — no models selected."
  read -r -p "Restart the selection or quit? [r=restart / q=quit]: " again || again="q"
  case "$again" in
    r|R|restart|Restart) continue ;;
    *)
      log "No models selected. Install later with:"
      log "  python3 -m installer.main install --mode $MODE --model <id>"
      log "Bootstrap complete."
      exit 0 ;;
  esac
done

# Persist the selection so an interruption below can be resumed next run.
# (When resuming, the journal already holds these as pending.)
[ "$RESUMING" -eq 0 ] && [ "$DRY_RUN" -eq 0 ] && session_write_pending "${SELECTED[@]}"

# ---- install every chosen model; one failure warns and the rest proceed ----
log "Selected models: ${SELECTED[*]}"
for mid in "${SELECTED[@]}"; do
  log "Installing '$mid' (mode=$MODE) ..."
  if pyinst $EXTRA install --mode "$MODE" --model "$mid"; then
    log "✓ $mid done."
    [ "$DRY_RUN" -eq 1 ] || session_mark_done "$mid"
  else
    warn "Install of '$mid' did not complete — it will be offered for resume next run."
  fi
done

# Clear the journal once nothing is left pending; otherwise keep it for resume.
if [ "$DRY_RUN" -eq 0 ] && [ -z "$(session_pending_ids)" ]; then
  rm -f "$SESSION_FILE"
fi

# ---- install the inference engines so the models can actually RUN ----------
# Downloaded weights are inert without a runtime. Text/GGUF chat runs via the
# embedded llama-cpp-python; image/video via the diffusers stack. We install
# whichever the installed models require, into the SAME venv the web app uses,
# so the studio works out of the box. Native mode only (docker runs models in
# containers). Skip with LLM_NO_ENGINES=1 (chat can also use an OpenAI endpoint).
ensure_engines() {
  [ "${LLM_NO_ENGINES:-0}" = "1" ] && { log "Skipping inference engines (LLM_NO_ENGINES=1)."; return 0; }
  if [ "$MODE" != "native" ]; then
    log "Engines: docker mode runs models in containers — skipping host engine install."
    return 0
  fi
  [ -x "$VENV_PY" ] || { warn "No venv found — cannot install engines."; return 0; }

  # Which engines do the installed models call for?
  local flags need_llama need_diff
  flags="$("$VENV_PY" - "$INSTALLED_JSON" <<'PY' 2>/dev/null
import json, sys
try:
    models = json.load(open(sys.argv[1])).get("models", {})
except Exception:
    models = {}
text = any(m.get("kind") == "text" and m.get("format") == "gguf" for m in models.values())
imgv = any(m.get("kind") in ("image", "video") for m in models.values())
print(("1" if text else "0") + " " + ("1" if imgv else "0"))
PY
)" || true
  if [ -z "$flags" ]; then need_llama=0; need_diff=0
  else need_llama="${flags%% *}"; need_diff="${flags##* }"; fi

  # Already present?
  local want_llama=0 want_diff=0
  [ "$need_llama" = "1" ] && ! "$VENV_PY" -c "import llama_cpp" 2>/dev/null && want_llama=1
  [ "$need_diff"  = "1" ] && ! "$VENV_PY" -c "import torch, diffusers" 2>/dev/null && want_diff=1
  [ "$want_llama$want_diff" = "00" ] && { log "Inference engines already in place."; return 0; }

  local msg="Engines needed to run your models:"
  [ "$want_llama" = "1" ] && msg="$msg  chat=llama-cpp-python (compiles)"
  [ "$want_diff"  = "1" ] && msg="$msg  images=torch+diffusers (large)"
  log "$msg"
  if [ "$ASSUME_YES" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    read -r -p "Install them now? [Y/n]: " a || a="y"
    case "$a" in n|N|no|No)
      log "Skipped. Install later:  $VENV_DIR/bin/pip install <engine>"; return 0 ;;
    esac
  fi

  local pip="$VENV_DIR/bin/pip"
  if [ "$want_llama" = "1" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then echo "[dry-run] $pip install llama-cpp-python"
    else
      log "Building llama-cpp-python (this compiles, a few minutes) ..."
      "$pip" install llama-cpp-python \
        && log "✓ chat engine ready." \
        || warn "llama-cpp-python failed — chat can still use an OpenAI endpoint (Settings)."
    fi
  fi
  if [ "$want_diff" = "1" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then echo "[dry-run] $pip install torch diffusers transformers accelerate safetensors pillow imageio imageio-ffmpeg"
    else
      log "Installing torch + diffusers (large download) ..."
      "$pip" install torch diffusers transformers accelerate safetensors pillow imageio imageio-ffmpeg \
        && log "✓ image engine ready." \
        || warn "diffusers stack failed — image generation will be unavailable."
    fi
  fi
}
ensure_engines

# ---- start the chat + image web app in the background ----------------------
# webui/app.py is a FastAPI status/control panel. We launch it detached so it
# keeps running after the script exits, and print a browser URL. Override the
# bind address/port with LLM_WEBUI_HOST / LLM_WEBUI_PORT; set LLM_NO_WEBUI=1 to
# skip. Default host is 127.0.0.1 (local only) since the panel can control
# services — set LLM_WEBUI_HOST=0.0.0.0 to reach it from other machines.
WEBUI_HOST="${LLM_WEBUI_HOST:-127.0.0.1}"
WEBUI_PORT="${LLM_WEBUI_PORT:-8090}"
WEBUI_URL="http://${WEBUI_HOST}:${WEBUI_PORT}"
WEBUI_LOG="$INSTALL_DIR/webui.log"
WEBUI_PID="$INSTALL_DIR/.webui.pid"

webui_running() {
  [ -f "$WEBUI_PID" ] && kill -0 "$(cat "$WEBUI_PID" 2>/dev/null)" 2>/dev/null
}

start_webui_native() {
  local uvicorn_bin
  uvicorn_bin="$INSTALL_DIR/.venv/bin/uvicorn"
  [ -x "$uvicorn_bin" ] || uvicorn_bin="$(command -v uvicorn 2>/dev/null || true)"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] (cd \"$INSTALL_DIR\" && nohup ${uvicorn_bin:-uvicorn} webui.app:app --host $WEBUI_HOST --port $WEBUI_PORT >\"$WEBUI_LOG\" 2>&1 &)"
    return 0
  fi
  if [ -z "$uvicorn_bin" ]; then
    warn "uvicorn not found — after 'pip install fastapi uvicorn' start it with:"
    warn "  (cd \"$INSTALL_DIR\" && uvicorn webui.app:app --host $WEBUI_HOST --port $WEBUI_PORT)"
    return 1
  fi
  # Detach with nohup so it survives this script; record the pid for stop/status.
  ( cd "$INSTALL_DIR" && nohup "$uvicorn_bin" webui.app:app \
      --host "$WEBUI_HOST" --port "$WEBUI_PORT" >"$WEBUI_LOG" 2>&1 & echo $! >"$WEBUI_PID" )
}

start_webui_docker() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] (cd \"$INSTALL_DIR\" && docker compose -f docker/docker-compose.yml --profile manager up -d --build manager)"
    return 0
  fi
  ( cd "$INSTALL_DIR" && docker compose -f docker/docker-compose.yml \
      --profile manager up -d --build manager )
}

wait_for_webui() {
  command -v curl >/dev/null 2>&1 || { sleep 2; return 0; }
  local i=0
  while [ "$i" -lt 20 ]; do
    curl -fsS "$WEBUI_URL" >/dev/null 2>&1 && return 0
    i=$((i + 1)); sleep 1
  done
  return 1
}

open_browser() {
  [ "$ASSUME_YES" -eq 1 ] && return 0   # don't hijack a browser in unattended runs
  if   command -v open     >/dev/null 2>&1; then (open     "$WEBUI_URL" >/dev/null 2>&1 &)
  elif command -v xdg-open >/dev/null 2>&1; then (xdg-open "$WEBUI_URL" >/dev/null 2>&1 &)
  fi
}

start_webui() {
  if [ "${LLM_NO_WEBUI:-0}" = "1" ]; then
    log "Skipping web UI (LLM_NO_WEBUI=1). Start it later with:"
    log "  (cd \"$INSTALL_DIR\" && uvicorn webui.app:app --host $WEBUI_HOST --port $WEBUI_PORT)"
    return 0
  fi
  if webui_running; then
    log "Web UI already running (pid $(cat "$WEBUI_PID")) at $WEBUI_URL"
    return 0
  fi
  if [ "$ASSUME_YES" -eq 0 ]; then
    read -r -p "Start the chat web app in the background now? [Y/n]: " ans || ans="y"
    case "$ans" in
      n|N|no|No)
        log "Not starting the web UI. Start it later with:"
        log "  (cd \"$INSTALL_DIR\" && uvicorn webui.app:app --host $WEBUI_HOST --port $WEBUI_PORT)"
        return 0 ;;
    esac
  fi

  log "Starting web UI ($MODE) ..."
  if [ "$MODE" = "docker" ]; then
    start_webui_docker || { warn "Could not start the manager container."; return 0; }
  else
    start_webui_native || return 0
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    log "[dry-run] web UI would be available at $WEBUI_URL"
    return 0
  fi

  if wait_for_webui; then
    log "✓ Chat web app is running in the background. Open it in any browser:"
    log "     $WEBUI_URL"
    open_browser
  else
    warn "Web UI did not respond yet; it may still be starting. Logs: $WEBUI_LOG"
    log  "Once up, open it at: $WEBUI_URL"
  fi
  if [ "$MODE" = "docker" ]; then
    log "  stop it with:  (cd \"$INSTALL_DIR\" && docker compose -f docker/docker-compose.yml stop manager)"
  else
    log "  logs: $WEBUI_LOG   stop it with:  kill \$(cat \"$WEBUI_PID\")"
  fi
}

start_webui || true
log "Bootstrap complete."
