#!/usr/bin/env bash
# webui.sh — start / stop / restart / status / logs for the Local LLM Chat web app.
#
#   ./webui.sh start          # launch in the background
#   ./webui.sh stop           # stop it
#   ./webui.sh restart        # stop then start
#   ./webui.sh status         # is it up? old panel or chat app?
#   ./webui.sh logs [-f]      # show (or follow) the log
#
# Native by default: manages a uvicorn process via .webui.pid + webui.log,
# reusing the same files the bootstrap uses. Set LLM_WEBUI_MODE=docker (or pass
# --docker) to drive the compose `manager` service instead.
#
# Chat and diffusion both run in their own processes now, so this script brings
# up three things: ollama (chat), the sd worker (image/video) and the web app.
# Only the web app holds the UI; the other two hold the weights, which is the
# point — killing them is how memory actually comes back on a 32 GB Mac.
#
# Env: LLM_WEBUI_HOST (127.0.0.1), LLM_WEBUI_PORT (8090), LLM_VENV_DIR (.venv),
#      LLM_SD_PORT (7862), OLLAMA_KEEP_ALIVE (2m).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

HOST="${LLM_WEBUI_HOST:-127.0.0.1}"
PORT="${LLM_WEBUI_PORT:-8090}"
# A wildcard bind is not an address anyone can open; show loopback instead.
case "$HOST" in 0.0.0.0|::|"*") SHOWN_HOST="127.0.0.1" ;; *) SHOWN_HOST="$HOST" ;; esac
URL="http://${SHOWN_HOST}:${PORT}"
TOKEN_FILE="${LLM_WEBUI_TOKEN_FILE:-$SCRIPT_DIR/.webui_token}"
VENV_DIR="${LLM_VENV_DIR:-$SCRIPT_DIR/.venv}"
PIDFILE="$SCRIPT_DIR/.webui.pid"
LOGFILE="$SCRIPT_DIR/webui.log"
SD_PORT="${LLM_SD_PORT:-7862}"
SD_PIDFILE="$SCRIPT_DIR/.sdworker.pid"
SD_LOGFILE="$SCRIPT_DIR/sdworker.log"
OLLAMA_LOGFILE="$SCRIPT_DIR/ollama.log"
# Short by design: an idle 5 GB of chat weights is 5 GB the diffusion worker
# can't have. The app also unloads on demand, so this is just the backstop.
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-2m}"
COMPOSE="docker compose -f docker/docker-compose.yml"
MODE="${LLM_WEBUI_MODE:-native}"

log()  { printf '\033[1;34m[webui]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n'  "$*" >&2; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

# ---- helpers ---------------------------------------------------------------
pid_running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }

pid_on_port() {  # PID listening on $1, or empty
  command -v lsof >/dev/null 2>&1 || { echo ""; return 0; }
  lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1 || true
}

port_pid() { pid_on_port "$PORT"; }

# Sets UVICORN as a command array; 0 if found, 1 if not.
#
# The venv's Python is preferred over its bin/uvicorn script. A console script
# hardcodes the absolute path of the venv that built it into its shebang, so a
# renamed or copied project directory leaves bin/uvicorn present and executable
# but dead ("bad interpreter"). Testing `import uvicorn` through the interpreter
# checks the thing that actually has to work.
resolve_uvicorn() {
  if [ -x "$VENV_DIR/bin/python" ] && \
     "$VENV_DIR/bin/python" -c "import uvicorn" >/dev/null 2>&1; then
    UVICORN=("$VENV_DIR/bin/python" -m uvicorn); return 0
  fi
  if command -v uvicorn >/dev/null 2>&1; then UVICORN=("$(command -v uvicorn)"); return 0; fi
  return 1
}

ensure_uvicorn() {  # resolve, else create a venv and install requirements
  resolve_uvicorn && return 0
  warn "uvicorn not found."
  # A bin/python that exists but cannot run means the base interpreter is gone;
  # reusing it would fail every install below, so start over.
  if [ -x "$VENV_DIR/bin/python" ] && \
     ! "$VENV_DIR/bin/python" -c "import sys" >/dev/null 2>&1; then
    warn "The virtualenv at $VENV_DIR is broken — rebuilding."
    rm -rf "$VENV_DIR"
  fi
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    log "Creating virtualenv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR" || { err "could not create venv (need python3-venv)"; return 1; }
  fi
  log "Installing dependencies ..."
  local py="$VENV_DIR/bin/python"
  "$py" -m pip install -q --upgrade pip || true
  if [ -f requirements.txt ]; then
    "$py" -m pip install -q -r requirements.txt || { err "dependency install failed"; return 1; }
  else
    "$py" -m pip install -q fastapi "uvicorn[standard]" requests || { err "install failed"; return 1; }
  fi
  resolve_uvicorn
}

wait_up() {  # poll without sleep via curl retry; 0 when it answers
  command -v curl >/dev/null 2>&1 || return 0
  curl -fsS --retry 30 --retry-connrefused --retry-delay 1 -o /dev/null "$URL" 2>/dev/null
}

# /api is token-gated (webui/auth.py), so a liveness probe has to present the
# token — otherwise a perfectly healthy app reads as "not the chat app".
read_token() {
  if [ -n "${LLM_WEBUI_TOKEN:-}" ]; then printf '%s' "$LLM_WEBUI_TOKEN"
  elif [ -r "$TOKEN_FILE" ]; then tr -d ' \t\r\n' < "$TOKEN_FILE"
  else printf ''; fi
}

# The URL worth printing: the token rides in the fragment, which the browser
# keeps to itself and never sends to the server.
open_url() {
  local t; t="$(read_token)"
  if [ -n "$t" ]; then printf '%s/#t=%s' "$URL" "$t"; else printf '%s' "$URL"; fi
}

is_chat_app() {
  local t; t="$(read_token)"
  curl -fsS -H "X-LLM-Token: $t" "$URL/api/models" >/dev/null 2>&1
}

# ---- chat backend (ollama) -------------------------------------------------
ensure_ollama() {
  if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    log "ollama: already running (keep_alive stays as that process was started)"
    return 0
  fi
  if ! command -v ollama >/dev/null 2>&1; then
    warn "ollama not installed — chat will be unavailable until you install it"
    warn "  brew install ollama && ollama create <model> -f Modelfile"
    return 0   # image/video still work, so this is not fatal
  fi
  log "Starting ollama (keep_alive $OLLAMA_KEEP_ALIVE) ..."
  nohup ollama serve >"$OLLAMA_LOGFILE" 2>&1 &
  curl -fsS --retry 20 --retry-connrefused --retry-delay 1 -o /dev/null \
       http://127.0.0.1:11434/api/version 2>/dev/null \
    && log "✓ ollama up" || warn "ollama did not answer; see $OLLAMA_LOGFILE"
}

# ---- diffusion worker ------------------------------------------------------
sd_running() { [ -f "$SD_PIDFILE" ] && kill -0 "$(cat "$SD_PIDFILE" 2>/dev/null)" 2>/dev/null; }

sd_start() {
  if sd_running; then log "sd worker: already running (pid $(cat "$SD_PIDFILE"))"; return 0; fi
  local other; other="$(pid_on_port "$SD_PORT")"
  if [ -n "$other" ]; then warn "sd worker port $SD_PORT held by PID $other"; return 0; fi
  log "Starting diffusion worker on port $SD_PORT ..."
  nohup "$VENV_DIR/bin/python" -m webui.sd_server >"$SD_LOGFILE" 2>&1 &
  echo $! > "$SD_PIDFILE"
  # It loads no model at boot, so it answers immediately; a slow reply means
  # torch itself failed to import, which the log will say.
  curl -fsS --retry 30 --retry-connrefused --retry-delay 1 -o /dev/null \
       "http://127.0.0.1:$SD_PORT/health" 2>/dev/null \
    && log "✓ sd worker up" || warn "sd worker not answering; see $SD_LOGFILE"
}

sd_stop() {
  if sd_running; then
    local p; p="$(cat "$SD_PIDFILE")"
    log "Stopping sd worker (pid $p) ..."
    kill "$p" 2>/dev/null || true
    local n=0
    while kill -0 "$p" 2>/dev/null && [ "$n" -lt 20 ]; do n=$((n+1)); sleep 0.25; done
    kill -0 "$p" 2>/dev/null && kill -9 "$p" 2>/dev/null || true
  fi
  rm -f "$SD_PIDFILE"
}

# ---- native mode -----------------------------------------------------------
native_start() {
  if pid_running; then log "Already running (pid $(cat "$PIDFILE")) at $URL"; return 0; fi
  local other; other="$(port_pid)"
  if [ -n "$other" ]; then
    warn "Port $PORT is already in use by PID $other (not managed by this script)."
    ps -p "$other" -o comm= 2>/dev/null | grep -qi docker && \
      warn "It looks like a Docker container. Use: LLM_WEBUI_MODE=docker $0 restart"
    return 1
  fi
  ensure_uvicorn || return 1
  ensure_ollama
  sd_start
  log "Starting chat web app on $URL ..."
  nohup "${UVICORN[@]}" webui.app:app --host "$HOST" --port "$PORT" >"$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  if wait_up; then
    log "✓ Running (pid $(cat "$PIDFILE")). Open:"
    log "     $(open_url)"
    is_chat_app && log "  chat app is live." || warn "  responding, but /api/models missing — check $LOGFILE"
  else
    warn "Started but not responding yet; see $LOGFILE"; return 1
  fi
}

native_stop() {
  local stopped=0
  sd_stop   # the worker holds the diffusion weights; drop it first
  if pid_running; then
    local p; p="$(cat "$PIDFILE")"
    log "Stopping pid $p ..."
    kill "$p" 2>/dev/null || true
    local n=0
    while kill -0 "$p" 2>/dev/null && [ "$n" -lt 20 ]; do n=$((n+1)); sleep 0.25; done
    if kill -0 "$p" 2>/dev/null; then warn "did not exit; forcing"; kill -9 "$p" 2>/dev/null || true; fi
    stopped=1
  fi
  rm -f "$PIDFILE"
  local other; other="$(port_pid)"
  if [ -n "$other" ]; then
    warn "Port $PORT is still held by PID $other (not started by this script)."
    ps -p "$other" -o comm= 2>/dev/null | grep -qi docker && \
      warn "Looks like Docker. Stop it with: LLM_WEBUI_MODE=docker $0 stop"
  elif [ "$stopped" -eq 1 ]; then log "✓ Stopped."
  else log "Not running."; fi
}

native_status() {
  if pid_running; then log "native process: running (pid $(cat "$PIDFILE"))"
  else log "native process: not running (no live pidfile)"; fi
  if sd_running; then log "sd worker: running (pid $(cat "$SD_PIDFILE")) on port $SD_PORT"
  else log "sd worker: not running"; fi
  if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    log "ollama: up — loaded: $(curl -fsS http://127.0.0.1:11434/api/ps \
        | sed -n 's/.*"name":"\([^"]*\)".*/\1/p' | paste -sd, - || echo none)"
  else log "ollama: not running (chat unavailable)"; fi
  local other; other="$(port_pid)"
  [ -n "$other" ] && log "port $PORT listener: PID $other ($(ps -p "$other" -o comm= 2>/dev/null || echo '?'))"
  local code; code="$(curl -s -o /dev/null -w '%{http_code}' "$URL" 2>/dev/null || true)"; code="${code:-000}"
  log "HTTP $URL -> $code"
  if is_chat_app; then log "serving: chat app (/api/models OK) → $(open_url)"
  elif [ "$code" = "200" ]; then warn "serving: something else on $PORT (old panel?) — /api/models missing"
  else log "serving: nothing reachable"; fi
}

# ---- docker mode -----------------------------------------------------------
docker_start()  { log "Building + starting compose 'manager' ..."; $COMPOSE --profile manager up -d --build manager; log "→ $URL"; }
docker_stop()   { log "Stopping compose 'manager' ..."; $COMPOSE stop manager; }
docker_status() { $COMPOSE ps manager; native_status; }

# ---- dispatch --------------------------------------------------------------
CMD="${1:-}"; shift || true
[ "${1:-}" = "--docker" ] && { MODE="docker"; shift || true; }

case "$CMD" in
  start)   [ "$MODE" = docker ] && docker_start  || native_start ;;
  stop)    [ "$MODE" = docker ] && docker_stop   || native_stop ;;
  restart) if [ "$MODE" = docker ]; then docker_stop || true; docker_start;
           else native_stop; native_start; fi ;;
  status)  [ "$MODE" = docker ] && docker_status || native_status ;;
  logs)    if [ "$MODE" = docker ]; then $COMPOSE logs "${1:-}" manager;
           elif [ -f "$LOGFILE" ]; then [ "${1:-}" = "-f" ] && tail -f "$LOGFILE" || tail -n 60 "$LOGFILE";
           else warn "no log at $LOGFILE"; fi ;;
  *) echo "usage: $0 {start|stop|restart|status|logs} [--docker]"; exit 2 ;;
esac
