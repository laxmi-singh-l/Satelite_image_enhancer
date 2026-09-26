#!/usr/bin/env bash
#
# Start the whole dashboard (FastAPI backend + React frontend) with one command.
#
#   ./run.sh              start backend (:8000) and frontend (:5173)
#   ./run.sh backend      backend only
#   ./run.sh frontend     frontend only
#   ./run.sh build        production build of the React app
#   ./run.sh install      install python + node dependencies
#   ./run.sh stop         stop both services
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
LOG_DIR="$ROOT/logs"
RUN_DIR="$ROOT/results/api"
mkdir -p "$LOG_DIR" "$RUN_DIR"

# --- pretty output ----------------------------------------------------------
if [ -t 1 ] && command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"; DIM="$(tput dim)"; RED="$(tput setaf 1)"
  GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"; CYAN="$(tput setaf 6)"
  RESET="$(tput sgr0)"
else
  BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; CYAN=""; RESET=""
fi
step() { echo "${CYAN}==>${RESET} ${BOLD}$*${RESET}"; }
ok()   { echo "    ${GREEN}✓${RESET} $*"; }
warn() { echo "    ${YELLOW}!${RESET} $*"; }
die()  { echo "    ${RED}✗${RESET} $*" >&2; exit 1; }

# --- process helpers --------------------------------------------------------
pidfile() { echo "$RUN_DIR/$1.pid"; }

is_running() {
  local pf; pf="$(pidfile "$1")"
  [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null
}

stop_one() {
  local name="$1" pf; pf="$(pidfile "$name")"
  if [ -f "$pf" ]; then
    local pid; pid="$(cat "$pf")"
    if kill -0 "$pid" 2>/dev/null; then
      # kill the whole process group started by setsid
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      sleep 0.6
      kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null
      ok "stopped $name (pid $pid)"
    fi
    rm -f "$pf"
  fi
}

free_port() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$" && return 1
  elif command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:${port}" >/dev/null 2>&1 && return 0 || return 1
  fi
  return 0
}

wait_for_http() {
  local url="$1" name="$2" tries="${3:-90}"
  for _ in $(seq 1 "$tries"); do
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  return 1
}

# --- setup ------------------------------------------------------------------
ensure_python() {
  if [ ! -x "$PY" ]; then
    warn "no virtualenv at .venv - creating one"
    command -v python3 >/dev/null 2>&1 || die "python3 not found on PATH"
    python3 -m venv "$VENV" || die "failed to create virtualenv"
  fi
  "$PY" - <<'EOF' || die "python dependencies missing - run: ./run.sh install"
import importlib.util, sys
missing = [m for m in ("fastapi", "uvicorn", "multipart", "torch", "cv2", "numpy", "PIL")
           if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
EOF
  [ $? -eq 0 ] || die "python dependencies missing - run: ./run.sh install"
}

ensure_node() {
  command -v npm >/dev/null 2>&1 || die "npm not found - install Node.js 18+"
  if [ ! -d "$ROOT/frontend/node_modules" ]; then
    warn "node_modules missing - installing (first run only)"
    (cd "$ROOT/frontend" && npm install --no-audit --no-fund) \
      || die "npm install failed"
  fi
}

do_install() {
  step "Installing Python dependencies"
  [ -x "$PY" ] || { command -v python3 >/dev/null || die "python3 not found"; python3 -m venv "$VENV"; }
  if "$PY" -m pip --version >/dev/null 2>&1; then
    "$PY" -m pip install -r "$ROOT/requirements.txt"
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" -r "$ROOT/requirements.txt"
  else
    die "need pip or uv to install python packages"
  fi
  ok "python dependencies installed"

  step "Installing Node dependencies"
  command -v npm >/dev/null 2>&1 || die "npm not found - install Node.js 18+"
  (cd "$ROOT/frontend" && npm install --no-audit --no-fund)
  ok "node dependencies installed"
}

# --- services ---------------------------------------------------------------
start_backend() {
  if is_running backend; then warn "backend already running"; return 0; fi
  free_port "$BACKEND_PORT" || die "port $BACKEND_PORT is busy - stop it or set BACKEND_PORT"

  step "Starting API backend on :$BACKEND_PORT"
  cd "$ROOT"
  setsid "$PY" -m uvicorn api.server:app \
    --host 127.0.0.1 --port "$BACKEND_PORT" \
    > "$LOG_DIR/backend.log" 2>&1 < /dev/null &
  echo $! > "$(pidfile backend)"

  if wait_for_http "http://127.0.0.1:$BACKEND_PORT/api/health" backend 120; then
    ok "backend ready  ->  http://127.0.0.1:$BACKEND_PORT  (logs: logs/backend.log)"
  else
    tail -20 "$LOG_DIR/backend.log" >&2
    die "backend failed to start - see logs/backend.log"
  fi
}

start_frontend() {
  if is_running frontend; then warn "frontend already running"; return 0; fi
  free_port "$FRONTEND_PORT" || die "port $FRONTEND_PORT is busy - stop it or set FRONTEND_PORT"

  step "Starting React frontend on :$FRONTEND_PORT"
  cd "$ROOT/frontend"
  setsid npm run dev -- --port "$FRONTEND_PORT" --strictPort \
    > "$LOG_DIR/frontend.log" 2>&1 < /dev/null &
  echo $! > "$(pidfile frontend)"

  if wait_for_http "http://127.0.0.1:$FRONTEND_PORT/" frontend 90; then
    ok "frontend ready ->  http://127.0.0.1:$FRONTEND_PORT  (logs: logs/frontend.log)"
  else
    tail -20 "$LOG_DIR/frontend.log" >&2
    die "frontend failed to start - see logs/frontend.log"
  fi
}

do_build() {
  ensure_node
  step "Building React production bundle"
  (cd "$ROOT/frontend" && npm run build) || die "build failed"
  ok "bundle written to frontend/dist"
}

do_stop() {
  step "Stopping services"
  stop_one frontend
  stop_one backend
  ok "all stopped"
}

# --- entry point ------------------------------------------------------------
case "${1:-start}" in
  install)  do_install ;;
  build)    do_build ;;
  stop)     do_stop ;;
  backend)  ensure_python; start_backend ;;
  frontend) ensure_node;   start_frontend ;;
  start|all|"")
    ensure_python
    ensure_node
    echo
    echo "${BOLD}Satellite IR Enhancement & Analysis${RESET}"
    echo "${DIM}Real-ESRGAN GAN enhancer + IR pipeline${RESET}"
    echo
    start_backend
    start_frontend
    echo
    echo "  ${GREEN}${BOLD}Dashboard ready:${RESET}  ${BOLD}http://127.0.0.1:$FRONTEND_PORT${RESET}"
    echo "  ${DIM}API docs:                        http://127.0.0.1:$BACKEND_PORT/docs${RESET}"
    echo "  ${DIM}Stop everything:                ./run.sh stop${RESET}"
    echo
    ;;
  *) die "unknown command '${1}' (use: start|backend|frontend|build|install|stop)" ;;
esac
