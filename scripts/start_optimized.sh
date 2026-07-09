#!/usr/bin/env bash
# ============================================================================
# start_optimized.sh — glm2api Optimized Launcher & Health Monitor
# ============================================================================
# Kills stale instances, starts the server with our optimized main.py,
# and monitors health on port 8000. Restarts automatically if health
# checks fail 3 consecutive times.
#
# Usage:
#   ./start_optimized.sh              — start server + foreground monitor
#   ./start_optimized.sh start        — same as above
#   ./start_optimized.sh stop         — kill running server
#   ./start_optimized.sh restart      — stop + start
#   ./start_optimized.sh status       — check health & process
#   ./start_optimized.sh once         — start server in background, no monitor
#
# Systemd integration:
#   ExecStart=/home/uluru/ZCodeProject/glm2api/scripts/start_optimized.sh
#   ExecStop=/home/uluru/ZCodeProject/glm2api/scripts/start_optimized.sh stop
#
# Environment variables (all optional):
#   GLM2API_DIR         — project root (default: script's parent directory)
#   GLM2API_PORT        — health-check port (default: 8000)
#   GLM2API_HOST        — health-check host (default: 127.0.0.1)
#   GLM2API_CHECK_INTERVAL — seconds between health checks (default: 15)
#   GLM2API_CONSEC_FAILURES  — failures before restart (default: 3)
#   GLM2API_LOG_DIR    — where to write logs (default: $GLM2API_DIR/log)
# ============================================================================

set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
: "${GLM2API_DIR:="$(cd "$SCRIPT_DIR/.." && pwd)"}"
: "${GLM2API_PORT:=8000}"
: "${GLM2API_HOST:=127.0.0.1}"
: "${GLM2API_CHECK_INTERVAL:=15}"
: "${GLM2API_CONSEC_FAILURES:=3}"
: "${GLM2API_LOG_DIR:="$GLM2API_DIR/log"}"

VENV_PYTHON="$GLM2API_DIR/.venv/bin/python3"
MAIN_PY="$GLM2API_DIR/main.py"
PID_FILE="$SCRIPT_DIR/.glm2api.pid"
HEALTH_URL="http://${GLM2API_HOST}:${GLM2API_PORT}/health"

mkdir -p "$GLM2API_LOG_DIR"

# ── Utilities ──────────────────────────────────────────────────────────────

_log() { echo "[glm2api-wrapper] $(date '+%Y-%m-%d %H:%M:%S') $*"; }
_err() { echo "[glm2api-wrapper] ERROR: $*" >&2; }

# ── Kill existing glm2api processes ────────────────────────────────────────
kill_glm2api() {
    local killed=0

    # 1. Kill by PID file first (most precise)
    if [[ -f "$PID_FILE" ]]; then
        local old_pid
        old_pid="$(cat "$PID_FILE" 2>/dev/null || echo "")"
        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
            _log "Killing PID $old_pid (from PID file)..."
            kill "$old_pid" 2>/dev/null || true
            # Allow graceful shutdown
            for i in 1 2 3; do
                if ! kill -0 "$old_pid" 2>/dev/null; then break; fi
                sleep 0.5
            done
            # Force if still alive
            kill -9 "$old_pid" 2>/dev/null || true
            killed=1
        fi
        rm -f "$PID_FILE"
    fi

    # 2. Kill any other processes from our project dir (catch orphans)
    local pids
    pids="$(pgrep -f "python.*main\.py" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
        # Only kill ones under our project
        for pid in $pids; do
            local exe_path
            exe_path="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || echo "")"
            if [[ "$exe_path" == "$GLM2API_DIR" ]]; then
                _log "Killing orphan PID $pid..."
                kill "$pid" 2>/dev/null || true
                killed=1
            fi
        done
        # Wait a moment for graceful shutdown
        sleep 0.5
        # Force kill any remaining
        for pid in $pids; do
            local exe_path
            exe_path="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || echo "")"
            if [[ "$exe_path" == "$GLM2API_DIR" ]]; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi

    # 3. Also catch via port (any process on our port from our dir)
    local port_pid
    port_pid="$(ss -tlnp 2>/dev/null | grep ":${GLM2API_PORT} " | grep -oP 'pid=\K[0-9]+' || true)"
    if [[ -n "$port_pid" ]]; then
        local cwd
        cwd="$(readlink -f "/proc/$port_pid/cwd" 2>/dev/null || echo "")"
        if [[ "$cwd" == "$GLM2API_DIR" ]]; then
            _log "Killing port-bound PID $port_pid..."
            kill "$port_pid" 2>/dev/null || true
            sleep 0.5
            kill -9 "$port_pid" 2>/dev/null || true
            killed=1
        fi
    fi

    if [[ "$killed" -eq 1 ]]; then
        _log "Waiting for port $GLM2API_PORT to free up..."
        for i in $(seq 1 10); do
            if ! ss -tlnp 2>/dev/null | grep -q ":${GLM2API_PORT} "; then
                break
            fi
            sleep 0.3
        done
    fi
}

# ── Start the server ───────────────────────────────────────────────────────
start_server() {
    # Make sure nothing else is on our port
    kill_glm2api

    # Verify environment
    if [[ ! -x "$VENV_PYTHON" ]]; then
        _err "Virtual environment python not found at $VENV_PYTHON"
        return 1
    fi
    if [[ ! -f "$MAIN_PY" ]]; then
        _err "main.py not found at $MAIN_PY"
        return 1
    fi

    # Random sleep (0–3s) to avoid thundering herd if multiple restart-trigger events fire
    local jitter=$((RANDOM % 4))
    sleep "$jitter"

    # Start the server with nohup in background
    cd "$GLM2API_DIR"
    local log_file="$GLM2API_LOG_DIR/server.log"
    _log "Starting server: $VENV_PYTHON -B main.py"
    nohup "$VENV_PYTHON" -B main.py >> "$log_file" 2>&1 &
    local server_pid=$!
    echo "$server_pid" > "$PID_FILE"
    _log "Server started with PID $server_pid"

    # Wait for it to listen on the port (up to 30s)
    local waited=0
    while [[ $waited -lt 30 ]]; do
        if ss -tlnp 2>/dev/null | grep -q ":${GLM2API_PORT} "; then
            _log "Server is listening on port $GLM2API_PORT (after ${waited}s)"
            return 0
        fi
        # Check if process is still alive
        if ! kill -0 "$server_pid" 2>/dev/null; then
            _err "Server process died during startup! Check logs: $log_file"
            tail -20 "$log_file" >&2
            return 1
        fi
        sleep 1
        ((waited++))
    done

    _err "Server did not start listening on port $GLM2API_PORT within 30s"
    return 1
}

# ── Health check ───────────────────────────────────────────────────────────
check_health() {
    # Use curl with a short timeout
    local resp
    resp="$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$HEALTH_URL" 2>/dev/null || true)"
    if [[ "$resp" == "200" ]]; then
        return 0
    fi
    return 1
}

# ── Stop ───────────────────────────────────────────────────────────────────
cmd_stop() {
    _log "Stopping glm2api server..."
    kill_glm2api
    # Double-check port is free
    if ss -tlnp 2>/dev/null | grep -q ":${GLM2API_PORT} "; then
        local stubborn_pid
        stubborn_pid="$(ss -tlnp 2>/dev/null | grep ":${GLM2API_PORT} " | grep -oP 'pid=\K[0-9]+' || true)"
        if [[ -n "$stubborn_pid" ]]; then
            _log "Force-killing PID $stubborn_pid still on port $GLM2API_PORT..."
            kill -9 "$stubborn_pid" 2>/dev/null || true
        fi
    fi
    _log "Stopped."
}

# ── Status ─────────────────────────────────────────────────────────────────
cmd_status() {
    local server_up=false
    local health_ok=false

    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(cat "$PID_FILE" 2>/dev/null || echo "")"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            server_up=true
            echo "Process: PID $pid is running"
        else
            echo "Process: PID file exists but process $pid is not running"
        fi
    else
        # Check by port
        local port_pid
        port_pid="$(ss -tlnp 2>/dev/null | grep ":${GLM2API_PORT} " | grep -oP 'pid=\K[0-9]+' || true)"
        if [[ -n "$port_pid" ]]; then
            server_up=true
            echo "Process: PID $port_pid listening on port $GLM2API_PORT"
        else
            echo "Process: Not running"
        fi
    fi

    if check_health; then
        health_ok=true
        echo "Health: OK (200 from $HEALTH_URL)"
    else
        echo "Health: FAILED"
    fi

    if [[ "$server_up" == true ]] && [[ "$health_ok" == true ]]; then
        return 0
    elif [[ "$server_up" == true ]] && [[ "$health_ok" == false ]]; then
        return 1
    else
        return 2
    fi
}

# ── Foreground monitor loop ────────────────────────────────────────────────
# Stays in foreground (so systemd can track it) and restarts the server
# if health checks fail consecutively.
monitor_loop() {
    local fail_count=0
    local restart_count=0
    local max_fail="$GLM2API_CONSEC_FAILURES"

    _log "Starting health monitor (interval=${GLM2API_CHECK_INTERVAL}s, max_failures=${max_fail})"

    # Trap SIGTERM/SIGINT to shut down cleanly
    trap_handler() {
        _log "Received shutdown signal, stopping server..."
        kill_glm2api
        exit 0
    }
    trap trap_handler SIGTERM SIGINT SIGQUIT SIGHUP

    # First, wait a few seconds for the server to settle after startup
    sleep 3

    while true; do
        if check_health; then
            if [[ $fail_count -gt 0 ]]; then
                _log "Health check passed (recovered after $fail_count failures)"
                fail_count=0
            fi
        else
            ((fail_count++))
            _log "Health check FAILED ($fail_count/$max_fail)"

            if [[ $fail_count -ge $max_fail ]]; then
                _log "Restarting server after $fail_count consecutive failures..."
                kill_glm2api
                sleep 1
                if start_server; then
                    fail_count=0
                    ((restart_count++))
                    _log "Server restarted successfully (total restarts: $restart_count)"
                else
                    _err "Server restart FAILED — will retry in ${GLM2API_CHECK_INTERVAL}s"
                    fail_count=1  # restart counter so we don't keep resetting
                fi
            fi
        fi

        sleep "$GLM2API_CHECK_INTERVAL"
    done
}

# ── Command dispatch ───────────────────────────────────────────────────────
case "${1:-start}" in
    start|--start)
        # Kill old, start server, then enter monitor loop
        if start_server; then
            monitor_loop
        else
            _err "Failed to start server"
            exit 1
        fi
        ;;
    once|--once)
        # Just start the server in background, no monitor
        if start_server; then
            _log "Server started in background (no monitor). PID: $(cat "$PID_FILE" 2>/dev/null || echo 'unknown')"
        else
            _err "Failed to start server"
            exit 1
        fi
        ;;
    stop|--stop)
        cmd_stop
        ;;
    restart|--restart)
        cmd_stop
        sleep 1
        if start_server; then
            _log "Server restarted successfully"
            # Show status info
            cmd_status || true
        else
            _err "Restart failed"
            exit 1
        fi
        ;;
    status|--status)
        cmd_status
        exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            echo "Status: RUNNING OK"
        elif [[ $exit_code -eq 1 ]]; then
            echo "Status: RUNNING (but health failing)"
        else
            echo "Status: NOT RUNNING"
        fi
        exit $exit_code
        ;;
    help|--help)
        echo "Usage: $(basename "$0") [command]"
        echo ""
        echo "Commands:"
        echo "  start     Kill old process, start server, and monitor health (default)"
        echo "  once      Start server in background only (no health monitor)"
        echo "  stop      Kill running server"
        echo "  restart   Stop then start"
        echo "  status    Check server health and process status"
        echo "  help      Show this help"
        echo ""
        echo "Environment:"
        echo "  GLM2API_DIR               Project root (default: auto-detected)"
        echo "  GLM2API_PORT              Health check port (default: 8000)"
        echo "  GLM2API_CHECK_INTERVAL    Seconds between checks (default: 15)"
        echo "  GLM2API_CONSEC_FAILURES   Failures before restart (default: 3)"
        echo "  GLM2API_LOG_DIR           Log directory (default: \$GLM2API_DIR/log)"
        exit 0
        ;;
    *)
        _err "Unknown command: $1"
        echo "Usage: $(basename "$0") [start|stop|restart|status|once|help]"
        exit 1
        ;;
esac
