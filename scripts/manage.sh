#!/usr/bin/env bash
# glm2api management script for Linux
set -euo pipefail

GLM2API_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_NAME="glm2api"
SERVICE_FILE="${GLM2API_DIR}/${SERVICE_NAME}.service"

cmd="${1:-help}"

case "$cmd" in
    start)
        cd "$GLM2API_DIR"
        echo "Starting glm2api..."
        exec "${GLM2API_DIR}/.venv/bin/python" -m glm2api
        ;;
    install-service)
        echo "Installing systemd service..."
        if [ -f "$SERVICE_FILE" ]; then
            sudo cp "$SERVICE_FILE" "/etc/systemd/system/${SERVICE_NAME}.service"
            sudo systemctl daemon-reload
            sudo systemctl enable "${SERVICE_NAME}"
            sudo systemctl start "${SERVICE_NAME}"
            echo "Service installed and started."
            sudo systemctl status "${SERVICE_NAME}" --no-pager
        else
            echo "Error: $SERVICE_FILE not found"
            exit 1
        fi
        ;;
    remove-service)
        echo "Removing systemd service..."
        sudo systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
        sudo systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
        sudo rm -f "/etc/systemd/system/${SERVICE_NAME}.service"
        sudo systemctl daemon-reload
        echo "Service removed."
        ;;
    status)
        if systemctl is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
            sudo systemctl status "${SERVICE_NAME}" --no-pager
        else
            echo "Service not running. Checking if installed..."
            if [ -f "/etc/systemd/system/${SERVICE_NAME}.service" ]; then
                echo "Service is installed but not active."
                sudo systemctl status "${SERVICE_NAME}" --no-pager 2>&1 | head -10
            else
                echo "Service not installed."
            fi
        fi
        ;;
    logs)
        sudo journalctl -u "${SERVICE_NAME}" -n 50 -f
        ;;
    test)
        echo "Testing glm2api health..."
        curl -s http://127.0.0.1:8000/health 2>/dev/null || echo "Not running"
        echo ""
        curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Models endpoint not responding"
        ;;
    help|*)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  start            Start proxy (foreground)"
        echo "  install-service  Install and enable systemd service"
        echo "  remove-service   Remove systemd service"
        echo "  status           Check service status"
        echo "  logs             Follow service logs"
        echo "  test             Test proxy health/model endpoints"
        echo "  help             Show this help"
        ;;
esac
