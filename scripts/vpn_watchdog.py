#!/usr/bin/env python3
"""VPN Watchdog — monitors the aimiligate VPN proxy and auto-reconnects on failure.

Periodically checks if the HTTP/SOCKS5 proxy at 127.0.0.1:7928 can reach a target
URL. On consecutive failures, it first tries a gentle reconnect via the VPN
manager's Web API. If that fails, it kills and restarts the entire vpngate_manager
process.

Run as:
    nohup python3 /home/uluru/ZCodeProject/glm2api/scripts/vpn_watchdog.py \
        > /tmp/vpn_watchdog.log 2>&1 &
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 7928
PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"
TARGET_URL = "https://chatglm.cn/"
CHECK_INTERVAL = 30  # seconds between health checks
MAX_FAILURES = 3     # consecutive failures before triggering reconnect

# VPN manager Web API (for gentle reconnect via /api/check)
VPNGATE_MANAGER_PATH = "/home/uluru/aimili-vpngate/vpngate_manager.py"
VPNGATE_DATA_DIR = "/home/uluru/aimili-vpngate/vpngate_data"
UI_AUTH_FILE = os.path.join(VPNGATE_DATA_DIR, "ui_auth.json")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("vpn_watchdog")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def check_proxy() -> bool:
    """Returns True if the proxy can successfully reach the target URL.

    Attempts both HTTP CONNECT (through the proxy for HTTPS URLs) and a
    direct SOCKS5-style check to cover both proxy modes that the VPN manager
    supports.  Returns False on any failure (timeout, non-200, connection
    refused, etc.).
    """
    # Method 1: urllib with ProxyHandler (HTTP proxy)
    try:
        proxy_handler = urllib.request.ProxyHandler({
            "http": PROXY_URL,
            "https": PROXY_URL,
        })
        opener = urllib.request.build_opener(proxy_handler)
        resp = opener.open(TARGET_URL, timeout=10)
        if resp.status == 200:
            return True
        log.debug("Proxy returned status %d", resp.status)
        return False
    except (urllib.error.URLError, urllib.error.HTTPError,
            socket.timeout, OSError) as exc:
        log.debug("Proxy check via HTTP failed: %s", exc)

    # Method 2: raw socket SOCKS5 handshake (in case only SOCKS5 is available)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(8)
        sock.connect((PROXY_HOST, PROXY_PORT))
        # SOCKS5 greeting
        sock.sendall(b"\x05\x01\x00")
        if sock.recv(2) != b"\x05\x00":
            sock.close()
            return False
        # Connect to target
        hostname = b"chatglm.cn"
        port = 443
        sock.sendall(b"\x05\x01\x00\x03" + bytes([len(hostname)]) + hostname + port.to_bytes(2, "big"))
        resp = sock.recv(10)
        sock.close()
        if len(resp) >= 2 and resp[1] == 0x00:
            return True
        return False
    except (OSError, socket.timeout) as exc:
        log.debug("Proxy check via SOCKS5 failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Reconnection strategies
# ---------------------------------------------------------------------------
def _read_ui_auth() -> dict[str, str] | None:
    """Read VPN manager Web API credentials from ui_auth.json."""
    try:
        if os.path.exists(UI_AUTH_FILE):
            with open(UI_AUTH_FILE) as f:
                return json.load(f)
    except Exception as exc:
        log.warning("Failed to read ui_auth.json: %s", exc)
    return None


def _api_reconnect() -> bool:
    """Attempt a gentle reconnect via the VPN manager's Web API.

    Reads credentials from ui_auth.json, logs in to get a session cookie,
    then calls /api/check to force a node refresh + reconnect.
    Returns True on success, False on any failure.
    """
    auth = _read_ui_auth()
    if not auth:
        log.warning("No Web API credentials available, skipping API reconnect")
        return False

    host = auth.get("host", "127.0.0.1")
    port = auth.get("port", 18787)
    secret_path = auth.get("secret_path", "EJsW2EeBo9lY")
    username = auth.get("username", "")
    password = auth.get("password", "")

    if not username or not password:
        log.warning("Web API credentials incomplete, skipping API reconnect")
        return False

    base_url = f"http://{host}:{port}/{secret_path}"

    try:
        # Step 1: Login to get session cookie
        login_data = json.dumps({"username": username, "password": password}).encode()
        req = urllib.request.Request(
            f"{base_url}/api/login",
            data=login_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=10)
        if resp.status != 200:
            log.warning("API login failed: HTTP %d", resp.status)
            return False

        # Extract session cookie from response
        set_cookie = resp.headers.get("Set-Cookie", "")
        session_token = ""
        for part in set_cookie.split(";"):
            part = part.strip()
            if part.startswith("session="):
                session_token = part[len("session="):]
                break

        if not session_token:
            log.warning("No session cookie received from API login")
            return False

        # Step 2: Call /api/check to force reconnect
        check_req = urllib.request.Request(
            f"{base_url}/api/check",
            headers={"Cookie": f"session={session_token}"},
            method="POST",
        )
        check_resp = urllib.request.urlopen(check_req, timeout=30)
        result = json.loads(check_resp.read().decode())
        if result.get("ok"):
            log.info("API reconnect succeeded: %s", result.get("message", ""))
            return True
        else:
            log.warning("API reconnect failed: %s", result.get("error", "unknown"))
            return False

    except Exception as exc:
        log.warning("API reconnect request failed: %s", exc)
        return False


def _kill_vpn_manager(force: bool = False) -> None:
    """Kill the vpngate_manager.py process."""
    signal = "-9" if force else "-TERM"
    try:
        subprocess.run(
            ["pkill", signal, "-f", "vpngate_manager.py"],
            timeout=5,
            capture_output=True,
        )
        log.info("Sent %s to vpngate_manager.py", "SIGKILL" if force else "SIGTERM")
    except subprocess.TimeoutExpired:
        log.warning("pkill timed out")
    except Exception as exc:
        log.warning("Failed to kill vpngate_manager: %s", exc)


def _restart_vpn_manager() -> bool:
    """Restart vpngate_manager.py as a background process."""
    try:
        subprocess.Popen(
            ["python3", VPNGATE_MANAGER_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        log.info("VPN manager process started")
        return True
    except Exception as exc:
        log.warning("Failed to start vpngate_manager: %s", exc)
        return False


def reconnect_vpn() -> bool:
    """Force VPN reconnect.

    Strategy:
      1. Try gentle API-based reconnect first (/api/check).
      2. If that fails, kill the process with SIGTERM, wait, then restart.
      3. If still running after SIGTERM, escalate to SIGKILL.
    """
    log.info("Attempting VPN reconnect...")

    # Strategy 1: Gentle API reconnect
    if _api_reconnect():
        # Give it time to establish the new tunnel
        time.sleep(15)
        return True

    log.info("API reconnect failed, falling back to process restart")

    # Strategy 2: Kill and restart
    # First try graceful SIGTERM
    _kill_vpn_manager(force=False)
    time.sleep(3)

    # Check if it's still running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vpngate_manager.py"],
            capture_output=True, timeout=3,
        )
        if result.returncode == 0:
            log.warning("VPN manager still running after SIGTERM, sending SIGKILL")
            _kill_vpn_manager(force=True)
            time.sleep(2)
    except Exception:
        pass

    # Also kill any lingering OpenVPN processes on tun0
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "openvpn.*tun0"],
            timeout=3, capture_output=True,
        )
    except Exception:
        pass

    # Start fresh
    if _restart_vpn_manager():
        log.info("Waiting 15s for VPN tunnel to establish...")
        time.sleep(15)
        return True

    return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 60)
    log.info("VPN Watchdog started")
    log.info("Proxy: %s", PROXY_URL)
    log.info("Target: %s", TARGET_URL)
    log.info("Check interval: %ds, Max failures: %d", CHECK_INTERVAL, MAX_FAILURES)
    log.info("=" * 60)

    failures = 0
    consecutive = 0  # track consecutive restarts to detect rapid failure loops

    while True:
        time.sleep(CHECK_INTERVAL)

        alive = check_proxy()
        if alive:
            if failures > 0:
                log.info("Proxy is back online (was %d consecutive failures)", failures)
            failures = 0
            continue

        failures += 1
        consecutive += 1
        log.warning(
            "Proxy check failed (%d/%d consecutive, %d total)",
            failures, MAX_FAILURES, consecutive,
        )

        if failures >= MAX_FAILURES:
            log.error("Proxy unreachable for %d checks — triggering reconnect", MAX_FAILURES)
            ok = reconnect_vpn()
            if ok:
                log.info("Reconnect sequence completed, resuming monitoring")
            else:
                log.error("Reconnect FAILED — will retry on next cycle")
            failures = 0  # reset counter regardless; next check will re-trigger if still down

            # If we've restarted many times in a short window, back off
            if consecutive >= 6:
                log.warning("Multiple rapid restarts detected (%d), backing off 120s", consecutive)
                time.sleep(120)
                consecutive = 0
        else:
            # Brief per-failure backoff: if we're at failure 2/3, wait a bit
            # longer before the next check to give the proxy a chance to recover
            # without accumulating failures too fast.
            if failures == MAX_FAILURES - 1:
                log.debug("Waiting extra 10s before final failure check")
                time.sleep(10)


if __name__ == "__main__":
    main()
