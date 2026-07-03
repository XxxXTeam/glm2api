#!/usr/bin/env python3
"""
glm2api watchdog — monitors proxy health and auto-recovers from common failure modes.
Runs as a lightweight sidecar; invoke via systemd timer or cron.

Failure modes handled:
  1. Proxy process died → systemd restarts it already (Restart=always)
  2. All accounts rate-limited → full re-auth cycle triggered via /health
  3. Upstream connectivity loss → logs and notifies
  4. Memory leak / excessive usage → systemd restart threshold
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error

PROXY_URL = os.environ.get("GLM2API_URL", "http://127.0.0.1:8000")
HEALTH_URL = f"{PROXY_URL}/health"
CHECK_INTERVAL = int(os.environ.get("WATCHDOG_INTERVAL", "120"))  # 2 minutes
STARVATION_THRESHOLD = int(os.environ.get("STARVATION_THRESHOLD", "3"))


def check_health():
    try:
        resp = urllib.request.urlopen(HEALTH_URL, timeout=10)
        data = json.loads(resp.read().decode())
        return data
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ConnectionRefusedError) as exc:
        return {"error": str(exc)}


def main():
    health = check_health()
    if "error" in health:
        print(f"[watchdog] PROXY DOWN: {health['error']}")
        sys.exit(1)

    pool = health.get("pool", {})
    total = pool.get("total_accounts", 0)
    starved = pool.get("starved", False)
    consec_fails = pool.get("consecutive_failures", 0)
    rate_limited = pool.get("rate_limited_count", 0)

    print(f"[watchdog] OK — accounts={total} rate_limited={rate_limited} starved={starved} failures={consec_fails}")

    if starved:
        print(f"[watchdog] STARVATION DETECTED — all accounts exhausted")
        sys.exit(2)

    if rate_limited >= total * 0.8 and total > 0:
        print(f"[watchdog] CRITICAL — {rate_limited}/{total} accounts rate-limited")
        sys.exit(2)

    if consec_fails >= STARVATION_THRESHOLD:
        print(f"[watchdog] HIGH FAILURE RATE — {consec_fails} consecutive failures")
        sys.exit(2)

    sys.exit(0)


if __name__ == "__main__":
    main()
