#!/usr/bin/env python3
"""Create multiple guest refresh tokens through DIFFERENT proxy IPs.

Reads working proxies from config/verified_proxies.json (produced by verify_proxies.py).
For each proxy, creates ONE guest token, ensuring each token comes from a unique IP.
Saves the refresh_tokens to token.txt for use by glm2api.

Target: 24 tokens × 5 slots = 120 concurrent conversations.

Usage:
  python3 scripts/create_guest_tokens.py [--count 24] [--delay 2] [--proxies socks5://...]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from curl_cffi import requests

SIGN_SECRET = "8a1317a7468aa3ad86e997d08f3f31cb"
TARGET_COUNT = 24
DELAY_BETWEEN = 2.0  # seconds between tokens to avoid rate limits
TIMEOUT = 20


def build_sign() -> tuple[str, str, str]:
    """Build the exact sign hash matching chatglm.cn JS."""
    ts = str(int(time.time() * 1000))
    digits = [int(c) for c in ts]
    checksum = (sum(digits) - digits[-2]) % 10
    timestamp = ts[:-2] + str(checksum) + ts[-1]
    nonce = uuid.uuid4().hex
    sign = hashlib.md5(f"{timestamp}-{nonce}-{SIGN_SECRET}".encode()).hexdigest()
    return timestamp, nonce, sign


def create_token_through_proxy(proxy_url: str, token_index: int) -> str | None:
    """Create ONE guest refresh token through the given proxy.
    Returns the refresh_token string on success, None on failure."""
    try:
        timestamp, nonce, sign = build_sign()
        device_id = uuid.uuid4().hex
        headers = {
            "Content-Type": "application/json;charset=utf-8",
            "Content-Length": "0",
            "App-Name": "chatglm",
            "Origin": "https://chatglm.cn",
            "Referer": "https://chatglm.cn/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0",
            "X-App-Fr": "browser_extension",
            "X-App-Platform": "pc",
            "X-App-Version": "1.0.83",
            "X-Device-Id": device_id,
            "X-Lang": "en",
            "X-Nonce": nonce,
            "X-Request-Id": uuid.uuid4().hex,
            "X-Sign": sign,
            "X-Timestamp": timestamp,
        }
        proxies = {"https": proxy_url, "http": proxy_url}
        resp = requests.post(
            "https://chatglm.cn/chatglm/user-api/guest/access",
            impersonate="chrome120",
            proxies=proxies,
            headers=headers,
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == 0:
                result = data.get("result", {})
                access_token = result.get("access_token")
                refresh_token = result.get("refresh_token")
                if access_token and refresh_token:
                    return str(refresh_token)
                else:
                    print(f"  [{token_index}] Response missing tokens: {list(result.keys())}", file=sys.stderr)
            else:
                print(f"  [{token_index}] API returned status={data.get('status')}: {data.get('message', '')}", file=sys.stderr)
        else:
            print(f"  [{token_index}] HTTP {resp.status_code} from proxy {proxy_url[:50]}", file=sys.stderr)
    except Exception as exc:
        print(f"  [{token_index}] Exception: {type(exc).__name__}: {exc}", file=sys.stderr)
    return None


def load_working_proxies(config_dir: str) -> list[str]:
    """Load verified proxies from config/verified_proxies.json.
    Returns proxy URLs that passed guest_token_test, falling back to connectivity_test."""
    proxies_path = os.path.join(config_dir, "verified_proxies.json")
    if not os.path.exists(proxies_path):
        print(f"File not found: {proxies_path}", file=sys.stderr)
        print("Run scripts/verify_proxies.py first.", file=sys.stderr)
        return []

    with open(proxies_path) as f:
        data = json.load(f)

    # Prefer proxies that passed guest token test
    if data.get("guest_token_test"):
        proxies = [entry["url"] for entry in data["guest_token_test"]]
        print(f"Loaded {len(proxies)} guest-token-verified proxies", file=sys.stderr)
        return proxies

    # Fall back to connectivity-tested proxies
    if data.get("connectivity_test"):
        proxies = [entry["url"] for entry in data["connectivity_test"]]
        print(f"Loaded {len(proxies)} connectivity-tested proxies (no token test results)", file=sys.stderr)
        return proxies

    print("No proxies found in verified_proxies.json", file=sys.stderr)
    return []


def main():
    # Parse args
    count = TARGET_COUNT
    delay = DELAY_BETWEEN
    proxy_list: list[str] = []
    specific_proxies: list[str] = []

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--count" and i + 1 < len(sys.argv):
            count = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--delay" and i + 1 < len(sys.argv):
            delay = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--proxies" and i + 1 < len(sys.argv):
            specific_proxies = sys.argv[i + 1].split(",")
            i += 2
        else:
            i += 1

    config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')

    if specific_proxies:
        proxy_list = specific_proxies
        print(f"Using {len(proxy_list)} user-specified proxies", file=sys.stderr)
    else:
        proxy_list = load_working_proxies(config_dir)

    if not proxy_list:
        print("No proxies available. Exiting.", file=sys.stderr)
        sys.exit(1)

    if len(proxy_list) < count:
        print(f"Warning: only {len(proxy_list)} proxies available for {count} tokens (will use all)", file=sys.stderr)
        count = len(proxy_list)

    print(f"\nCreating {count} guest tokens through unique proxies...", file=sys.stderr)
    print(f"Delay between tokens: {delay}s", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    tokens: list[str] = []
    failed_count = 0
    max_attempts = 3  # max retries per proxy

    for idx in range(count):
        if idx >= len(proxy_list):
            print(f"\nRan out of unique proxies after {idx} tokens.", file=sys.stderr)
            break

        proxy = proxy_list[idx]
        print(f"\n[{idx+1}/{count}] Creating token through {proxy[:60]}...", file=sys.stderr)

        token = None
        for attempt in range(max_attempts):
            if attempt > 0:
                print(f"  Retry {attempt+1}/{max_attempts}...", file=sys.stderr)
                time.sleep(delay)  # extra delay before retry

            token = create_token_through_proxy(proxy, idx + 1)
            if token:
                break

        if token:
            tokens.append(token)
            print(f"  ✓ Token created successfully", file=sys.stderr)
        else:
            failed_count += 1
            print(f"  ✗ Failed after {max_attempts} attempts", file=sys.stderr)

        # Respectful delay between tokens to avoid rate limiting
        if idx < count - 1:
            time.sleep(delay)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Results: {len(tokens)} tokens created, {failed_count} failed", file=sys.stderr)

    if not tokens:
        print("No tokens created. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Save to token.txt
    token_path = os.path.join(os.path.dirname(__file__), '..', 'token.txt')
    with open(token_path, "w") as f:
        for t in tokens:
            f.write(t + "\n")
    print(f"Saved {len(tokens)} tokens to {token_path}", file=sys.stderr)

    # Also save a .env-compatible batch
    batch_path = os.path.join(config_dir, "guest_tokens.txt")
    with open(batch_path, "w") as f:
        f.write("# Guest refresh tokens - one per line, one per unique IP\n")
        f.write(f"# Created: {time.ctime()}\n")
        f.write(f"# Count: {len(tokens)}\n")
        for t in tokens:
            f.write(t + "\n")
    print(f"Backup saved to {batch_path}", file=sys.stderr)

    # Print summary
    print(f"\nToken creation complete! {len(tokens)} tokens ready for use.", file=sys.stderr)
    print(f"Set GLM_MAX_CONCURRENCY={len(tokens)} in .env for {len(tokens)*5} max concurrent slots.", file=sys.stderr)

    # Return first token preview
    if tokens:
        print(f"\nFirst token preview: {tokens[0][:50]}...", file=sys.stderr)


if __name__ == "__main__":
    main()
