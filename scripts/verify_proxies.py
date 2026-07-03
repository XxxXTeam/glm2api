#!/usr/bin/env python3
"""Verify SOCKS5/HTTP proxies against chatglm.cn using curl_cffi.
This matches how the proxy will actually be used (curl_cffi Chrome 120).

Two-stage verification:
  Stage 1: curl_cffi GET to https://chatglm.cn/ — basic connectivity + WAF check
  Stage 2: curl_cffi POST to /chatglm/user-api/guest/access — full guest token flow

Outputs verified_proxies.json with working proxy URLs and latency.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from curl_cffi import requests

TARGET = "https://chatglm.cn/"
TIMEOUT = 10
SIGN_SECRET = "8a1317a7468aa3ad86e997d08f3f31cb"

PUBLIC_SOCKS5_URLS = (
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=all",
    "https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt",
    "https://api.openproxylist.xyz/socks5.txt",
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/socks5.txt",
    "https://raw.githubusercontent.com/roosterkid/openproxylist/main/SOCKS5_RAW.txt",
    "https://raw.githubusercontent.com/mmpx12/proxy-list/master/socks5.txt",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=CN",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=HK",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=JP",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=SG",
)


def build_sign() -> tuple[str, str, str]:
    """Build the exact sign hash matching chatglm.cn JS."""
    ts = str(int(time.time() * 1000))
    digits = [int(c) for c in ts]
    checksum = (sum(digits) - digits[-2]) % 10
    timestamp = ts[:-2] + str(checksum) + ts[-1]
    nonce = uuid.uuid4().hex
    sign = hashlib.md5(f"{timestamp}-{nonce}-{SIGN_SECRET}".encode()).hexdigest()
    return timestamp, nonce, sign


def verify_proxy(proxy_url: str) -> tuple[str, float | None]:
    """Stage 1: curl_cffi GET to chatglm.cn through proxy.
    Returns (proxy_url, latency_ms) on success, (proxy_url, None) on failure."""
    try:
        proxies = {"https": proxy_url, "http": proxy_url}
        t0 = time.perf_counter()
        resp = requests.get(
            TARGET,
            impersonate="chrome120",
            proxies=proxies,
            timeout=TIMEOUT,
        )
        if resp.status_code == 200:
            return (proxy_url, (time.perf_counter() - t0) * 1000)
    except Exception:
        pass
    return (proxy_url, None)


def verify_guest_token(proxy_url: str) -> tuple[str, bool]:
    """Stage 2: curl_cffi POST to guest token endpoint through proxy.
    This is the definitive test — can this proxy create a guest token?
    Returns (proxy_url, True/False)."""
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
            timeout=20,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == 0 and data.get("result", {}).get("access_token"):
                return (proxy_url, True)
    except Exception:
        pass
    return (proxy_url, False)


def fetch_public_proxies() -> list[str]:
    """Fetch SOCKS5 proxy URLs from public sources."""
    import httpx
    seen: set[str] = set()
    proxies: list[str] = []
    for url in PUBLIC_SOCKS5_URLS:
        try:
            resp = httpx.get(url, timeout=10)
            for line in resp.text.strip().split("\n"):
                addr = line.strip()
                if addr and ":" in addr and not addr.startswith("0.0.0.0") and addr not in seen:
                    seen.add(addr)
                    proxies.append(f"socks5://{addr}")
        except Exception:
            pass
    return proxies


def socks5_handshake(proxy_url: str, timeout: float = 2) -> bool:
    """Quick SOCKS5 handshake pre-filter. Just checks if proxy responds.
    Much faster than curl_cffi GET — used to eliminate dead proxies before
    the expensive curl_cffi verification."""
    import socket as _sck
    raw = proxy_url.replace("socks5://", "").replace("http://", "").replace("https://", "")
    host, port_str = raw.split(":") if ":" in raw else (raw, "1080")
    try:
        s = _sck.create_connection((host, int(port_str)), timeout=timeout)
        s.sendall(b"\x05\x01\x00")
        if s.recv(2) != b"\x05\x00":
            s.close()
            return False
        hostname = b"chatglm.cn"
        s.sendall(b"\x05\x01\x00\x03" + bytes([len(hostname)]) + hostname + (443).to_bytes(2, "big"))
        resp = s.recv(10)
        s.close()
        return len(resp) >= 2 and resp[1] == 0x00
    except Exception:
        return False


def main():
    proxies: list[str] = []

    # Read proxies from args or stdin
    if len(sys.argv) > 1 and sys.argv[1] not in ("--skip-fetch",):
        file_or_list = sys.argv[1]
        if os.path.isfile(file_or_list):
            # File: one proxy per line or comma-separated
            text = Path(file_or_list).read_text().strip()
            if "," in text:
                proxies = [f"socks5://{p.strip()}" if not p.strip().startswith(("socks5://", "http://", "https://")) else p.strip() for p in text.split(",") if p.strip()]
            else:
                for line in text.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if not line.startswith(("socks5://", "http://", "https://")):
                            line = f"socks5://{line}"
                        proxies.append(line)
        elif sys.argv[1] == "--fetch":
            # Fetch from public sources
            print("Fetching public SOCKS5 proxies...", file=sys.stderr)
            proxies = fetch_public_proxies()
        else:
            # Literal proxy URLs
            proxies = sys.argv[1:]
    elif not sys.stdin.isatty():
        for line in sys.stdin:
            line = line.strip()
            if line and not line.startswith("#"):
                if not line.startswith(("socks5://", "http://", "https://")):
                    line = f"socks5://{line}"
                proxies.append(line)
    else:
        # Default: fetch from public sources
        print("Fetching public SOCKS5 proxies...", file=sys.stderr)
        proxies = fetch_public_proxies()

    if not proxies:
        print("No proxies to verify. Provide proxy URLs as arguments, via stdin, or use --fetch.", file=sys.stderr)
        sys.exit(1)

    print(f"Total proxies: {len(proxies)}", file=sys.stderr)

    # Pre-filter: quick SOCKS5 handshake to eliminate dead proxies
    # This is NOT the verification — it's a fast pre-filter so we don't
    # waste curl_cffi time on clearly dead proxies.
    if len(proxies) > 100:
        print(f"\n{'='*60}", file=sys.stderr)
        print("Pre-filter: quick SOCKS5 handshake (eliminate dead proxies)...", file=sys.stderr)
        alive: list[str] = []
        with ThreadPoolExecutor(max_workers=100) as pool:
            fut_map = {pool.submit(socks5_handshake, p): p for p in proxies[:2000]}
            for f in as_completed(fut_map):
                p = fut_map[f]
                if f.result():
                    alive.append(p)
                    if len(alive) >= 300:
                        break
        print(f"Pre-filter: {len(alive)} proxies pass SOCKS5 handshake", file=sys.stderr)
        proxies = alive

    if not proxies:
        print("No proxies pass SOCKS5 handshake.", file=sys.stderr)
        sys.exit(1)

    print(f"Verifying {len(proxies)} proxies with curl_cffi Chrome 120...", file=sys.stderr)

    # Stage 1: Quick connectivity test (GET chatglm.cn)
    print(f"\n{'='*60}", file=sys.stderr)
    print("Stage 1: curl_cffi GET to chatglm.cn (connectivity + WAF check)", file=sys.stderr)
    working: list[tuple[str, float]] = []
    with ThreadPoolExecutor(max_workers=50) as pool:
        fut_map = {pool.submit(verify_proxy, p): p for p in proxies[:500]}
        for f in as_completed(fut_map):
            p, lat = f.result()
            if lat is not None:
                working.append((p, lat))
                if len(working) >= 100:
                    break

    print(f"Stage 1 result: {len(working)}/{min(len(proxies), 2000)} proxies reachable via curl_cffi", file=sys.stderr)

    if not working:
        print("No working proxies found in Stage 1.", file=sys.stderr)
        result = {"connectivity_test": [], "guest_token_test": [], "summary": {"stage1": 0, "stage2": 0}}
        out_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'verified_proxies.json')
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved empty result to {out_path}", file=sys.stderr)
        return

    # Stage 2: Test guest token creation through best proxies
    print(f"\n{'='*60}", file=sys.stderr)
    print("Stage 2: curl_cffi POST to guest/access (full token flow)", file=sys.stderr)
    # Sort by latency and take top 50 for token testing
    working.sort(key=lambda x: x[1])
    token_working: list[tuple[str, float]] = []
    token_candidates = [p for p, _ in working[:50]]

    with ThreadPoolExecutor(max_workers=10) as pool:
        fut_map = {pool.submit(verify_guest_token, p): p for p in token_candidates}
        for f in as_completed(fut_map):
            p, ok = f.result()
            if ok:
                lat = next(l for pp, l in working if pp == p)
                token_working.append((p, lat))
                print(f"  ✓ TOKEN OK  {lat:.0f}ms  {p}", file=sys.stderr)
                if len(token_working) >= 30:
                    break

    print(f"Stage 2 result: {len(token_working)} proxies can create guest tokens", file=sys.stderr)

    # Build output
    result = {
        "connectivity_test": [{"url": p, "latency_ms": round(lat, 1)} for p, lat in working[:100]],
        "guest_token_test": [{"url": p, "latency_ms": round(lat, 1)} for p, lat in token_working],
        "summary": {
            "stage1_total": len(proxies),
            "stage1_working": len(working),
            "stage2_tested": len(token_candidates),
            "stage2_working": len(token_working),
        }
    }

    # Save to config
    out_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'verified_proxies.json')
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}", file=sys.stderr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
