"""Proxy rotator with reliability tracking — auto-switches on rate-limit detection.

Tracks per-proxy:
  - success/failure count
  - latency history
  - consecutive failures → blacklist → cooldown → retry
  - total calls served

Reads proxies from:
  1. GLM_PROXY_LIST env var (comma-separated)
  2. GLM_PROXY_0..GLM_PROXY_9 env vars
  3. GLM_PROXY / HTTPS_PROXY / HTTP_PROXY fallback
  4. GLM_PROXY_AUTO_FETCH=true → scrapes free public SOCKS5 lists, verifies against chatglm.cn
Auto-detects rate-limiting via circuit breaker in glm_auth.
Auto-refreshes when >50% of proxies are blacklisted.
"""

from __future__ import annotations

import heapq
import os
import random
import time
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass

# Lazy proxy executor — avoids atexit lifecycle issues with AppImage Python.
_proxy_executor: ThreadPoolExecutor | None = None
_proxy_executor_lock = threading.Lock()


def _get_proxy_executor() -> ThreadPoolExecutor:
    """Get or create the shared proxy executor."""
    global _proxy_executor
    if _proxy_executor is None or _proxy_executor._shutdown:
        with _proxy_executor_lock:
            if _proxy_executor is None or _proxy_executor._shutdown:
                _proxy_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="proxy")
    return _proxy_executor

log = logging.getLogger("glm2api.proxy")

# ponytail: ~50 verified-working public SOCKS5/HTTP/SOCKS4 proxy list URLs.
# Includes major GitHub proxy repos + proxyscrape v2/v3 APIs + regional endpoints.
# Format: plain IP:PORT per line (compatible with _fetch_public_socks5 parser).
# Combined unique total: ~48,000+ proxies across all sources.
# Verified on 2026-07-07 — dead sources removed, new sources added.
_PUBLIC_SOCKS5_URLS = (
    # ============================================================ #
    #  GitHub raw — Major repos (1000+ proxies each)
    # ============================================================ #
    "https://raw.githubusercontent.com/zevtyardt/proxy-list/main/socks5.txt",       # ~8,800 socks5
    "https://raw.githubusercontent.com/zevtyardt/proxy-list/main/http.txt",          # ~10,600 http
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt",      # ~3,000 socks5
    "https://raw.githubusercontent.com/sunny9577/proxy-scraper/master/proxies.txt",  # ~2,200 mixed
    "https://raw.githubusercontent.com/ALIILAPRO/Proxy/main/socks5.txt",             # ~1,700 socks5
    "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-socks5.txt",  # ~400 socks5
    "https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list-raw.txt",  # ~400 mixed
    "https://raw.githubusercontent.com/opsxcq/proxy-list/master/list.txt",           # ~343 mixed
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/socks5.txt",       # ~279 socks5
    "https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt",        # ~206 mixed

    # ============================================================ #
    #  GitHub raw — Medium repos (100-700 proxies each)
    # ============================================================ #
    "https://raw.githubusercontent.com/vakhov/fresh-proxy-list/master/http.txt",     # ~528 http
    "https://raw.githubusercontent.com/almroot/proxylist/master/list.txt",           # ~419 mixed
    "https://raw.githubusercontent.com/vakhov/fresh-proxy-list/master/socks5.txt",   # ~21 socks5
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/http.txt",   # ~140 http
    "https://raw.githubusercontent.com/monosans/proxy-list/main/proxies/socks5.txt", # ~39 socks5
    "https://raw.githubusercontent.com/roosterkid/openproxylist/main/HTTPS_RAW.txt", # ~144 https
    "https://raw.githubusercontent.com/roosterkid/openproxylist/main/SOCKS5_RAW.txt",# ~9 socks5

    # ============================================================ #
    #  Proxy scraper APIs (refresh on every request)
    # ============================================================ #
    "https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&protocol=socks5&timeout=10000",  # ~1,600 socks5
    "https://api.proxyscrape.com/v3/free-proxy-list/get?request=displayproxies&protocol=http&timeout=10000",   # ~990 http
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=all",              # ~1,600 socks5
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all",               # ~966 http
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks4&timeout=10000&country=all",             # ~233 socks4
    "https://api.openproxylist.xyz/socks5.txt",                                                                  # ~4,800 socks5

    # ============================================================ #
    #  Country-specific proxyscrape endpoints (regional diversity)
    # ============================================================ #
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=US",  # ~184
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=CN",  # ~8
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=SG",  # ~11
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=JP",  # ~4
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=HK",  # ~3
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=KR",  # ~2
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=DE",  # ~15
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=GB",  # ~11
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=RU",  # ~28
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=FR",  # ~2
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=ID",  # ~6
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=VN",  # ~3
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=TH",  # ~1

    # ============================================================ #
    #  HTML-scraped sources (reliable, ~300 proxies each)
    # ============================================================ #
    "https://free-proxy-list.net/",
    "https://www.socks-proxy.net/",

    # ============================================================ #
    #  CDN-backed (jsdelivr) — faster delivery for critical repos
    # ============================================================ #
    "https://cdn.jsdelivr.net/gh/TheSpeedX/PROXY-List@master/socks5.txt",
    "https://cdn.jsdelivr.net/gh/ALIILAPRO/Proxy@main/socks5.txt",
    "https://cdn.jsdelivr.net/gh/hookzof/socks5_list@master/proxy.txt",
    "https://cdn.jsdelivr.net/gh/ShiftyTR/Proxy-List@master/socks5.txt",
    "https://cdn.jsdelivr.net/gh/jetkai/proxy-list@main/online-proxies/txt/proxies-socks5.txt",
    "https://cdn.jsdelivr.net/gh/roosterkid/openproxylist@main/SOCKS5_RAW.txt",
    "https://cdn.jsdelivr.net/gh/vakhov/fresh-proxy-list@master/socks5.txt",
    "https://cdn.jsdelivr.net/gh/monosans/proxy-list@main/proxies/socks5.txt",
)


@dataclass
class ProxyScore:
    url: str
    alive: bool = True
    successes: int = 0
    max_successes: int = 2  # ponytail: rotate after 2 successes to stay ahead of WAF
    success_count: int = 0   # ponytail: current-cycle counter, resets on cooldown revive
    failures: int = 0
    consec_failures: int = 0
    total_calls: int = 0
    latency_ms: float = 0.0
    _score_cache: float = 100.0  # ponytail: cached score, recomputed on mutation
    last_used: float = 0.0
    last_fail: float = 0.0
    blacklisted_until: float = 0.0  # timestamp when it can be retried
    verified_working: bool = False   # passed SOCKS5 handshake test
    country: str = ""
    region: str = ""
    premium: bool = False   # Authenticated or VPN proxy, rotate less aggressively

    def __post_init__(self) -> None:
        self._score_cache = self._recompute_score()

    def _recompute_score(self) -> float:
        """Reliability score: higher is better. Based on success rate + latency + depletion.
        ponytail: depletion penalty ensures proxies close to rotation are less favored,
        spreading load across the pool."""
        if self.total_calls == 0:
            return 30.0  # Low starting score until proven
        if self.total_calls < 5:
            confidence = self.total_calls / 5.0
            # Scale base score by confidence
            base = (self.successes / max(self.total_calls, 1)) * 50 * confidence
            return base - min(self.latency_ms / 1000, 10) - (self.consec_failures * 5) + 30 + self._geo_bonus()
        rate = self.successes / max(self.total_calls, 1)
        lat_penalty = min(self.latency_ms / 1000, 10)
        fail_penalty = self.consec_failures * 5
        # ponytail: depletion penalty — deprioritize proxies close to rotation
        depletion_penalty = 0.0
        if self.max_successes > 0 and self.success_count > 0:
            usage_ratio = self.success_count / self.max_successes
            if usage_ratio > 0.5:
                # Linear penalty 0..10 as usage goes from 50% → 100%
                depletion_penalty = (usage_ratio - 0.5) * 2 * 10
        return (rate * 50) - lat_penalty - fail_penalty - depletion_penalty + self._geo_bonus()

    @property
    def score(self) -> float:
        """Cached reliability score. Invalidated on each mutation."""
        return self._score_cache

    def _geo_bonus(self) -> float:
        """Geographic proximity bonus: proxies near China get higher scores."""
        if not self.country:
            return 0.0
        return {"CN": 20.0, "HK": 15.0, "SG": 10.0, "JP": 10.0, "KR": 10.0, "TW": 10.0}.get(self.country, 0.0)

    @property
    def effective_max_successes(self) -> int:
        """Dynamic rotation threshold based on proxy type.
        Premium (authenticated) proxies rotate less aggressively since they have real credentials.
        Public proxies rotate frequently to stay ahead of WAF."""
        if self.premium:
            return 10
        if "127.0.0.1" in self.url:
            return 50  # Local VPN tunnel
        return 2  # Public proxies, rotate after 2 successes


class SmartProxyPool:
    """Manages SOCKS5/HTTP proxy pool with health tracking + auto-switch.

    On rate-limit or degradation, marks proxy as blacklisted with cooldown.
    Cooldown doubles on each failure cycle (10s, 20s, 40s... max 300s).
    After cooldown, proxy is retried automatically.

    When >50% of proxies are blacklisted, auto-refreshes from public sources
    and only keeps proxies that pass SOCKS5 handshake verification.
    """

    COOLDOWN_BASE = 10
    COOLDOWN_MAX = 300
    HEALTH_CHECK_INTERVAL = 60

    def __init__(self) -> None:
        # RLock: get_best/get_next call _auto_replenish which also takes the lock.
        # Non-reentrant Lock deadlocks here; RLock is required.
        self._lock = threading.RLock()
        self._proxies: dict[str, ProxyScore] = {}
        self._current = ""
        self._last_health_check = 0.0
        self._last_refresh = 0.0
        self._guest_rr_index = 0
        self._main_rr_index = 0
        self._hot_pool: list[str] = []  # URLs of hot (recently-verified) proxies
        self._last_hot_refresh: float = 0.0
        self._hot_pool_size: int = 100  # maintain 100 hot proxies
        self._hot_refresh_interval: float = 30.0  # refresh every 30s
        # Seed from on-disk cache first for instant availability
        self._seed_from_cache()
        self._populate()
        self._start_health_checks()
        self._start_background_scraper()
        self._start_hot_pool_refresher()
        # Seed the initial hot pool
        self._refresh_hot_pool()

    @staticmethod
    def _fetch_public_socks5() -> list[str]:
        """Fetch free public SOCKS5 proxies from all sources. Returns socks5://ip:port URLs.
        ponytail: httpx+socksio already installed. Parallel fetch via ThreadPoolExecutor.
        Skips 0.0.0.0, empty lines, and invalid entries. Dead sources are logged, not fatal."""
        import concurrent.futures as _cf
        seen: set[str] = set()
        results: list[str] = []

        def _fetch_one(url: str) -> list[str]:
            try:
                import httpx
                resp = httpx.get(url, timeout=10, follow_redirects=True)
                resp.raise_for_status()
                lines: list[str] = []
                for line in resp.text.strip().split("\n"):
                    addr = line.strip()
                    if not addr or addr.startswith("0.0.0.0") or addr.count(":") != 1:
                        continue
                    if addr not in seen:
                        seen.add(addr)
                        lines.append(f"socks5://{addr}")
                log.info("Fetched %d SOCKS5 proxies from %s", len(lines), url.split("/")[2])
                return lines
            except Exception as exc:
                log.warning("Proxy source failed %s: %s", url.split("/")[2], exc)
                return []

        with _cf.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_fetch_one, u) for u in _PUBLIC_SOCKS5_URLS]
            for f in _cf.as_completed(futures):
                results.extend(f.result())

        log.info("Public proxy fetch done: %d unique proxies", len(results))
        return results

    @staticmethod
    def _verify_socks5(proxy_url: str, target_host: str = "chatglm.cn", target_port: int = 443, timeout: float = 10) -> float | None:
        """Test SOCKS5/HTTP proxy with curl_cffi Chrome 120 impersonation.

        Uses curl_cffi (matching the real HTTP client) instead of raw socket+TLS 1.2.
        This ensures proxies are verified with the same TLS fingerprint the actual
        requests will use — a proxy that passes raw TLS 1.2 may still fail with
        curl_cffi's Chrome 120 JA3 fingerprint (which is what the WAF sees).

        Two-phase test:
          1. GET chatglm.cn homepage — checks basic connectivity + WAF response
          2. POST guest/access endpoint — verifies full token flow works

        Returns latency in ms on success, None on failure.
        """
        try:
            from curl_cffi import requests as _cffi_req
        except ImportError:
            # Fallback to raw socket verification if curl_cffi is unavailable
            return SmartProxyPool._verify_socks5_raw(proxy_url, target_host, target_port, timeout)

        start = time.monotonic()
        proxies = {"https": proxy_url, "http": proxy_url}

        try:
            # Phase 1: Quick GET to chatglm.cn
            resp = _cffi_req.get(
                f"https://{target_host}/",
                impersonate="chrome120",
                proxies=proxies,
                timeout=timeout,
            )
            if resp.status_code != 200:
                return None

            # Phase 2: Guest token POST (the real test)
            import hashlib as _hl, uuid as _uid
            ts = str(int(time.time() * 1000))
            digits = [int(c) for c in ts]
            checksum = (sum(digits) - digits[-2]) % 10
            ts2 = ts[:-2] + str(checksum) + ts[-1]
            nonce = _uid.uuid4().hex
            sign = _hl.md5(f"{ts2}-{nonce}-8a1317a7468aa3ad86e997d08f3f31cb".encode()).hexdigest()

            headers = {
                "Content-Type": "application/json;charset=utf-8",
                "Content-Length": "0",
                "App-Name": "chatglm",
                "Origin": f"https://{target_host}",
                "Referer": f"https://{target_host}/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "X-App-Fr": "browser_extension",
                "X-App-Platform": "pc",
                "X-App-Version": "1.0.83",
                "X-Device-Id": _uid.uuid4().hex,
                "X-Lang": "en",
                "X-Nonce": nonce,
                "X-Request-Id": _uid.uuid4().hex,
                "X-Sign": sign,
                "X-Timestamp": ts2,
            }
            resp2 = _cffi_req.post(
                f"https://{target_host}/chatglm/user-api/guest/access",
                impersonate="chrome120",
                proxies=proxies,
                headers=headers,
                timeout=timeout + 5,
            )
            if resp2.status_code == 200:
                data = resp2.json()
                if data.get("status") == 0 and data.get("result", {}).get("access_token"):
                    return (time.monotonic() - start) * 1000
            return None
        except Exception:
            return None

    @staticmethod
    def _verify_socks5_raw(proxy_url: str, target_host: str = "chatglm.cn", target_port: int = 443, timeout: float = 3) -> float | None:
        """Fallback raw socket SOCKS5 verifier (original method).
        Used when curl_cffi is not installed."""
        import socket as _sck, hashlib, uuid, ssl as _ssl
        raw = proxy_url.replace("socks5://", "").replace("http://", "").replace("https://", "")
        host, port_str = raw.split(":") if ":" in raw else (raw, "1080")
        port = int(port_str)
        try:
            start = time.monotonic()
            s = _sck.create_connection((host, port), timeout=timeout)
            s.sendall(b"\x05\x01\x00")
            if s.recv(2) != b"\x05\x00": s.close(); return None
            hostname = target_host.encode()
            s.sendall(b"\x05\x01\x00\x03" + bytes([len(hostname)]) + hostname + target_port.to_bytes(2, "big"))
            resp = s.recv(10)
            if len(resp) < 2 or resp[1] != 0x00: s.close(); return None

            ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
            ctx.minimum_version = _ssl.TLSVersion.TLSv1_2
            ctx.maximum_version = _ssl.TLSVersion.TLSv1_2
            ctx.check_hostname = False; ctx.verify_mode = _ssl.CERT_NONE
            ss = ctx.wrap_socket(s, server_hostname=target_host)

            ts = str(int(time.time() * 1000))
            digits = [int(c) for c in ts]
            checksum = (sum(digits) - digits[-2]) % 10
            ts2 = ts[:-2] + str(checksum) + ts[-1]
            nonce = uuid.uuid4().hex
            sign = hashlib.md5(f"{ts2}-{nonce}-8a1317a7468aa3ad86e997d08f3f31cb".encode()).hexdigest()

            req = (
                f"POST /chatglm/user-api/guest/access HTTP/1.1\r\n"
                f"Host: {target_host}\r\n"
                f"Content-Type: application/json;charset=utf-8\r\n"
                f"Content-Length: 0\r\n"
                f"App-Name: chatglm\r\n"
                f"Origin: https://{target_host}\r\n"
                f"Referer: https://{target_host}/\r\n"
                f"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36\r\n"
                f"X-App-Fr: browser_extension\r\n"
                f"X-App-Platform: pc\r\n"
                f"X-App-Version: 1.0.83\r\n"
                f"X-Device-Id: {uuid.uuid4().hex}\r\n"
                f"X-Lang: en\r\n"
                f"X-Nonce: {nonce}\r\n"
                f"X-Request-Id: {uuid.uuid4().hex}\r\n"
                f"X-Sign: {sign}\r\n"
                f"X-Timestamp: {ts2}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            ).encode()

            ss.sendall(req)
            resp = b""
            while True:
                try:
                    chunk = ss.recv(4096)
                    if not chunk: break
                    resp += chunk
                except: break
            ss.close()

            if b"HTTP/1.1 200" in resp:
                import json as _json
                body = resp[resp.index(b"\r\n\r\n") + 4:]
                try:
                    data = _json.loads(body.decode())
                    if data.get("status") == 0:
                        return (time.monotonic() - start) * 1000
                except: pass
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Fast 3-phase progressive verification
    # ------------------------------------------------------------------ #

    @staticmethod
    def _tcp_precheck(urls: list[str], timeout: float = 3.0) -> list[str]:
        """Quick TCP connect check to filter dead proxies before curl_cffi test.

        Tests hundreds of proxies in <5s using raw sockets with 200 concurrent workers.
        Filters out proxies where the host:port is not reachable, leaving only those
        that pass the TCP handshake for further curl_cffi verification."""
        import socket as _sck, concurrent.futures as _cf

        def _check(u: str) -> str | None:
            try:
                raw = u.replace('socks5://', '').replace('http://', '').replace('https://', '')
                host, port = raw.split('@')[-1].split(':')
                s = _sck.create_connection((host, int(port)), timeout=timeout)
                s.close()
                return u
            except Exception:
                return None

        working: list[str] = []
        with _cf.ThreadPoolExecutor(max_workers=200) as pool:
            for result in pool.map(_check, urls):
                if result:
                    working.append(result)
        return working

    def _fast_verify_one(self, proxy_url: str) -> float | None:
        """Lightning-fast proxy verification using direct curl_cffi (no SessionPool warmup).

        Creates curl_cffi sessions WITHOUT the SessionPool warmup GET to avoid
        doubling the verification time. Phase 1 (GET) serves as the warmup.

        Phase 1: GET chatglm.cn (5s timeout) — warms WAF cookies + checks connectivity.
        Phase 2: POST guest/access (6s timeout) — verifies full token flow.
        Total worst case: ~11s per proxy.
        """
        try:
            from curl_cffi import requests as _cffi
            from curl_cffi.const import CurlHttpVersion
        except ImportError:
            return SmartProxyPool._verify_socks5_raw(proxy_url)

        start = time.monotonic()
        try:
            proxies = {"https": proxy_url, "http": proxy_url}
            # NO warmup — Phase 1 GET serves as warmup
            session = _cffi.Session(
                impersonate="chrome120",
                proxies=proxies,
                timeout=30,
                allow_redirects=True,
                http_version=CurlHttpVersion.V2TLS,  # HTTP/2 for multiplexed streams
                curl_options={
                    121: 1,   # TCP_NODELAY
                    92: 300,  # DNS cache 5min
                    244: 1,   # TCP Fast Open
                },
            )

            # Phase 1: GET chatglm.cn (5s) — warms WAF cookies + checks connectivity
            resp = session.get(
                "https://chatglm.cn/",
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
                timeout=5,
            )
            if resp.status_code != 200:
                session.close()
                return None

            # Phase 2: POST guest/access (6s)
            import hashlib as _hl, uuid as _uid
            ts = str(int(time.time() * 1000))
            digits = [int(c) for c in ts]
            checksum = (sum(digits) - digits[-2]) % 10
            ts2 = ts[:-2] + str(checksum) + ts[-1]
            nonce = _uid.uuid4().hex
            sign = _hl.md5(f"{ts2}-{nonce}-8a1317a7468aa3ad86e997d08f3f31cb".encode()).hexdigest()

            headers = {
                "Content-Type": "application/json;charset=utf-8",
                "Content-Length": "0",
                "Accept": "application/json, text/plain, */*",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "App-Name": "chatglm",
                "Cache-Control": "no-cache",
                "Origin": "https://chatglm.cn",
                "Pragma": "no-cache",
                "Referer": "https://chatglm.cn/",
                "Sec-Ch-Ua": '"Google Chrome";v="120", "Chromium";v="120", "Not A(Brand";v="24"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Linux"',
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "X-App-Fr": "browser_extension",
                "X-App-Platform": "pc",
                "X-App-Version": "1.0.83",
                "X-Device-Id": _uid.uuid4().hex,
                "X-Lang": "en",
                "X-Nonce": nonce,
                "X-Request-Id": _uid.uuid4().hex,
                "X-Sign": sign,
                "X-Timestamp": ts2,
            }
            resp2 = session.post(
                "https://chatglm.cn/chatglm/user-api/guest/access",
                headers=headers,
                timeout=6,
            )
            session.close()

            if resp2.status_code == 200:
                data = resp2.json()
                if data.get("status") == 0 and data.get("result", {}).get("access_token"):
                    return (time.monotonic() - start) * 1000
            return None
        except Exception:
            return None

    def _fast_verify_and_filter(self, urls: list[str], min_working: int = 200, max_to_check: int = 1500) -> list[str]:
        """3-phase progressive proxy verification using TCP pre-check + SessionPool.

        Phase 0: TCP connect pre-check (3s, 200 workers) — filters dead proxies fast.
        Phase 1: curl_cffi GET chatglm.cn via SessionPool (5s) — checks WAF response.
        Phase 2: curl_cffi POST guest/access via SessionPool (8s) — verifies token flow.

        Progressive: Phase 0 runs on ALL proxies, Phases 1+2 run only on survivors.
        Cancels remaining checks once min_working targets are met.

        With 1500 proxies and 50 concurrent workers, this typically completes in
        ~15-30s (vs 120-300s for the old two-phase curl_cffi-only approach)."""
        import concurrent.futures as _cf

        check = urls[:max_to_check]
        log.info("Fast proxy verification: %d proxies via 3-phase progressive test (need %d)...", len(check), min_working)

        # Phase 0: TCP pre-check — kills dead proxies in ~3-5s
        check = self._tcp_precheck(check, timeout=3.0)
        if not check:
            log.info("  Phase 0 (TCP connect): 0/%d passed, nothing to verify", len(urls[:max_to_check]))
            return []

        log.info("  Phase 0 (TCP connect): %d/%d passed", len(check), len(urls[:max_to_check]))

        # Phases 1+2: curl_cffi GET + POST via SessionPool (50 workers)
        working: list[tuple[str, float]] = []
        total_checked = len(check)

        with _cf.ThreadPoolExecutor(max_workers=50) as pool:
            fut_map = {pool.submit(self._fast_verify_one, u): u for u in check}
            for f in _cf.as_completed(fut_map):
                u = fut_map[f]
                try:
                    latency = f.result(timeout=15)
                    if latency is not None:
                        working.append((u, latency))
                        if len(working) >= min_working:
                            # Cancel remaining futures, we have enough verified proxies
                            for of in fut_map:
                                of.cancel()
                            break
                except Exception:
                    pass

        result = [u for u, _ in working]
        log.info("Fast verification done: %d/%d proxies work via curl_cffi Chrome 120", len(result), total_checked)
        return result

    def _verify_and_filter(self, urls: list[str], min_working: int = 200, max_to_check: int = 1500) -> list[str]:
        """Verify proxies against chatglm.cn using curl_cffi Chrome 120 impersonation.
        ponytail: checks first max_to_check proxies concurrently, cancels remaining once target hit.
        Uses curl_cffi (not raw socket) for verification — matches how proxy is used in production.
        ~1500 x 15s / 50 workers = ~450s worst case, typically ~120s."""
        import concurrent.futures as _cf
        check = urls[:max_to_check]
        log.info("Verifying %d proxies against chatglm.cn via curl_cffi (need %d)...", len(check), min_working)
        working: list[tuple[str, float]] = []

        with _cf.ThreadPoolExecutor(max_workers=50) as pool:
            fut_map = {pool.submit(self._verify_socks5, u): u for u in check}
            for f in _cf.as_completed(fut_map):
                u = fut_map[f]
                try:
                    latency = f.result(timeout=25)  # curl_cffi two-phase test needs more time
                    if latency is not None:
                        working.append((u, latency))
                        if len(working) >= min_working:
                            # Cancel remaining futures, we have enough
                            for of in fut_map:
                                of.cancel()
                            break
                except Exception:
                    pass

        result = [u for u, _ in working]
        log.info("Verification done: %d/%d proxies work via curl_cffi Chrome 120", len(result), len(check))
        return result

    def _auto_refresh(self) -> None:
        """When many proxies are dead, auto-fetch new verified ones in background.
        ponytail: only refreshes once per 300s. Runs in daemon thread, never blocks caller."""
        try:
            now = time.monotonic()
            if now - self._last_refresh < 300:
                return
            with self._lock:
                total = len(self._proxies)
                alive = sum(1 for p in self._proxies.values() if p.alive)
                # Refresh when <30% alive — triggers regardless of pool size
                if alive / max(total, 1) > 0.3:
                    return
                log.warning("Proxy pool degraded: %d/%d alive. Auto-refreshing...", alive, total)

            # Fetch and verify new proxies (outside lock to avoid blocking)
            fetched = self._fetch_public_socks5()
            verified = self._fast_verify_and_filter(fetched, min_working=50, max_to_check=200) if fetched else []
            with self._lock:
                now = time.monotonic()
                added = 0
                for u in verified:
                    if u not in self._proxies:
                        self._proxies[u] = ProxyScore(url=u, verified_working=True, alive=True)
                        added += 1
                self._last_refresh = now
            log.info("Auto-refresh complete: added %d fresh proxies (pool now %d total)", added, len(self._proxies))
        except Exception as exc:
            log.warning("Auto-refresh failed: %s", exc)

    def _auto_replenish(self) -> None:
        """Emergency replenish: when <5 proxies are alive, fetch & test fresh ones with httpx.
        Uses short timeouts (5s connect, 8s total) so the caller doesn't hang for long.
        ponytail: lightweight httpx-only verification, no curl_cffi dependency needed here."""
        try:
            now = time.monotonic()
            if now - self._last_refresh < 30:  # ponytail: 30s cooldown for faster proxy refresh
                return
            with self._lock:
                alive = sum(1 for p in self._proxies.values() if p.alive)
                if alive >= 5:
                    return
                total = len(self._proxies)
                log.warning("Proxy pool critically low: %d/%d alive. Replenishing...", alive, total)

            fetched = self._fetch_public_socks5()
            if not fetched:
                return

            import httpx as _httpx
            import hashlib as _hl
            import uuid as _uid

            working: list[str] = []
            for url in fetched:
                try:
                    ts = str(int(time.time() * 1000))
                    digits = [int(c) for c in ts]
                    checksum = (sum(digits) - digits[-2]) % 10
                    ts2 = ts[:-2] + str(checksum) + ts[-1]
                    nonce = _uid.uuid4().hex
                    sign = _hl.md5(f"{ts2}-{nonce}-8a1317a7468aa3ad86e997d08f3f31cb".encode()).hexdigest()

                    headers = {
                        "Content-Type": "application/json;charset=utf-8",
                        "Content-Length": "0",
                        "App-Name": "chatglm",
                        "Origin": "https://chatglm.cn",
                        "Referer": "https://chatglm.cn/",
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "X-App-Fr": "browser_extension",
                        "X-App-Platform": "pc",
                        "X-App-Version": "1.0.83",
                        "X-Device-Id": _uid.uuid4().hex,
                        "X-Lang": "en",
                        "X-Nonce": nonce,
                        "X-Request-Id": _uid.uuid4().hex,
                        "X-Sign": sign,
                        "X-Timestamp": ts2,
                    }

                    with _httpx.Client(proxies=url, timeout=_httpx.Timeout(8.0, connect=5.0)) as client:
                        resp = client.post(
                            "https://chatglm.cn/chatglm/user-api/guest/access",
                            headers=headers,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get("status") == 0 and data.get("result", {}).get("access_token"):
                                working.append(url)
                except Exception:
                    continue

            with self._lock:
                now = time.monotonic()
                added = 0
                for u in working:
                    if u not in self._proxies:
                        self._proxies[u] = ProxyScore(url=u, verified_working=True, alive=True)
                        added += 1
                self._last_refresh = now
                alive_after = sum(1 for p in self._proxies.values() if p.alive)
                log.info("Auto-replenish added %d fresh proxies (pool: %d total, %d alive)",
                         added, len(self._proxies), alive_after)
        except Exception as exc:
            log.warning("Auto-replenish failed: %s", exc)

    def _seed_from_cache(self) -> None:
        """Load previously-verified proxies from disk cache for instant cold-start availability."""
        cache_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'verified_proxies.json')
        if not os.path.exists(cache_path):
            return
        import json
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            seeded = 0
            for entry in cache.get("guest_token_test", []):
                url = entry.get("url", "")
                if url and url not in self._proxies:
                    self._proxies[url] = ProxyScore(
                        url=url,
                        verified_working=True,
                        alive=True,
                        latency_ms=entry.get("latency_ms", 0),
                    )
                    seeded += 1
            log.info("Seeded %d proxies from verified_proxies.json cache", seeded)
        except Exception as exc:
            log.warning("Failed to load proxy cache: %s", exc)

    def _populate(self) -> None:
        urls: list[str] = []
        lst = os.environ.get("GLM_PROXY_LIST", "").strip()
        if lst:
            urls.extend(u.strip() for u in lst.split(",") if u.strip())
        extra = os.environ.get("GLM_PROXY_LIST_EXTRA", "").strip()
        if extra:
            urls.extend(u.strip() for u in extra.split(",") if u.strip())
        for i in range(10):
            val = os.environ.get(f"GLM_PROXY_{i}", "").strip()
            if val:
                urls.append(val)
        if not urls:
            for var in ("GLM_PROXY", "HTTPS_PROXY", "HTTP_PROXY"):
                val = os.environ.get(var, "").strip()
                if val:
                    urls.append(val)
                    break
        self._proxies = {}
        for u in urls:
            score = ProxyScore(url=u)
            if "@" in u:
                score.premium = True  # Authenticated proxy, rotate less aggressively
            self._proxies[u] = score
        # GLM_PROXY_AUTO_FETCH=true: always scrape + verify against chatglm.cn,
        # supplementing env-sourced proxies with fresh public ones
        if os.environ.get("GLM_PROXY_AUTO_FETCH", "").strip().lower() in ("true", "1", "yes"):
            fetched = self._fetch_public_socks5()
            verified = self._fast_verify_and_filter(fetched, min_working=50, max_to_check=200)
            for u in verified:
                if u not in self._proxies:
                    self._proxies[u] = ProxyScore(url=u, verified_working=True)
            if verified:
                log.info("Auto-fetch complete: %d verified SOCKS5 proxies (pool total: %d)", len(verified), len(self._proxies))
        # Auto-discover multi-node VPN proxies on localhost
        # self._discover_multi_vpn() — disabled: VPN proxy TLS issues
        self._current = next(iter(self._proxies)) if self._proxies else ""
        # Geo-locate newly added proxies
        self._batch_geolocate()

    def _batch_geolocate(self, proxies: list[ProxyScore] | None = None) -> None:
        """Batch geo-IP lookup for proxies using ip-api.com (free, 45 req/min)."""
        if proxies is None:
            with self._lock:
                proxies = [p for p in self._proxies.values() if not p.country and p.alive]
        if not proxies:
            return
        # Respect rate limit: max 45 per batch
        batch = proxies[:45]
        import httpx as _httpx
        with _httpx.Client() as client:
            for proxy in batch:
                ip = proxy.url.split("@")[-1].split(":")[0].replace("/", "")
                try:
                    r = client.get(f"http://ip-api.com/json/{ip}", timeout=3)
                    if r.status_code == 200:
                        data = r.json()
                        if data.get("status") == "success":
                            proxy.country = data.get("countryCode", "")
                            proxy.region = data.get("regionName", "")
                except Exception:
                    pass

    def _start_health_checks(self) -> None:
        def _loop():
            while True:
                time.sleep(self.HEALTH_CHECK_INTERVAL)
                try:
                    self._check_all()
                except Exception as exc:
                    log.error("Health-check thread crashed: %s", exc, exc_info=True)
                    # Continue looping — thread must survive transient errors
        threading.Thread(target=_loop, daemon=True).start()

    def _start_background_scraper(self) -> None:
        """Daemon thread: every 120s fetch fresh public SOCKS5 proxies, test each
        against chatglm.cn via httpx, and add working ones to the pool.
        ponytail: lightweight httpx-only, no curl_cffi. Runs forever in background."""
        def _loop():
            while True:
                time.sleep(120)  # ponytail: faster refresh (120s) to stay ahead of WAF bans
                try:
                    fetched = self._fetch_public_socks5()
                    if not fetched:
                        continue
                    import httpx as _httpx
                    import hashlib as _hl, uuid as _uid
                    working: list[str] = []
                    for url in fetched:
                        try:
                            ts = str(int(time.time() * 1000))
                            digits = [int(c) for c in ts]
                            checksum = (sum(digits) - digits[-2]) % 10
                            ts2 = ts[:-2] + str(checksum) + ts[-1]
                            nonce = _uid.uuid4().hex
                            sign = _hl.md5(f"{ts2}-{nonce}-8a1317a7468aa3ad86e997d08f3f31cb".encode()).hexdigest()
                            headers = {
                                "Content-Type": "application/json;charset=utf-8",
                                "Content-Length": "0",
                                "App-Name": "chatglm",
                                "Origin": "https://chatglm.cn",
                                "Referer": "https://chatglm.cn/",
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                                "X-App-Fr": "browser_extension",
                                "X-App-Platform": "pc",
                                "X-App-Version": "1.0.83",
                                "X-Device-Id": _uid.uuid4().hex,
                                "X-Lang": "en",
                                "X-Nonce": nonce,
                                "X-Request-Id": _uid.uuid4().hex,
                                "X-Sign": sign,
                                "X-Timestamp": ts2,
                            }
                            with _httpx.Client(proxies=url, timeout=_httpx.Timeout(5.0, connect=5.0)) as client:
                                resp = client.post(
                                    "https://chatglm.cn/chatglm/user-api/guest/access",
                                    headers=headers,
                                )
                                if resp.status_code == 200:
                                    data = resp.json()
                                    if data.get("status") == 0 and data.get("result", {}).get("access_token"):
                                        working.append(url)
                        except Exception:
                            continue
                    with self._lock:
                        now = time.monotonic()
                        added = 0
                        for u in working:
                            if u not in self._proxies:
                                self._proxies[u] = ProxyScore(url=u, verified_working=True, alive=True)
                                added += 1
                        log.info("Background scraper found %d new proxies", added)
                except Exception as exc:
                    log.warning("Background scraper failed: %s", exc)
        threading.Thread(target=_loop, daemon=True).start()

    def _check_all(self) -> None:
        """Ping each proxy concurrently, update latency + health.
        ponytail: ThreadPoolExecutor batches health checks so 6000+ proxies
        finish in ~30s instead of hours. Dead proxies get blacklisted with
        exponential backoff cooldown. Also retries recently dead proxies
        (within last 300s) more aggressively."""
        import socket as _sck, concurrent.futures as _cf
        now = time.monotonic()
        with self._lock:
            # Check alive proxies AND recently dead ones (within last 300s)
            proxies = [p for p in self._proxies.values() 
                       if p.blacklisted_until <= now or 
                       (not p.alive and now - p.last_fail < 300)]

        def _check_one(p) -> None:
            # ponytail: HTTP-level test through proxy, not just TCP connect.
            # TCP connect only checks if SOCKS5 proxy is running, but can't
            # detect WAF blocks or upstream unreachable. HTTP test catches both.
            try:
                proxy_url = p.url
                import httpx as _httpx
                start = time.monotonic()
                with _httpx.Client(proxy=proxy_url, timeout=_httpx.Timeout(6, connect=4)) as _c:
                    _r = _c.get("https://chatglm.cn/", headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "text/html",
                    })
                    _r.read()  # consume response to ensure proxy works end-to-end
                    p.latency_ms = (time.monotonic() - start) * 1000
                    p.alive = True
                    p.consec_failures = 0
                    p.success_count = 0
                    p._score_cache = p._recompute_score()
            except Exception:
                p.consec_failures += 1
                p._score_cache = p._recompute_score()
                if p.consec_failures >= 2:
                    p.alive = False
                    p.blacklisted_until = time.monotonic() + min(
                        self.COOLDOWN_BASE * (2 ** min(p.consec_failures - 2, 5)),
                        self.COOLDOWN_MAX
                    )

        with _cf.ThreadPoolExecutor(max_workers=50) as pool:
            list(pool.map(_check_one, proxies))

        # -- Auto-discover any new multi-node VPN proxies --
        # self._discover_multi_vpn() — disabled: VPN proxy TLS issues

        # -- Specific check for the aimiligate VPN proxy (http://127.0.0.1:7928) --
        # The generic TCP-level check above can report the VPN proxy as alive because
        # the local proxy server process is still listening on the port, even when the
        # VPN tunnel (tun0) is broken and actual proxied requests return 502.
        # Here we do an application-level probe through the proxy to detect that case.
        self._check_vpn_proxy(now)

    # ------------------------------------------------------------------ #
    #  Aimiligate VPN proxy special handling
    # ------------------------------------------------------------------ #
    _VPN_PROXY_MARKERS = ["127.0.0.1:7928", "127.0.0.1:7929", "127.0.0.1:7930",
                          "127.0.0.1:7931", "127.0.0.1:7932", "127.0.0.1:7933",
                          "127.0.0.1:7934", "127.0.0.1:7935"]

    def _check_vpn_proxy(self, now: float) -> None:
        """Application-level health check for the aimiligate VPN proxy(es).

        For each local VPN proxy (127.0.0.1:7928-7935), do an application-level
        probe through the proxy to verify real connectivity.

        If the primary proxy (7928) has been unreachable for >60s, trigger a
        VPN reconnect.
        """
        vpn_proxies: list[tuple[str, ProxyScore]] = []
        primary_vpn: ProxyScore | None = None
        with self._lock:
            for p in self._proxies.values():
                for marker in self._VPN_PROXY_MARKERS:
                    if marker in p.url:
                        vpn_proxies.append((marker, p))
                        if marker == "127.0.0.1:7928":
                            primary_vpn = p
                        break

        if not vpn_proxies:
            return  # No VPN proxies in this pool, nothing to do

        now_m = time.monotonic()
        for marker, vpn_proxy in vpn_proxies:
            # Do an actual HTTP request through the proxy to verify real connectivity
            vpn_alive = self._probe_vpn_proxy(vpn_proxy.url)

            if not vpn_alive:
                # Mark it as failed in the pool
                vpn_proxy.consec_failures += 1
                vpn_proxy.last_fail = now_m
                if vpn_proxy.consec_failures >= 2:
                    vpn_proxy.alive = False
                    vpn_proxy.blacklisted_until = now_m + min(
                        self.COOLDOWN_BASE * (2 ** min(vpn_proxy.consec_failures - 2, 5)),
                        self.COOLDOWN_MAX,
                    )
                vpn_proxy._score_cache = vpn_proxy._recompute_score()
                log.warning(
                    "VPN proxy %s application-level check failed (%d consecutive)",
                    vpn_proxy.url, vpn_proxy.consec_failures,
                )
            else:
                # Restore to alive if it was down
                if not vpn_proxy.alive:
                    log.info("VPN proxy %s is back online", vpn_proxy.url)
                    vpn_proxy.alive = True
                    vpn_proxy.consec_failures = 0
                    vpn_proxy.blacklisted_until = 0.0
                    vpn_proxy._score_cache = vpn_proxy._recompute_score()

        # If the primary VPN proxy (7928 — tun0, main vpngate_manager) has been
        # down for >60s, trigger a full VPN reconnect
        if primary_vpn is not None and now_m - primary_vpn.last_fail > 60:
            log.warning("Primary VPN proxy down for >60s, triggering VPN reconnect")
            self._reconnect_vpn()

    @staticmethod
    def _probe_vpn_proxy(proxy_url: str, target: str = "https://chatglm.cn/") -> bool:
        """Make an actual HTTP request through the proxy to verify real connectivity.

        This catches the case where the proxy port is listening but the VPN tunnel
        is broken (returning 502/bad gateway).
        """
        try:
            import urllib.request
            proxy_handler = urllib.request.ProxyHandler({
                "http": proxy_url,
                "https": proxy_url,
            })
            opener = urllib.request.build_opener(proxy_handler)
            resp = opener.open(target, timeout=10)
            return resp.status == 200
        except Exception:
            return False

    @staticmethod
    def _reconnect_vpn() -> None:
        """Try to restart the aimiligate VPN.

        Kills the vpngate_manager process and starts a fresh instance.
        The watchdog at scripts/vpn_watchdog.py will also catch this case,
        but this provides faster in-process detection and recovery.
        """
        import subprocess as _sp
        vpngate_path = "/home/uluru/aimili-vpngate/vpngate_manager.py"
        try:
            # Kill existing process
            _sp.run(
                ["pkill", "-9", "-f", "vpngate_manager.py"],
                timeout=5, capture_output=True,
            )
            time.sleep(2)
            # Also kill lingering OpenVPN processes on tun0
            _sp.run(
                ["pkill", "-9", "-f", "openvpn.*tun0"],
                timeout=3, capture_output=True,
            )
            time.sleep(1)
            # Start fresh
            _sp.Popen(
                ["python3", vpngate_path],
                stdout=_sp.DEVNULL,
                stderr=_sp.DEVNULL,
                stdin=_sp.DEVNULL,
            )
            log.warning("VPN reconnect triggered — waiting for tunnel...")
        except Exception as exc:
            log.warning("VPN reconnect failed: %s", exc)

    def _refresh_hot_pool(self) -> None:
        """Promote top-scoring proxies to the hot pool for fast O(1) selection."""
        with self._lock:
            candidates = [p for p in self._proxies.values() if p.alive and p.blacklisted_until <= time.monotonic()]
            top = heapq.nlargest(self._hot_pool_size, candidates, key=lambda p: p.score + (5 if p.premium else 0))
            self._hot_pool = [p.url for p in top]
            self._last_hot_refresh = time.monotonic()

    def _start_hot_pool_refresher(self) -> None:
        """Continuous background hot pool refresher daemon thread."""
        def _loop():
            while True:
                time.sleep(self._hot_refresh_interval)
                try:
                    self._refresh_hot_pool()
                except Exception as exc:
                    log.warning("Hot pool refresh failed: %s", exc)
        threading.Thread(target=_loop, daemon=True).start()

    def get_best(self) -> str | None:
        """Return a proxy URL using hot pool for O(1) weighted random selection.
        Falls back to full scan (_get_best_fallback) when hot pool is empty.
        
        Hot pool is kept fresh by a dedicated background thread, so no inline
        refresh check is needed here.
        """
        now = time.monotonic()

        with self._lock:
            # Revive expired cooldowns first
            for p in self._proxies.values():
                if p.blacklisted_until and now >= p.blacklisted_until and not p.alive:
                    p.alive = True
                    p.blacklisted_until = 0
                    p.consec_failures = 0
                    p.success_count = 0

            # Filter hot pool: remove depleted/expired entries
            active_hot = [url for url in self._hot_pool
                          if url in self._proxies
                          and self._proxies[url].alive
                          and self._proxies[url].blacklisted_until <= now
                          and self._proxies[url].success_count < self._proxies[url].max_successes]

            if active_hot:
                # Weighted random from active hot pool
                total_score = sum(self._proxies[url].score for url in active_hot)
                if total_score > 0:
                    r = random.uniform(0, total_score)
                    cumulative = 0.0
                    for url in active_hot:
                        cumulative += self._proxies[url].score
                        if r <= cumulative:
                            self._current = url
                            return url
                chosen = random.choice(active_hot)
                self._current = chosen
                return chosen

        # Hot pool empty — fallback to full scan (original O(n) logic preserved)
        return self._get_best_fallback(now)

    def _get_best_fallback(self, now: float) -> str | None:
        """Original get_best() logic — full scan with heapq.nlargest top-N selection.
        Used as fallback when hot pool is empty or depleted.
        This preserves the original O(n log TOP_N) behavior for edge cases."""
        TOP_N = 15

        # Phase 1 (under lock): revive expired cooldowns, gather candidates + soonest.
        with self._lock:
            candidates: list[ProxyScore] = []
            soonest: ProxyScore | None = None
            for p in self._proxies.values():
                # Revive expired cooldowns
                if p.blacklisted_until and now >= p.blacklisted_until and not p.alive:
                    p.alive = True
                    p.blacklisted_until = 0
                    p.consec_failures = 0
                    p.success_count = 0
                # Track alive candidates (skip depleted ones needing cooldown)
                if p.alive and p.blacklisted_until <= now and p.success_count < p.max_successes:
                    candidates.append(p)
                # Track soonest to recover for fallback
                if not p.alive and p.blacklisted_until > now:
                    if soonest is None or p.blacklisted_until < soonest.blacklisted_until:
                        soonest = p

            if candidates:
                # Pick top N by score using heapq (O(N log M) vs O(N log N) for full sort)
                top = heapq.nlargest(TOP_N, candidates, key=lambda p: p.score + (5 if p.premium else 0))
                positive = [p for p in top if p.score > 0]
                pool_for_pick = positive if positive else top
                total_score = sum(p.score for p in pool_for_pick)
                if total_score > 0:
                    r = random.uniform(0, total_score)
                    cumulative = 0.0
                    for p in pool_for_pick:
                        cumulative += p.score
                        if r <= cumulative:
                            self._current = p.url
                            return p.url
                # Fallback: uniform random from top
                chosen = random.choice(pool_for_pick)
                self._current = chosen.url
                return chosen.url

        # No alive candidates. The expensive operations below run OUTSIDE the lock
        # to avoid blocking other threads on a 2s socket connect or replenish fetch.
        if len(self._proxies) > 10 and now - self._last_refresh > 300:
            threading.Thread(target=self._auto_refresh, daemon=True).start()

        # Emergency replenish (outside lock).
        self._auto_replenish()

        # Re-check candidates after replenish may have added proxies.
        with self._lock:
            revived = [p for p in self._proxies.values()
                       if p.alive and p.blacklisted_until <= now and p.success_count < p.max_successes]
            if revived:
                chosen = random.choice(revived)
                self._current = chosen.url
                return chosen.url

        # Retry the soonest proxy with an immediate socket probe (outside lock).
        if soonest is not None and now - soonest.last_fail > 30:
            try:
                import socket as _sck
                host, port = soonest.url.replace("http://","").replace("socks5://","").replace("https://","").split(":")
                _sck.create_connection((host, int(port)), timeout=2).close()
                with self._lock:
                    soonest.alive = True
                    soonest.blacklisted_until = 0
                    soonest.consec_failures = 0
                    soonest.success_count = 0
                    soonest._score_cache = soonest._recompute_score()
                    self._current = soonest.url
                log.info("Proxy back online (immediate retry): %s", soonest.url)
                return soonest.url
            except Exception:
                pass

        if soonest is not None:
            log.warning("All proxies blacklisted, retrying soonest: %s (remaining %ss)",
                        soonest.url, max(0, soonest.blacklisted_until - now))
            with self._lock:
                self._current = soonest.url
            return soonest.url

        return None

    def get_unique(self) -> str | None:
        """Return a proxy URL, rotating through all available proxies.
        Used for guest token creation to avoid per-IP rate limiting."""
        now = time.monotonic()
        with self._lock:
            candidates: list[ProxyScore] = []
            for p in self._proxies.values():
                if p.blacklisted_until and now >= p.blacklisted_until and not p.alive:
                    p.alive = True
                    p.blacklisted_until = 0
                    p.consec_failures = 0
                    p.success_count = 0
                if p.alive and p.blacklisted_until <= now:
                    candidates.append(p)

            if candidates:
                chosen = candidates[self._guest_rr_index % len(candidates)]
                self._guest_rr_index = (self._guest_rr_index + 1) % len(candidates)
                self._current = chosen.url
                return chosen.url
            return None

    def get_next(self) -> str | None:
        """Return the next proxy using strict round-robin through ALL alive proxies.
        Unlike get_best(), this does NOT use score weighting — it cycles through
        all candidates evenly, preferring least-used proxies first.
        This ensures every request gets a different proxy to bypass per-IP rate limits."""
        now = time.monotonic()
        with self._lock:
            # Revive expired cooldowns
            for p in self._proxies.values():
                if p.blacklisted_until and now >= p.blacklisted_until and not p.alive:
                    p.alive = True
                    p.blacklisted_until = 0
                    p.consec_failures = 0
                    p.success_count = 0

            candidates = [p for p in self._proxies.values() if p.alive and p.blacklisted_until <= now]
            if candidates:
                # Sort by total_calls ascending so least-used proxies are picked first
                candidates.sort(key=lambda p: p.total_calls)
                self._main_rr_index %= len(candidates)
                chosen = candidates[self._main_rr_index]
                self._main_rr_index = (self._main_rr_index + 1) % len(candidates)
                self._current = chosen.url
                return chosen.url

        # No alive proxies — try any dead ones via an immediate socket probe (outside lock).
        with self._lock:
            dead_candidates = [p for p in self._proxies.values()
                               if not p.alive and now - p.last_fail > 15]
        for p in dead_candidates:
            try:
                host, port = p.url.replace("http://","").replace("socks5://","").replace("https://","").split(":")
                import socket as _sck
                _sck.create_connection((host, int(port)), timeout=2).close()
                with self._lock:
                    p.alive = True
                    p.blacklisted_until = 0
                    p.consec_failures = 0
                    p.success_count = 0
                    self._current = p.url
                log.info("Proxy back online: %s", p.url)
                return p.url
            except Exception:
                with self._lock:
                    p.last_fail = now
                continue

        # All blacklisted — trigger background auto-refresh, then replenish (both outside lock).
        if len(self._proxies) > 10 and now - self._last_refresh > 300:
            threading.Thread(target=self._auto_refresh, daemon=True).start()
        self._auto_replenish()
        return None

    def report_success(self, url: str, latency: float = 0) -> None:
        now = time.monotonic()
        with self._lock:
            p = self._proxies.get(url)
            if p:
                p.successes += 1
                p.success_count += 1
                if p.success_count >= p.effective_max_successes:
                    p.alive = False
                    p.blacklisted_until = now + 120
                    # Reset success_count here so the depletion contract is self-contained
                    # and does not depend on health-check / get_best revive paths clearing it.
                    p.success_count = 0
                    log.info("Proxy depleted after %d successes, rotating: %s", p.effective_max_successes, url)
                else:
                    p.alive = True
                p.total_calls += 1
                p.consec_failures = 0
                p.latency_ms = latency if latency else p.latency_ms
                p.last_used = now
                p._score_cache = p._recompute_score()

    def report_failure(self, url: str) -> None:
        now = time.monotonic()
        with self._lock:
            p = self._proxies.get(url)
            if p:
                p.failures += 1; p.total_calls += 1; p.consec_failures += 1; p.last_fail = now
                p._score_cache = p._recompute_score()
                # Blacklist after 2 consecutive failures (down from 3) with exponential backoff
                if p.consec_failures >= 2:
                    cooldown = min(
                        self.COOLDOWN_BASE * (2 ** min(p.consec_failures - 2, 5)),
                        self.COOLDOWN_MAX,
                    )
                    p.blacklisted_until = now + cooldown
                    p.alive = False
                    log.info("Proxy degraded %s (%ss cooldown, consec=%d)", url, cooldown, p.consec_failures)
                # Immediately evict from hot pool — a failed proxy should not be
                # considered best until the next hot pool refresh.
                if url in self._hot_pool:
                    self._hot_pool.remove(url)

    def report_rate_limited(self, url: str) -> None:
        """Called when the upstream returns rate-limit signals via the circuit breaker."""
        now = time.monotonic()
        with self._lock:
            p = self._proxies.get(url)
            if p:
                p.consec_failures += 3
                p._score_cache = p._recompute_score()
                cooldown = min(self.COOLDOWN_BASE * (2 ** min(p.consec_failures - 2, 5)), self.COOLDOWN_MAX)
                p.blacklisted_until = now + cooldown
                p.alive = False
                log.warning("Proxy rate-limited %s (cooldown %ss)", url, cooldown)
                if url in self._hot_pool:
                    self._hot_pool.remove(url)

    def get_status(self) -> dict:
        with self._lock:
            now = time.monotonic()
            return {
                "proxies": [
                    {
                        "url": p.url,
                        "alive": p.alive,
                        "score": round(p.score, 1),
                        "successes": p.successes,
                        "failures": p.failures,
                        "calls": p.total_calls,
                        "latency_ms": round(p.latency_ms, 1),
                        "blacklisted_for": max(0, round(p.blacklisted_until - now, 1)) if p.blacklisted_until else 0,
                        "verified": p.verified_working,
                    }
                    for p in sorted(self._proxies.values(), key=lambda x: x.score, reverse=True)
                ]
            }

    # ------------------------------------------------------------------ #
    #  Reliability: Pool health summary for /health endpoint
    # ------------------------------------------------------------------ #

    def get_alive_count(self) -> int:
        """Return the number of currently usable (alive + not blacklisted) proxies."""
        with self._lock:
            now = time.monotonic()
            return sum(1 for p in self._proxies.values()
                       if p.alive and p.blacklisted_until <= now)

    def get_summary(self) -> dict[str, object]:
        """Return a lightweight summary of proxy pool health.

        Suitable for the /health endpoint — avoids the full proxy list.
        """
        with self._lock:
            now = time.monotonic()
            total = len(self._proxies)
            alive = sum(1 for p in self._proxies.values()
                        if p.alive and p.blacklisted_until <= now)
            blacklisted = sum(1 for p in self._proxies.values()
                              if not p.alive and p.blacklisted_until > now)
            active_calls = sum(1 for p in self._proxies.values()
                               if p.alive and p.blacklisted_until <= now
                               and p.success_count < p.max_successes)
            latencies = [p.latency_ms for p in self._proxies.values()
                         if p.latency_ms > 0]
            avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
            return {
                "total": total,
                "alive": alive,
                "blacklisted": blacklisted,
                "active_not_depleted": active_calls,
                "average_latency_ms": round(avg_latency, 1),
                "alive_ratio": round(alive / max(total, 1), 2),
            }


# Singleton
_pool: SmartProxyPool | None = None
_lock = threading.Lock()


def get_pool() -> SmartProxyPool:
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                _pool = SmartProxyPool()
    return _pool


# ------------------------------------------------------------------ #
#  Unified proxy fallback chain
# ------------------------------------------------------------------ #

class AllProxiesExhausted(RuntimeError):
    """All proxy candidates have been tried and all failed."""


# Shared executor path — removed atexit, uses lazy _get_proxy_executor() instead.


def _try_proxies(
    make_request,
    effective_timeout: float,
    pool: SmartProxyPool | None = None,
    use_unique_proxy: bool = False,
) -> object:
    """Unified proxy fallback chain with concurrent racing.

    Races all proxy candidates (pool proxies, env proxies, VPN tunnels, direct)
    concurrently via the shared ThreadPoolExecutor and returns the first successful
    result. Falls back to sequential pool retry if all concurrent attempts fail.

    Args:
        make_request: Callable(proxy_url, timeout) -> result. Must raise on failure.
        effective_timeout: Overall timeout cap.
        pool: Optional SmartProxyPool for pool-sourced proxies.
        use_unique_proxy: If True, use pool.get_unique() for fallback retries
                          (avoids per-IP rate limits for guest token creation).

    Returns:
        Whatever make_request returns on first success.

    Raises:
        AllProxiesExhausted: When every candidate has been tried and failed.
    """
    candidates: list[tuple[str | None, float]] = []  # (proxy_url, timeout)
    seen: set[str] = set()

    # 1. SOCKS5 proxy pool — up to 3 weighted-random proxies
    if pool is not None:
        for _ in range(3):
            p = pool.get_best()
            if p and p not in seen:
                seen.add(p)
                candidates.append((p, 10))

    # 2. Environment proxies (GLM_PROXY_LIST)
    env_list = os.environ.get("GLM_PROXY_LIST", "").strip()
    if env_list:
        for ep in (u.strip() for u in env_list.split(",") if u.strip()):
            if ep not in seen:
                seen.add(ep)
                candidates.append((ep, 10))

    # 3. Local VPN tunnels (127.0.0.1:7928-7932)
    for port in ("7928", "7929", "7930", "7931", "7932"):
        url = f"socks5://127.0.0.1:{port}"
        if url not in seen:
            seen.add(url)
            # Primary VPN tunnel (7928) gets a slightly longer timeout
            candidates.append((url, 8 if port == "7928" else 6))

    # 4. Direct connection — brief try
    if None not in seen:
        seen.add("__direct__")
        candidates.append((None, 3))

    # Race all candidates concurrently via the shared executor
    max_race = min(effective_timeout, 12)
    future_map = {_get_proxy_executor().submit(make_request, p, t): (p, t) for p, t in candidates}
    done, not_done = wait(future_map, timeout=max_race, return_when=FIRST_COMPLETED)

    for f in done:
        try:
            result = f.result()
            # Cancel any remaining in-flight requests (best-effort)
            for nf in not_done:
                nf.cancel()
            log.debug("Proxy racing succeeded (first completed)")
            return result
        except Exception:
            continue

    # 5. Fallback: sequential pool retry with full timeout
    if pool is not None:
        for attempt in range(2):
            proxy_url = pool.get_unique() if use_unique_proxy else pool.get_best()
            if not proxy_url:
                break
            try:
                result = make_request(proxy_url, effective_timeout)
                log.debug("Pool fallback succeeded (attempt %d): %s", attempt + 1, proxy_url)
                return result
            except Exception:
                pool.report_failure(proxy_url)
                continue

    # 6. Emergency direct connection — last resort when pool is completely starved.
    # Uses the full effective_timeout (not the brief 3s from the racing phase)
    # since this is the final attempt before failure.
    try:
        log.warning("紧急模式: 所有代理耗尽，尝试直接连接 (timeout=%.0fs)", effective_timeout)
        result = make_request(None, effective_timeout)
        log.warning("紧急模式: 直接连接成功 — 所有代理已恢复?")
        return result
    except Exception:
        log.warning("紧急模式: 直接连接也失败")

    raise AllProxiesExhausted("All proxies and VPN tunnels exhausted")
