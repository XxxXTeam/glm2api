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

import os
import random
import time
import threading
import logging
from dataclasses import dataclass, field

log = logging.getLogger("glm2api.proxy")

# ponytail: 12 working public SOCKS5 proxy list URLs.
# Includes country-specific proxyscrape endpoints for China + nearby Asia.
# Combined unique total: ~7,500+ proxies.
_PUBLIC_SOCKS5_URLS = (
    "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/socks5.txt",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=all",
    "https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt",
    "https://api.openproxylist.xyz/socks5.txt",
    "https://raw.githubusercontent.com/ShiftyTR/Proxy-List/master/socks5.txt",
    "https://raw.githubusercontent.com/roosterkid/openproxylist/main/SOCKS5_RAW.txt",
    "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-socks5.txt",
    "https://raw.githubusercontent.com/mmpx12/proxy-list/master/socks5.txt",
    # Country-specific proxyscrape — these exit in China/HK/JP/SG/KR
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=CN",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=HK",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=JP",
    "https://api.proxyscrape.com/v2/?request=getproxies&protocol=socks5&timeout=10000&country=SG",
)


@dataclass
class ProxyScore:
    url: str
    alive: bool = True
    successes: int = 0
    failures: int = 0
    consec_failures: int = 0
    total_calls: int = 0
    latency_ms: float = 0.0
    _score_cache: float = 100.0  # ponytail: cached score, recomputed on mutation
    last_used: float = 0.0
    def __post_init__(self) -> None:
        self._score_cache = self._recompute_score()
    last_fail: float = 0.0
    blacklisted_until: float = 0.0  # timestamp when it can be retried
    verified_working: bool = False   # passed SOCKS5 handshake test

    def _recompute_score(self) -> float:
        """Reliability score: higher is better. Based on success rate + latency."""
        if self.total_calls == 0:
            return 100.0
        rate = self.successes / max(self.total_calls, 1)
        lat_penalty = min(self.latency_ms / 1000, 10)
        fail_penalty = self.consec_failures * 5
        return (rate * 50) - lat_penalty - fail_penalty

    @property
    def score(self) -> float:
        """Cached reliability score. Invalidated on each mutation."""
        return self._score_cache


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
    _REFRESH_LOCK = threading.Lock()  # prevent concurrent refresh storms

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proxies: dict[str, ProxyScore] = {}
        self._current = ""
        self._last_health_check = 0.0
        self._last_refresh = 0.0
        self._populate()
        self._start_health_checks()

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
    def _verify_socks5(proxy_url: str, target_host: str = "chatglm.cn", target_port: int = 443, timeout: float = 3) -> float | None:
        """Test SOCKS5 proxy with full HTTP guest token request.
        ponytail: does SOCKS5 handshake + TLS 1.2 + HTTP POST with proper JS-matched headers.
        Returns latency in ms on success, None on failure. This ensures only proxies
        that can actually route GLM API traffic are accepted."""
        import socket as _sck, hashlib, uuid, ssl as _ssl
        raw = proxy_url.replace("socks5://", "").replace("http://", "").replace("https://", "")
        host, port_str = raw.split(":") if ":" in raw else (raw, "1080")
        port = int(port_str)
        try:
            # SOCKS5 handshake
            start = time.monotonic()
            s = _sck.create_connection((host, port), timeout=timeout)
            s.sendall(b"\x05\x01\x00")
            if s.recv(2) != b"\x05\x00": s.close(); return None
            hostname = target_host.encode()
            s.sendall(b"\x05\x01\x00\x03" + bytes([len(hostname)]) + hostname + target_port.to_bytes(2, "big"))
            resp = s.recv(10)
            if len(resp) < 2 or resp[1] != 0x00: s.close(); return None

            # TLS 1.2 through tunnel (WAF blocks TLS 1.3)
            ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
            ctx.minimum_version = _ssl.TLSVersion.TLSv1_2
            ctx.maximum_version = _ssl.TLSVersion.TLSv1_2
            ctx.check_hostname = False; ctx.verify_mode = _ssl.CERT_NONE
            ss = ctx.wrap_socket(s, server_hostname=target_host)

            # Guest token POST with exact JS-matched headers
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
                f"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0\r\n"
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

    def _verify_and_filter(self, urls: list[str], min_working: int = 200, max_to_check: int = 1500) -> list[str]:
        """Verify proxies against chatglm.cn with full HTTP test, return only working ones.
        ponytail: checks first max_to_check proxies concurrently, cancels remaining once target hit.
        Uses full SOCKS5 + TLS 1.2 + HTTP guest token request for verification.
        ~1500 × 5s / 50 workers = ~150s worst case, typically ~60s."""
        import concurrent.futures as _cf
        check = urls[:max_to_check]
        log.info("Verifying %d proxies against chatglm.cn (need %d)...", len(check), min_working)
        working: list[tuple[str, float]] = []

        with _cf.ThreadPoolExecutor(max_workers=50) as pool:
            fut_map = {pool.submit(self._verify_socks5, u): u for u in check}
            for f in _cf.as_completed(fut_map):
                u = fut_map[f]
                try:
                    latency = f.result(timeout=8)  # HTTP test needs more time
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
        log.info("Verification done: %d/%d proxies work against chatglm.cn", len(result), len(check))
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
                # Only refresh if >50% dead AND pool has meaningful size
                if total > 50 and alive / total > 0.5:
                    return
                log.warning("Proxy pool degraded: %d/%d alive. Auto-refreshing...", alive, total)

            # Fetch and verify new proxies (outside lock to avoid blocking)
            fetched = self._fetch_public_socks5()
            verified = self._verify_and_filter(fetched, min_working=200) if fetched else []
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

    def _populate(self) -> None:
        urls: list[str] = []
        lst = os.environ.get("GLM_PROXY_LIST", "").strip()
        if lst:
            urls.extend(u.strip() for u in lst.split(",") if u.strip())
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
            self._proxies[u] = ProxyScore(url=u)
        # GLM_PROXY_AUTO_FETCH=true: always scrape + verify against chatglm.cn,
        # supplementing env-sourced proxies with fresh public ones
        if os.environ.get("GLM_PROXY_AUTO_FETCH", "").strip().lower() in ("true", "1", "yes"):
            fetched = self._fetch_public_socks5()
            verified = self._verify_and_filter(fetched, min_working=200)
            for u in verified:
                if u not in self._proxies:
                    self._proxies[u] = ProxyScore(url=u, verified_working=True)
            if verified:
                log.info("Auto-fetch complete: %d verified SOCKS5 proxies (pool total: %d)", len(verified), len(self._proxies))
        self._current = next(iter(self._proxies)) if self._proxies else ""

    def _start_health_checks(self) -> None:
        def _loop():
            while True:
                time.sleep(self.HEALTH_CHECK_INTERVAL)
                self._check_all()
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
            try:
                url = p.url.replace("socks5://", "").replace("http://", "").replace("https://", "")
                host, port = url.split(":") if ":" in url else (url, "1080")
                start = time.monotonic()
                s = _sck.create_connection((host, int(port)), timeout=3)
                s.close()
                p.latency_ms = (time.monotonic() - start) * 1000
                p.alive = True
                p.consec_failures = 0
                p._score_cache = p._recompute_score()
            except Exception:
                p.consec_failures += 1
                p._score_cache = p._recompute_score()
                if p.consec_failures >= 2:
                    p.alive = False
                    p.blacklisted_until = time.monotonic() + min(
                        self.COOLDOWN_BASE * (2 ** min(p.consec_failures, 5)),
                        self.COOLDOWN_MAX
                    )

        with _cf.ThreadPoolExecutor(max_workers=50) as pool:
            list(pool.map(_check_one, proxies))

    def get_best(self) -> str | None:
        """Return a proxy URL using weighted random selection from top N candidates.
        ponytail: distributes load across good proxies to avoid thundering herd."""
        import random as _rnd
        now = time.monotonic()
        TOP_N = 15  # Select from top 15 proxies by score for better distribution
        with self._lock:
            candidates: list[ProxyScore] = []
            soonest: ProxyScore | None = None
            for p in self._proxies.values():
                # Revive expired cooldowns
                if p.blacklisted_until and now >= p.blacklisted_until and not p.alive:
                    p.alive = True
                    p.blacklisted_until = 0
                    p.consec_failures = 0
                # Track alive candidates
                if p.alive and p.blacklisted_until <= now:
                    candidates.append(p)
                # Track soonest to recover for fallback
                if not p.alive and p.blacklisted_until > now:
                    if soonest is None or p.blacklisted_until < soonest.blacklisted_until:
                        soonest = p

            if candidates:
                # Sort by score descending and pick from top N with weighted random
                candidates.sort(key=lambda p: p.score, reverse=True)
                top = candidates[:TOP_N]
                # Weighted random: score-based probability
                total_score = sum(p.score for p in top)
                if total_score > 0:
                    r = _rnd.uniform(0, total_score)
                    cumulative = 0.0
                    for p in top:
                        cumulative += p.score
                        if r <= cumulative:
                            self._current = p.url
                            return p.url
                # Fallback: uniform random from top
                chosen = _rnd.choice(top)
                self._current = chosen.url
                return chosen.url

            # All blacklisted — trigger background auto-refresh
            if len(self._proxies) > 50 and now - self._last_refresh > 300:
                threading.Thread(target=self._auto_refresh, daemon=True).start()

            # retry the soonest proxy immediately if it's been down for a while
            if soonest is not None and now - soonest.last_fail > 30:
                try:
                    import socket as _sck
                    host, port = soonest.url.replace("http://","").replace("socks5://","").replace("https://","").split(":")
                    _sck.create_connection((host, int(port)), timeout=2).close()
                    # Proxy is back up
                    soonest.alive = True
                    soonest.blacklisted_until = 0
                    soonest.consec_failures = 0
                    soonest._score_cache = soonest._recompute_score()
                    self._current = soonest.url
                    log.info("Proxy back online (immediate retry): %s", soonest.url)
                    return soonest.url
                except Exception:
                    pass

            if soonest is not None:
                log.warning("All proxies blacklisted, retrying soonest: %s (remaining %ss)", 
                           soonest.url, max(0, soonest.blacklisted_until - now))
                self._current = soonest.url
                return soonest.url

            return None

    def get_unique(self) -> str | None:
        """Return a proxy URL, rotating through all available proxies.
        Used for guest token creation to avoid per-IP rate limiting."""
        import random as _rnd
        now = time.monotonic()
        with self._lock:
            candidates: list[ProxyScore] = []
            for p in self._proxies.values():
                if p.blacklisted_until and now >= p.blacklisted_until and not p.alive:
                    p.alive = True
                    p.blacklisted_until = 0
                    p.consec_failures = 0
                if p.alive and p.blacklisted_until <= now:
                    candidates.append(p)
            
            if candidates:
                # Round-robin through all candidates for guest token creation
                if not hasattr(self, '_guest_rr_index'):
                    self._guest_rr_index = 0
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

            candidates = [p for p in self._proxies.values() if p.alive and p.blacklisted_until <= now]
            if not candidates:
                # No alive proxies — try any dead ones immediately
                now = time.monotonic()
                for p in self._proxies.values():
                    if now - p.last_fail > 15:  # Only retry if it's been >15s since last fail
                        try:
                            host, port = p.url.replace("http://","").replace("socks5://","").replace("https://","").split(":")
                            _sck = __import__('socket')
                            _sck.create_connection((host, int(port)), timeout=2).close()
                            p.alive = True
                            p.blacklisted_until = 0
                            p.consec_failures = 0
                            self._current = p.url
                            log.info("Proxy back online: %s", p.url)
                            return p.url
                        except Exception:
                            p.last_fail = now
                            pass
                # All blacklisted — trigger background auto-refresh
                if len(self._proxies) > 50 and now - self._last_refresh > 300:
                    threading.Thread(target=self._auto_refresh, daemon=True).start()
                return None

            # Sort by total_calls ascending so least-used proxies are picked first
            candidates.sort(key=lambda p: p.total_calls)

            # Round-robin through sorted candidates
            if not hasattr(self, '_main_rr_index'):
                self._main_rr_index = 0
            self._main_rr_index %= len(candidates)
            chosen = candidates[self._main_rr_index]
            self._main_rr_index = (self._main_rr_index + 1) % len(candidates)
            self._current = chosen.url
            return chosen.url

    def report_success(self, url: str, latency: float = 0) -> None:
        now = time.monotonic()
        with self._lock:
            p = self._proxies.get(url)
            if p:
                p.successes += 1
                p.total_calls += 1
                p.consec_failures = 0
                p.latency_ms = latency if latency else p.latency_ms
                p.last_used = now
                p.alive = True
                p._score_cache = p._recompute_score()

    def report_failure(self, url: str) -> None:
        now = time.monotonic()
        with self._lock:
            p = self._proxies.get(url)
            if p:
                p.failures += 1; p.total_calls += 1; p.consec_failures += 1; p.last_fail = now
                p._score_cache = p._recompute_score()
                if p.consec_failures >= 3:
                    cooldown = min(15 * p.consec_failures, 60)
                    p.blacklisted_until = now + cooldown
                    p.alive = False
                    log.info("Proxy degraded %s (%ss cooldown)", url, cooldown)

    def report_rate_limited(self, url: str) -> None:
        """Called when the upstream returns rate-limit signals via the circuit breaker."""
        now = time.monotonic()
        with self._lock:
            p = self._proxies.get(url)
            if p:
                p.consec_failures += 3
                p._score_cache = p._recompute_score()
                cooldown = min(self.COOLDOWN_BASE * (2 ** min(p.consec_failures, 5)), self.COOLDOWN_MAX)
                p.blacklisted_until = now + cooldown
                p.alive = False
                log.warning("Proxy rate-limited %s (cooldown %ss)", url, cooldown)

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
