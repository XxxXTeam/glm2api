from __future__ import annotations

import hashlib
import http.client
import orjson
import os
import random
import threading
import time
import uuid
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from logging import Logger

from ..config import AppConfig, GUEST_REFRESH_TOKEN_MARKER
from ..logging_utils import debug_dump


class UpstreamBlockedError(RuntimeError):
    """Upstream is blocked/unreachable — circuit breaker opened."""


from enum import Enum

class ErrorClass(Enum):
    """Error classification for reliability engineering and metrics."""
    TRANSIENT = "transient"        # Retryable: timeout, connection reset, EOF
    PERMANENT = "permanent"        # Non-retryable: bad request, invalid params
    UPSTREAM = "upstream"          # Upstream error: 5xx, WAF block, service unavailable
    PROXY = "proxy"                # Proxy failure: connection refused, DNS, SSL handshake
    AUTH = "auth"                  # Auth failure: token expired, invalid/expired token
    QUOTA = "quota"                # Quota exhausted: rate limited, empty response, 429
    UNKNOWN = "unknown"            # Unclassifiable — treat as transient


def classify_error(exc: Exception, status_code: int | None = None) -> ErrorClass:
    """Classify an exception into an ErrorClass for metrics and circuit-breaking decisions."""
    if status_code is not None:
        if status_code == 429:
            return ErrorClass.QUOTA
        if status_code in (401, 403):
            return ErrorClass.AUTH
        if status_code in (502, 503, 504):
            return ErrorClass.UPSTREAM
        if status_code in (400, 422):
            return ErrorClass.PERMANENT
    if isinstance(exc, socket.timeout):
        return ErrorClass.TRANSIENT
    if isinstance(exc, TimeoutError):
        return ErrorClass.TRANSIENT
    if isinstance(exc, ConnectionError):
        return ErrorClass.TRANSIENT
    if isinstance(exc, http.client.IncompleteRead):
        return ErrorClass.TRANSIENT
    if isinstance(exc, UpstreamBlockedError):
        return ErrorClass.UPSTREAM
    err_str = str(exc).lower()
    if any(kw in err_str for kw in ("token", "unauthorized", "authentication", "unauth")):
        return ErrorClass.AUTH
    if any(kw in err_str for kw in ("rate limit", "too many", "quota", "exhausted", "多次体验", "请登录")):
        return ErrorClass.QUOTA
    if any(kw in err_str for kw in ("proxy", "socks5", "connection refused", "dns", "no route to host")):
        return ErrorClass.PROXY
    if any(kw in err_str for kw in ("timeout", "reset", "broken pipe", "eof", "incomplete read")):
        return ErrorClass.TRANSIENT
    if any(kw in err_str for kw in ("ssl", "certificate", "handshake", "tls")):
        return ErrorClass.PROXY
    return ErrorClass.UNKNOWN


SIGN_SECRET = "8a1317a7468aa3ad86e997d08f3f31cb"
_SIGN_SECRET_BYTES = SIGN_SECRET.encode("utf-8")
ACCESS_TOKEN_EXPIRES_SECONDS = 3600


def build_sign() -> tuple[str, str, str]:
    now = str(time.time_ns() // 1_000_000)
    digits = [int(char) for char in now]
    checksum = (sum(digits) - digits[-2]) % 10
    timestamp = now[:-2] + str(checksum) + now[-1]
    nonce = os.urandom(8).hex()
    sign = hashlib.md5(f"{timestamp}-{nonce}-".encode("utf-8") + _SIGN_SECRET_BYTES).hexdigest()
    return timestamp, nonce, sign


def _random_ip() -> str:
    """Generate a single random non-internal IP address."""
    while True:
        first = random.randint(1, 223)
        if first not in {10, 127, 169, 172, 192}:
            break
    return f"{first}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"


# Number of pre-built header variants per cache key; higher = better IP diversity
_HEADER_POOL_SIZE = 128

# Shared ThreadPoolExecutor for background prewarm
_PREWARM_EXECUTOR = None
_PREWARM_EXECUTOR_LOCK = threading.Lock()

def _get_prewarm_executor():
    global _PREWARM_EXECUTOR
    if _PREWARM_EXECUTOR is None or _PREWARM_EXECUTOR._shutdown:
        with _PREWARM_EXECUTOR_LOCK:
            if _PREWARM_EXECUTOR is None or _PREWARM_EXECUTOR._shutdown:
                from concurrent.futures import ThreadPoolExecutor
                _PREWARM_EXECUTOR = ThreadPoolExecutor(max_workers=5, thread_name_prefix="prewarm")
    return _PREWARM_EXECUTOR


class _HeaderPool:
    """Thread-safe rotating pool of pre-built header dicts.

    Each entry has a different X-Forwarded-For IP already embedded,
    eliminating the per-call dict copy + IP insertion.
    """
    __slots__ = ('_headers', '_index', '_lock')

    def __init__(self, headers: list[dict[str, str]]) -> None:
        self._headers = headers
        self._index = 0
        self._lock = threading.Lock()

    def get(self) -> dict[str, str]:
        """Return the next header dict from the rotating pool."""
        with self._lock:
            h = self._headers[self._index]
            self._index = (self._index + 1) % len(self._headers)
            return h


@dataclass(slots=True)
class AccessToken:
    access_token: str
    refresh_token: str
    expires_at: float


@dataclass(slots=True)
class AccountState:
    refresh_token: str
    is_guest: bool = False
    cached_token: AccessToken | None = None
    tool_call_count: int = 0
    search_count: int = 0
    # Ponytail: EWMA latency tracking for gradient failover
    ewma_latency: float = 0.0  # ms, sliding average
    ewma_alpha: float = 0.125  # weight for new samples
    _device_id: str | None = None  # persistent device ID (matches JS localStorage behavior)


class GLMAccessTokenManager:
    def __init__(self, config: AppConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self._accounts: list[AccountState] = []
        self._current_index = 0
        self._lock = threading.RLock()  # RLock for reentrancy: get_fill_first_account is called while lock held
        self._persist_lock = threading.Lock()
        self._round_robin_counter = 0
        self._rate_limited_accounts = {}
        self._rate_limit_cooldown = float(__import__("os").environ.get("GLM_GUEST_COOLDOWN_SECONDS", "3"))
        self._last_guest_fetch = 0.0
        self._guest_fetch_interval = float(__import__("os").environ.get("GLM_GUEST_FETCH_INTERVAL", "3.0"))
        self._consecutive_failures = 0
        self._starvation_threshold = int(__import__("os").environ.get("GLM_STARVATION_THRESHOLD", "6"))
        self._silent_accounts = {}
        self._silence_cooldown = float(__import__("os").environ.get("GLM_SILENCE_COOLDOWN_SECONDS", "120"))
        self._last_full_refresh_at = 0.0
        # ponytail: circuit breaker — upstream failure protection
        self._circuit_open = False
        self._circuit_opened_at = 0.0
        self._circuit_failures = 0
        self._circuit_threshold = int(__import__("os").environ.get("GLM_CIRCUIT_BREAKER_THRESHOLD", "6"))
        self._circuit_cooldown = float(__import__("os").environ.get("GLM_CIRCUIT_COOLDOWN_SECONDS", "30"))
        self._full_refresh_cooldown = float(__import__("os").environ.get("GLM_FULL_REFRESH_COOLDOWN_SECONDS", "300"))
        self._account_fail_count: dict[int, int] = {}
        self._account_fail_threshold = 3  # after 3 consecutive fails, circuit-break
        # ponytail: fill-first counter for GLM_ACCOUNT_STRATEGY=fill_first
        self._fill_first_index = 0
        # Reliability: circuit breaker half-open probe state
        self._circuit_probe_at: float = 0.0
        self._circuit_probe_failures: int = 0
        self._circuit_probe_in_progress: bool = False
        # Reliability: last successful request timestamp (monotonic) for watchdog
        self._last_success_at: float = 0.0
        # Reliability: per-account progressive backoff cooldowns
        self._rate_limit_cooldowns: dict[int, float] = {}
        # ponytail: warm spare pool — always have fresh accounts ready for instant swap
        self._warm_spares: list[AccountState] = []
        self._warm_spare_target = 3
        self._init_guest_pool(config.glm_refresh_tokens)
        self._start_sliding_refresh()
        self._warm_connection()
        self._start_heartbeat()
        self._start_warm_spare_refiller()
        self._start_watchdog()

    def _warm_connection(self) -> None:
        """Warm TLS session + WAF cookie by triggering get_client() which auto-warms.
        The cached Session in http_client is now auto-warmed on creation so no separate
        warmup is needed. This just triggers the lazy Session creation."""
        from .http_client import get_client
        try:
            client = get_client(proxy_url=None, logger=self.logger)
            if client:
                self.logger.info("Connection warmed (TLS session + WAF cookie established)")
        except Exception:
            pass  # non-fatal

    def _start_heartbeat(self) -> None:
        """Background thread: sends GET to chatglm.cn every 30s to keep TLS +
        WAF cookie alive. Prevents the intermittent TLS block from the WAF by
        maintaining a persistent warm connection."""
        if not self.config.glm_use_guest_refresh_token:
            return
        def _beat():
            while True:
                time.sleep(30)
                try:
                    from .http_client import do_request_oneshot
                    do_request_oneshot("GET", "https://chatglm.cn/",
                        headers={"User-Agent": self.config.glm_user_agent,
                                 "Accept": "text/html", "Accept-Language": "zh-CN,zh;q=0.9",
                                 "Origin": "https://chatglm.cn", "Referer": "https://chatglm.cn/"},
                        timeout=5, logger=self.logger)
                except Exception:
                    pass  # WAF blocking this cycle — next heartbeat will retry
        threading.Thread(target=_beat, daemon=True).start()
        self.logger.info("Heartbeat started (30s interval)")

    def record_upstream_failure(self) -> None:
        with self._lock:
            if self._circuit_probe_in_progress:
                # Probe failure — re-open the circuit immediately with fresh cooldown
                self._circuit_probe_in_progress = False
                self._circuit_opened_at = time.monotonic()
                self._circuit_failures = self._circuit_threshold  # keep circuit open
                self.logger.warning("断路器探测失败 — 重新开放电路，冷却 %ss", self._circuit_cooldown)
                return
            self._circuit_failures += 1
            self.logger.warning("上游失败 #%s/%s", self._circuit_failures, self._circuit_threshold)
            try:
                from .glm2api_proxy import get_pool as _get_pool
                _pool = _get_pool()
                if _pool._current:
                    _pool.report_rate_limited(_pool._current)
            except Exception:
                pass
            if self._circuit_failures >= self._circuit_threshold:
                self._circuit_open = True
                self._circuit_opened_at = time.monotonic()
                self.logger.warning("断路器打开 — %s 次连续失败，暂停 %ss", self._circuit_failures, self._circuit_cooldown)

    def record_upstream_success(self) -> None:
        with self._lock:
            if self._circuit_probe_in_progress:
                # Probe success — close the circuit fully
                self._circuit_probe_in_progress = False
                self._circuit_open = False
                self._circuit_failures = 0
                self.logger.info("断路器探测成功 — 关闭电路，上游已恢复")
            else:
                if self._circuit_failures > 0:
                    self._circuit_failures = 0
                    self.logger.info("上游恢复 — 重置失败计数")
                if self._circuit_open:
                    self._circuit_open = False
                    self.logger.info("断路器关闭 — 上游已恢复")

    def _do_upstream_request(self, request: urllib.request.Request, use_unique_proxy: bool = False) -> tuple[int, dict]:
        """Execute upstream request via unified HTTP client, delegating proxy
        fallback to the shared _try_proxies() chain in glm2api_proxy.

        The unified function races pool proxies, env proxies, VPN tunnels, and
        direct connection concurrently, then falls back to sequential pool retry.
        """
        from .glm2api_proxy import get_pool, _try_proxies, AllProxiesExhausted
        from .http_client import do_json_request

        method = request.get_method() if hasattr(request, 'get_method') else 'POST'
        url = str(request.full_url) if hasattr(request, 'full_url') else str(request)
        headers = dict(request.header_items())
        data = request.data if hasattr(request, "data") and request.data is not None else None
        effective_timeout = self.config.request_timeout

        pool = get_pool()

        def _make_request(proxy_url, timeout):
            status_code, payload = do_json_request(
                method=method, url=url, headers=headers, data=data,
                proxy_url=proxy_url, timeout=timeout, logger=self.logger,
            )
            if proxy_url:
                pool.report_success(proxy_url, 0)
            return (status_code, payload)

        try:
            return _try_proxies(_make_request, effective_timeout, pool, use_unique_proxy=use_unique_proxy)
        except AllProxiesExhausted:
            raise urllib.error.URLError("Cannot reach upstream: all proxies and VPN tunnels exhausted")

    @staticmethod
    def _is_jwt_guest_token(token: str) -> bool:
        """Check if a JWT token is a guest token by inspecting the payload."""
        if token == GUEST_REFRESH_TOKEN_MARKER:
            return True
        try:
            import base64 as _b64
            parts = token.strip().split(".")
            if len(parts) == 3:
                payload = parts[1]
                # Add padding for base64 decode
                payload += "=" * (4 - len(payload) % 4)
                decoded = _b64.urlsafe_b64decode(payload)
                data = orjson.loads(decoded)
                return data.get("is_guest") is True
        except Exception:
            pass
        return False

    def _init_guest_pool(self, raw_tokens: list[str]) -> None:
        """Initialize account pool.
        Detects guest mode via:
          1. All tokens are GUEST_REFRESH_TOKEN_MARKER (explicit guest mode)
          2. All tokens are guest JWT tokens (is_guest: true in payload)
        When pre-created guest tokens are provided, preserves them in AccountState
        so each slot starts with a unique guest session (from a different proxy IP)."""
        # Detect guest mode
        if raw_tokens:
            is_guest = all(t == GUEST_REFRESH_TOKEN_MARKER or self._is_jwt_guest_token(t) for t in raw_tokens)
        else:
            is_guest = True

        if is_guest:
            pool_size = max(3, self.config.glm_max_concurrency)
            if raw_tokens and not all(t == GUEST_REFRESH_TOKEN_MARKER for t in raw_tokens):
                # Pre-created guest tokens — spin up background pre-warm so server
                # starts immediately while tokens warm up in the background
                self._accounts = []
                for i in range(pool_size):
                    if i < len(raw_tokens) and raw_tokens[i] != GUEST_REFRESH_TOKEN_MARKER:
                        self._accounts.append(AccountState(refresh_token=raw_tokens[i], is_guest=True))
                    else:
                        self._accounts.append(AccountState(refresh_token="", is_guest=True))
                pre_created = sum(1 for a in self._accounts if a.refresh_token)
                self.logger.info(
                    "游客槽位初始化 数量=%s (其中 %s 个预创建 token) — 后台预刷新开始",
                    pool_size, pre_created
                )
                # Background pre-warm: staggered 3s intervals to avoid per-IP rate limits
                threading.Thread(target=self._background_prewarm, daemon=True).start()
            else:
                # No pre-created tokens — create empty guest slots (lazy fetch)
                self._accounts = [AccountState(refresh_token="", is_guest=True) for _ in range(pool_size)]
                self.logger.info(
                    "游客槽位初始化 数量=%s (延迟获取) — 后台预刷新开始", pool_size
                )
                # Background pre-warm all guest slots
                threading.Thread(target=self._background_prewarm, daemon=True).start()
        else:
            self._accounts = [
                AccountState(refresh_token=t, is_guest=False) for t in raw_tokens
            ]
            self.logger.info(
                "账号池初始化 数量=%s", len(self._accounts)
            )

    def _background_prewarm(self) -> None:
        """Background thread: pre-warms up to 5 guest access tokens.
        ponytail: limited to conserve quota for real requests."""
        from concurrent.futures import as_completed
        with self._lock:
            remaining = [i for i, a in enumerate(self._accounts)
                         if a.is_guest and a.cached_token is None][:5]
        if not remaining:
            return
        executor = _get_prewarm_executor()
        futures = {executor.submit(self._fetch_guest_access_token, i, True): i for i in remaining}
        for f in as_completed(futures):
            idx = futures[f]
            try:
                token = f.result()
                if token:
                    with self._lock:
                        self._accounts[idx].cached_token = token
                    self.logger.info("后台预刷新 access_token 成功 index=%s", idx)
            except Exception as exc:
                self.logger.warning("后台预刷新 access_token 失败 index=%s error=%s", idx, exc)
        with self._lock:
            active = sum(1 for a in self._accounts if a.cached_token is not None)
        self.logger.info("后台预刷新完成 active=%s/%s", active, len(self._accounts))

    def _start_warm_spare_refiller(self) -> None:
        """Background thread: keeps warm spare pool filled with fresh accounts.
        ponytail: spares are immediately ready for swap when an account exhausts.
        Sleeps 30s between checks since the pool is full the vast majority of the time;
        _refill_warm_spares returns immediately when above or at target."""
        def _loop():
            import time as _t
            while True:
                _t.sleep(30)
                try:
                    self._refill_warm_spares()
                except Exception:
                    pass
        threading.Thread(target=_loop, daemon=True).start()

    def _refill_warm_spares(self) -> None:
        """Create warm spare accounts if below target. Each has fresh independent quota.
        ponytail: fetches tokens directly without going through account pool."""
        from .http_client import do_json_request
        import hashlib as _hl, uuid as _uid, time as _t
        with self._lock:
            needed = self._warm_spare_target - len(self._warm_spares)
        if needed <= 0:
            return
        for _ in range(needed):
            try:
                ts = str(int(_t.time() * 1000))
                digits = [int(c) for c in ts]
                checksum = (sum(digits) - digits[-2]) % 10
                ts2 = ts[:-2] + str(checksum) + ts[-1]
                nonce = _uid.uuid4().hex
                sign = _hl.md5(f"{ts2}-{nonce}-8a1317a7468aa3ad86e997d08f3f31cb".encode()).hexdigest()
                headers = {
                    "Content-Type": "application/json;charset=utf-8", "Content-Length": "0",
                    "App-Name": "chatglm", "Origin": "https://chatglm.cn", "Referer": "https://chatglm.cn/",
                    "User-Agent": self.config.glm_user_agent,
                    "X-App-Fr": "browser_extension", "X-App-Platform": "pc", "X-App-Version": "1.0.83",
                    "X-Device-Id": _uid.uuid4().hex, "X-Lang": "en", "X-Nonce": nonce,
                    "X-Request-Id": _uid.uuid4().hex, "X-Sign": sign, "X-Timestamp": ts2,
                }
                # ponytail: try through proxy pool first, direct fallback
                _pool_proxy = None
                try:
                    from .glm2api_proxy import get_pool as _get_pool
                    _p = _get_pool()
                    _pool_proxy = _p.get_best() if _p._proxies else None
                except Exception:
                    pass
                status, payload = do_json_request(
                    "POST", self.config.guest_refresh_url, headers, data=b"",
                    proxy_url=_pool_proxy, timeout=20, logger=self.logger,
                )
                result = payload.get("result") or {}
                access_token = result.get("access_token")
                refresh_token = result.get("refresh_token")
                if status == 200 and access_token and refresh_token:
                    spare = AccountState(refresh_token=refresh_token, is_guest=True)
                    spare.cached_token = AccessToken(
                        access_token=access_token, refresh_token=refresh_token,
                        expires_at=_t.time() + 3600 - __import__("random").randint(10, 30),
                    )
                    with self._lock:
                        self._warm_spares.append(spare)
                    self.logger.info("温暖备用账号已准备 (pool=%s)", len(self._warm_spares))
                else:
                    self.logger.warning("温暖备用: 获取token失败 status=%s", status)
            except Exception as exc:
                self.logger.warning("温暖备用账号创建失败: %s", exc)
                break

    def get_warm_spare(self) -> AccountState | None:
        """Pop a pre-warmed spare account. Returns None if pool empty."""
        with self._lock:
            if self._warm_spares:
                spare = self._warm_spares.pop(0)
                self.logger.info("使用温暖备用账号 (remaining=%s)", len(self._warm_spares))
                return spare
        return None

    def spawn_fresh_guest_account(self) -> int:
        """Dynamically create a new guest account with fresh independent quota pool.
    
        Each guest token gets its own ~5 search / ~8 tool call quota.
        Returns the account index.
        """
        fresh = AccountState(refresh_token="", is_guest=True)
        with self._lock:
            self._accounts.append(fresh)
            idx = len(self._accounts) - 1
    
        self.logger.info("生成新游客账号 index=%s total=%s", idx, len(self._accounts))
        try:
            fresh.cached_token = self._fetch_guest_access_token(idx)
            self.logger.info("新游客账号就绪 index=%s", idx)
        except Exception as exc:
            self.logger.warning("新游客账号获取失败 index=%s error=%s", idx, exc)
            with self._lock:
                if idx < len(self._accounts) and self._accounts[idx].cached_token is None:
                    self._accounts.pop(idx)
            raise
        return idx
    
    def spare_guest_slot_available(self) -> bool:
        """Check if any lazy slot hasn't fetched its token yet."""
        with self._lock:
            return any(a.is_guest and a.cached_token is None for a in self._accounts)

    def record_tool_call(self, account_index: int) -> None:
        """Increment tool call usage counter for an account."""
        with self._lock:
            if 0 <= account_index < len(self._accounts):
                self._accounts[account_index].tool_call_count += 1

    def record_search(self, account_index: int) -> None:
        """Increment search usage counter for an account."""
        with self._lock:
            if 0 <= account_index < len(self._accounts):
                self._accounts[account_index].search_count += 1

    def get_usage(self, account_index: int) -> tuple[int, int]:
        """Return (tool_calls_used, searches_used) for an account."""
        with self._lock:
            if 0 <= account_index < len(self._accounts):
                a = self._accounts[account_index]
                return a.tool_call_count, a.search_count
            return 0, 0

    def is_near_quota_limit(self, account_index: int) -> bool:
        """Check if account is approaching its quota limits.
        Returns True if tool calls >= 6 or searches >= 4 (out of ~8 tool / ~5 search)."""
        with self._lock:
            if 0 <= account_index < len(self._accounts) and self._accounts[account_index].is_guest:
                a = self._accounts[account_index]
                return a.tool_call_count >= 6 or a.search_count >= 4
            return False

    def prewarm_next_guest_slot(self) -> int | None:
        """If a lazy slot is available, pre-fetch its token so it's ready instantly."""
        with self._lock:
            for idx, a in enumerate(self._accounts):
                if a.is_guest and a.cached_token is None:
                    pass
                else:
                    continue
                break
            else:
                return None
        try:
            token = self._fetch_guest_access_token(idx)
            with self._lock:
                if idx < len(self._accounts):
                    self._accounts[idx].cached_token = token
            self.logger.info("提前预热游客账号 index=%s", idx)
            return idx
        except Exception as exc:
            self.logger.warning("预热游客账号失败 index=%s error=%s", idx, exc)
            return None
    

    # Cache of pre-built header pools keyed by (app_fr, user_agent, lang)
    _STATIC_HEADERS_CACHE: dict[str, _HeaderPool] = {}

    def get_browser_headers(self, app_fr: str = "browser_extension") -> dict[str, str]:
        """Return browser-like headers with a randomized X-Forwarded-For IP.

        Pre-builds a rotating pool of fully-formed header dicts per cache key
        so the hot path is a single pool rotation -- no dict copy, no IP generation.
        """
        cache_key = f"{app_fr}:{self.config.glm_user_agent}:{self.config.glm_use_guest_refresh_token}"
        pool = self._STATIC_HEADERS_CACHE.get(cache_key)
        if pool is not None:
            return pool.get()

        # Build one pool entry's worth of shared headers (excl. X-Forwarded-For)
        base = {
            "Accept": "application/json, text/plain, */*" if app_fr == "default" else "text/event-stream",
            "Accept-Encoding": "gzip, deflate" if app_fr == "default" else "identity",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "App-Name": "chatglm",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "Origin": "https://chatglm.cn",
            "Pragma": "no-cache",
            "Priority": "u=1, i",
            "Sec-Ch-Ua": '"Google Chrome";v="120", "Chromium";v="120", "Not A(Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Linux"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": self.config.glm_user_agent,
            "X-App-Fr": app_fr,
            "X-App-Platform": "pc",
            "X-App-Version": "1.0.83",  # ponytail: match real app version
            "X-Device-Brand": "",
            "X-Device-Model": "",
            "X-Lang": "en" if self.config.glm_use_guest_refresh_token else "zh",
        }

        # Pre-compute pool of headers with different X-Forwarded-For IPs
        pool_headers = []
        for _ in range(_HEADER_POOL_SIZE):
            h = base.copy()
            h["X-Forwarded-For"] = _random_ip()
            pool_headers.append(h)

        pool = _HeaderPool(pool_headers)
        self._STATIC_HEADERS_CACHE[cache_key] = pool
        return pool.get()

    def read_json_response(self, raw_body: bytes) -> dict[str, object]:
        try:
            debug_dump(self.logger, self.config.debug_dump_all, "GLM 原始 JSON 响应体", raw_body)
            payload = orjson.loads(raw_body)
        except UnicodeDecodeError as exc:
            raise RuntimeError("GLM 响应不是合法 UTF-8") from exc
        except orjson.JSONDecodeError as exc:
            raise RuntimeError(f"GLM 响应不是合法 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"GLM 响应格式异常，期望 JSON 对象，实际是: {type(payload).__name__}")
        return payload

    def get_access_token(self) -> str:
        with self._lock:
            idx = self._current_index
        return self._get_access_token_for_index(idx)

    def get_account_count(self) -> int:
        with self._lock:
            return len(self._accounts)

    def is_guest_account(self, account_index: int) -> bool:
        with self._lock:
            return self._accounts[account_index].is_guest

    def advance_account(self, failed_index: int, reason: str) -> int:
        with self._lock:
            if failed_index != self._current_index:
                return self._current_index
            next_index = (failed_index + 1) % len(self._accounts)
            self._current_index = next_index
            self.logger.warning(
                "账号请求失败，切换 refresh_token 账号 index=%s -> %s reason=%s",
                failed_index,
                next_index,
                reason,
            )
            return next_index

    def reset_account_cycle(self) -> None:
        with self._lock:
            self._current_index = 0

    def invalidate_account(self, account_index: int) -> None:
        with self._lock:
            self._accounts[account_index].cached_token = None

    def get_access_token_for_account(self, account_index: int) -> str:
        with self._lock:
            if self._circuit_open:
                now = time.monotonic()
                if now - self._circuit_opened_at > self._circuit_cooldown:
                    if not self._circuit_probe_in_progress:
                        # Half-open: allow exactly ONE probe request through.
                        # Keep _circuit_failures intact so a single failure
                        # re-opens the circuit immediately.
                        self._circuit_probe_in_progress = True
                        self._circuit_probe_at = now
                        self._circuit_probe_failures = 0
                        self.logger.info("断路器半开 — 允许探测请求通过")
                    else:
                        raise UpstreamBlockedError(
                            "上游断路保护 — 探测请求正在进行中，请稍后重试。"
                        )
                else:
                    remaining = self._circuit_cooldown - (now - self._circuit_opened_at)
                    raise UpstreamBlockedError(
                        f"上游断路保护 — 最近 {self._circuit_threshold} 次连续失败，剩余 {remaining:.0f}s 冷却，请稍后重试。"
                    )
        return self._get_access_token_for_index(account_index)

    def has_cached_token(self, account_index: int) -> bool:
        """Check if account has a valid cached token without triggering a refresh."""
        with self._lock:
            if account_index < 0 or account_index >= len(self._accounts):
                return False
            account = self._accounts[account_index]
            return bool(account.cached_token and time.time() < account.cached_token.expires_at - 60)


    def get_fill_first_account(self, preferred: int | None = None) -> int | None:
        """Fill First: use one account until near quota, then advance.
        Returns None if no usable account found — caller should spawn a spare."""
        with self._lock:
            idx = preferred if preferred is not None else self._fill_first_index
            if 0 <= idx < len(self._accounts):
                a = self._accounts[idx]
                is_full = a.tool_call_count >= 6 or a.search_count >= 4
                is_limited = idx in self._rate_limited_accounts or idx in self._silent_accounts
                if not is_full and not is_limited and a.cached_token:
                    return idx
            for offset in range(len(self._accounts)):
                n = (idx + 1 + offset) % len(self._accounts) if idx >= 0 else offset
                if n < len(self._accounts):
                    a = self._accounts[n]
                    if n not in self._rate_limited_accounts and n not in self._silent_accounts and a.cached_token:
                        self._fill_first_index = n
                        return n
            self._fill_first_index = 0
            return None

    def _select_ewma_account(self) -> int:
        """Select account using EWMA-weighted routing (power-of-two-choices).

        Caller must hold self._lock. Cleans expired rate limits before selection.
        Returns the index of the selected account.
        """
        now = time.time()
        expired = [i for i, ts in self._rate_limited_accounts.items()
                   if now - ts > self._rate_limit_cooldowns.get(i, self._rate_limit_cooldown)]
        for i in expired:
            del self._rate_limited_accounts[i]
            self.logger.info("账号限速冷却结束 index=%s", i)
        # EWMA-weighted routing: prefer accounts with lower latency, skip rate-limited
        # Power-of-two-choices for O(1) selection with good distribution
        usable = [i for i in range(len(self._accounts))
                  if i not in self._rate_limited_accounts and i not in self._silent_accounts]
        if not usable:
            return min(self._rate_limited_accounts, key=self._rate_limited_accounts.get, default=0)
        if len(usable) == 1:
            return usable[0]
        c1 = random.choice(usable)
        c2 = random.choice(usable)
        a1 = self._accounts[c1]
        a2 = self._accounts[c2]
        # If either has no EWMA data, prefer the one without (fresh accounts)
        if a1.ewma_latency == 0.0 and a2.ewma_latency > 0.0:
            return c1
        if a2.ewma_latency == 0.0 and a1.ewma_latency > 0.0:
            return c2
        # Add jitter to avoid stampede
        jitter1 = a1.ewma_latency * (1 + random.uniform(-0.1, 0.1))
        jitter2 = a2.ewma_latency * (1 + random.uniform(-0.1, 0.1))
        return c1 if jitter1 <= jitter2 else c2

    def get_next_account_index(self) -> int:
        # ponytail: strategy dispatch — check env var first (runtime override), then config
        _strategy = __import__("os").environ.get("GLM_ACCOUNT_STRATEGY", self.config.glm_account_strategy)
        if _strategy == "fill_first":
            idx = self.get_fill_first_account()
            if idx is not None:
                return idx
            # Fall through to EWMA selection when fill_first finds no usable account
        with self._lock:
            return self._select_ewma_account()

    def mark_rate_limited(self, account_index: int) -> None:
        import time as _t
        with self._lock:
            self._rate_limited_accounts[account_index] = _t.time()
            # Progressive backoff: each consecutive failure doubles the cooldown
            base = self._rate_limit_cooldown
            failures = self._account_fail_count.get(account_index, 0) + 1
            self._account_fail_count[account_index] = failures
            progressive = min(base * (2 ** (failures - 1)), 30.0)
            self._rate_limit_cooldowns[account_index] = progressive
            if account_index < len(self._accounts):
                self._accounts[account_index].cached_token = None
            self.logger.warning(
                "账号限速 index=%s 冷却=%.0fs (渐进式第%d次)",
                account_index, progressive, failures
            )

    def force_refresh_all_guest_tokens(self) -> None:
        import time as _t
        now = _t.monotonic()
        if now - self._last_full_refresh_at < self._full_refresh_cooldown:
            self.logger.info("跳过全量刷新（冷却中）last=%.0fs ago cooldown=%.0fs", now - self._last_full_refresh_at, self._full_refresh_cooldown)
            return
        self._last_full_refresh_at = now
        with self._lock:
            for a in self._accounts:
                if a.is_guest: a.cached_token = None
            self._rate_limited_accounts.clear()
            self._round_robin_counter = 0
            self._consecutive_failures = 0
            self._silent_accounts.clear()
        # ponytail: parallel token refresh via ThreadPoolExecutor.
        # Sequential 3 accounts × 2s = 6s; parallel = ~2s total.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        guests = [i for i, a in enumerate(self._accounts) if a.is_guest]
        if not guests:
            self.logger.info("没有需要刷新的游客 token")
            return
        with ThreadPoolExecutor(max_workers=min(len(guests), 3)) as pool:
            futures = {pool.submit(self._fetch_guest_access_token, i): i for i in guests}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    self._accounts[i].cached_token = future.result()
                except Exception as exc:
                    self.logger.warning("全量刷新槽位失败 index=%s error=%s", i, exc)
        self.logger.warning("强制刷新所有游客 token")

    def _start_sliding_refresh(self) -> None:
        """Background thread: proactively refresh tokens before they expire.
        ponytail: prevents 7s per-request token refresh delay. Runs every 5 min,
        refreshes any token with < 30 min remaining."""
        if not self.config.glm_use_guest_refresh_token:
            return

        def _loop():
            while True:
                time.sleep(300)  # every 5 minutes
                try:
                    now = time.time()
                    refresh_before = now + 1800  # refresh if < 30 min remaining
                    stale = [i for i, a in enumerate(self._accounts)
                             if a.is_guest and a.cached_token and a.cached_token.expires_at < refresh_before]
                    if not stale:
                        continue
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    with ThreadPoolExecutor(max_workers=min(len(stale), 10)) as pool:
                        futures = {pool.submit(self._fetch_guest_access_token, i, True): i for i in stale}
                        for f in as_completed(futures):
                            i = futures[f]
                            try:
                                self._accounts[i].cached_token = f.result()
                            except Exception:
                                pass
                    self.logger.info("Sliding refresh: %d tokens refreshed", len(stale))
                except Exception:
                    pass

        threading.Thread(target=_loop, daemon=True).start()

    def prewarm_guest_tokens(self) -> None:
        """Prewarm guest tokens in parallel. ponytail: limit to 5 to conserve quota."""
        guests = [i for i, a in enumerate(self._accounts) if a.is_guest and not a.cached_token][:5]
        if not guests:
            return
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(len(guests), 3)) as pool:
            futures = {pool.submit(self._fetch_guest_access_token, i, True): i for i in guests}
            for f in as_completed(futures):
                i = futures[f]
                try:
                    token = f.result()
                    if token:
                        self._accounts[i].cached_token = token
                        self.logger.info("预热游客账号 token index=%s", i)
                except Exception as exc:
                    self.logger.warning("预热失败 index=%s %s", i, exc)

    def _get_access_token_for_index(self, account_index: int) -> str:
        with self._lock:
            account = self._accounts[account_index]
            if account.cached_token and time.time() < account.cached_token.expires_at - 60:
                self.logger.debug("使用缓存 access_token account=%s 剩余=%.0fs", account_index, account.cached_token.expires_at - time.time())
                return account.cached_token.access_token

        # Cache miss — do refresh WITHOUT holding self._lock
        # This prevents blocking all other threads during network I/O
        new_token = self._refresh_access_token(account_index)

        # Update cache under lock
        with self._lock:
            self._accounts[account_index].cached_token = new_token
            return new_token.access_token

    def _exec_with_retry(self, operation, max_attempts=3, base_delay=2.0):
        """Execute operation with exponential backoff retry."""
        last_exc = None
        for attempt in range(max_attempts):
            try:
                return operation()
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    delay = min(30, base_delay * (2 ** attempt))
                    time.sleep(delay)
        raise last_exc

    def _refresh_access_token(self, account_index: int) -> AccessToken:
        account = self._accounts[account_index]
        if account.is_guest or not account.refresh_token:
            return self._fetch_guest_access_token(account_index)

        def _attempt() -> AccessToken:
            timestamp, nonce, sign = build_sign()
            request = urllib.request.Request(
                self.config.refresh_url,
                data=b"{}",
                method="POST",
                headers={
                    **self.get_browser_headers(),
                    "Authorization": f"Bearer {account.refresh_token}",
                    "X-Device-Id": uuid.uuid4().hex,
                    "X-Nonce": nonce,
                    "X-Request-Id": uuid.uuid4().hex,
                    "X-Sign": sign,
                    "X-Timestamp": timestamp,
                },
            )
            debug_dump(self.logger, self.config.debug_dump_all, f"GLM 刷新 access_token 请求头 account={account_index}", dict(request.header_items()))
            debug_dump(self.logger, self.config.debug_dump_all, f"GLM 刷新 access_token 请求体 account={account_index}", b"{}")
            # ponytail: route through proxy pool to avoid direct connection SSL issues
            status_code, payload = self._do_upstream_request(request, use_unique_proxy=True)
            code = payload.get("code", payload.get("status"))
            result = payload.get("result") or {}
            access_token = result.get("access_token")
            refresh_token = result.get("refresh_token", account.refresh_token)
            if status_code != 200 or code not in {0, None} or not access_token:
                raise RuntimeError(f"刷新 GLM token 失败: {payload}")
            if refresh_token != account.refresh_token:
                try:
                    self._persist_refresh_token(account_index, refresh_token)
                except Exception as exc:
                    self.logger.warning("写回 GLM refresh_token 失败 index=%s error=%s", account_index, exc)
                account.refresh_token = refresh_token
                self.config.glm_refresh_tokens[account_index] = refresh_token
                if account_index == 0:
                    self.config.glm_refresh_token = refresh_token
                self.logger.info("GLM refresh_token 已自动刷新并写回账号存储 index=%s", account_index)
            return AccessToken(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_at=time.time() + ACCESS_TOKEN_EXPIRES_SECONDS - random.randint(10, 30),
            )

        try:
            result = self._exec_with_retry(_attempt, max_attempts=3, base_delay=2.0)
            self.record_upstream_success()
            return result
        except Exception as last_exc:
            self.record_upstream_failure()
            self.logger.error(
                "刷新 access_token 最终失败 account=%s after %s attempts: %s",
                account_index, 3, last_exc,
            )
            raise

    def _fetch_guest_access_token(self, account_index: int, rate_limit: bool = True) -> AccessToken:
        import time as _t
        if rate_limit:
            with self._lock:
                elapsed = _t.time() - self._last_guest_fetch
                if elapsed < self._guest_fetch_interval:
                    import random as _r
                    sleep_time = self._guest_fetch_interval - elapsed + _r.uniform(0, 0.5)
                else:
                    sleep_time = 0
                self._last_guest_fetch = _t.time()
            if sleep_time > 0:
                _t.sleep(sleep_time)
        account = self._accounts[account_index]
        # ponytail: match JS — X-Device-Id is persistent per account (not random each call)
        if not account._device_id:
            account._device_id = uuid.uuid4().hex

        def _attempt() -> AccessToken:
            timestamp, nonce, sign = build_sign()
            request_id = uuid.uuid4().hex
            device_id = account._device_id
            request = urllib.request.Request(
                self.config.guest_refresh_url,
                data=b"",
                method="POST",
	            headers={
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
	                    # ponytail: match config User-Agent for consistency with curl_cffi TLS
	                    "User-Agent": self.config.glm_user_agent,
	                    "X-App-Fr": "browser_extension",
	                    "X-App-Platform": "pc",
	                    "X-App-Version": "1.0.83",
	                    "X-Device-Id": device_id,
	                    "X-Nonce": nonce,
	                    "X-Request-Id": request_id,
	                    "X-Sign": sign,
	                    "X-Timestamp": timestamp,
	                    "X-Lang": "en",
	                },
            )
            debug_dump(self.logger, self.config.debug_dump_all, f"GLM 游客 token 请求头 account={account_index}", dict(request.header_items()))
            debug_dump(self.logger, self.config.debug_dump_all, f"GLM 游客 token 请求体 account={account_index}", b"")
            # Use unique proxy rotation for guest token creation to avoid per-IP rate limiting
            status_code, payload = self._do_upstream_request(request, use_unique_proxy=True)
            code = payload.get("code", payload.get("status"))
            result = payload.get("result") or {}
            access_token = result.get("access_token")
            refresh_token = result.get("refresh_token")
            if status_code != 200 or code not in {0, None} or not access_token or not refresh_token:
                raise RuntimeError(f"获取 GLM 游客 token 失败: {payload}")
            account.refresh_token = str(refresh_token)
            self.logger.info("已获取新的 GLM 游客 refresh_token index=%s", account_index)
            return AccessToken(
                access_token=str(access_token),
                refresh_token=str(refresh_token),
                expires_at=time.time() + ACCESS_TOKEN_EXPIRES_SECONDS - random.randint(10, 30),
            )

        try:
            result = self._exec_with_retry(_attempt, max_attempts=3, base_delay=5.0)
            self.record_upstream_success()
            return result
        except Exception as last_exc:
            self.record_upstream_failure()
            self.logger.error(
                "获取游客 token 最终失败 account=%s after %s attempts: %s",
                account_index, 3, last_exc,
            )
            raise

    def _persist_refresh_token(self, account_index: int, refresh_token: str) -> None:
        with self._persist_lock:
            if self._accounts[account_index].is_guest:
                return
            if self.config.token_file_path.exists() or len(self.config.glm_refresh_tokens) > 1:
                tokens = list(self.config.glm_refresh_tokens)
                tokens[account_index] = refresh_token
                content = "\n".join(tokens) + "\n"
                try:
                    self.config.token_file_path.write_text(content, encoding="utf-8")
                except OSError as exc:
                    raise RuntimeError(f"写入 token 文件失败: {self.config.token_file_path} error={exc}") from exc
                return
            self._persist_env_refresh_token(refresh_token)

    def _persist_env_refresh_token(self, refresh_token: str) -> None:
        env_path = self.config.env_file_path
        if not env_path.exists():
            self.logger.warning(".env 文件不存在，无法自动写回新的 refresh_token")
            return

        try:
            content = env_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f".env 不是有效的 UTF-8 编码: {env_path}") from exc
        except OSError as exc:
            raise RuntimeError(f"读取 .env 失败: {env_path} error={exc}") from exc
        lines = content.splitlines()
        updated = False

        for index, line in enumerate(lines):
            if line.startswith("GLM_REFRESH_TOKEN="):
                lines[index] = f"GLM_REFRESH_TOKEN={refresh_token}"
                updated = True
                break

        if not updated:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(f"GLM_REFRESH_TOKEN={refresh_token}")

        new_content = "\n".join(lines) + "\n"
        try:
            env_path.write_text(new_content, encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"写入 .env 失败: {env_path} error={exc}") from exc

    def should_switch_account(self, exc: Exception) -> bool:
        if isinstance(exc, socket.timeout):
            return True
        if isinstance(exc, TimeoutError):
            return True
        if isinstance(exc, OSError) and hasattr(exc, 'errno'):
            import errno as _e
            if exc.errno in (_e.ETIMEDOUT, _e.EHOSTUNREACH, _e.ENETUNREACH, _e.ECONNRESET, _e.ECONNREFUSED, _e.ECONNABORTED):
                return True
        if isinstance(exc, http.client.IncompleteRead):
            return True
        if isinstance(exc, urllib.error.URLError):
            # ponytail: urllib wraps ConnectionResetError into URLError
            return True
        if isinstance(exc, ConnectionError):
            return True
        if isinstance(exc, urllib.error.HTTPError):
            code = getattr(exc, "code", 0)
            if code in {401, 403}:
                return True
            return False
        if hasattr(exc, "status_code"):
            code = getattr(exc, "status_code", 0)
            if code in {401, 403, 429, 502, 503}:
                return True
        if isinstance(exc, urllib.error.URLError):
            return False
        if isinstance(exc, RuntimeError):
            return "token" in str(exc).lower()
        return False

    def record_silence(self, account_index: int) -> None:
        import time as _t
        with self._lock:
            self._silent_accounts[account_index] = _t.time()
            if account_index < len(self._accounts):
                self._accounts[account_index].cached_token = None
            self._consecutive_failures += 1
        self.logger.warning("账号无响应 index=%s 冷却=%.0fs consecutive_failures=%s", account_index, self._silence_cooldown, self._consecutive_failures)

    def is_starved(self) -> bool:
        with self._lock:
            active = sum(1 for i, a in enumerate(self._accounts)
                         if a.cached_token is not None
                         and i not in self._rate_limited_accounts
                         and i not in self._silent_accounts)
            return active == 0 and len(self._accounts) > 0

    def is_account_usable(self, account_index: int) -> bool:
        with self._lock:
            if account_index in self._rate_limited_accounts:
                import time as _t
                cooldown = self._rate_limit_cooldowns.get(account_index, self._rate_limit_cooldown)
                if _t.time() - self._rate_limited_accounts[account_index] < cooldown:
                    return False
            if account_index in self._silent_accounts:
                import time as _t
                if _t.time() - self._silent_accounts[account_index] < self._silence_cooldown:
                    return False
            if account_index >= len(self._accounts):
                return False
            return True

    def acquire_account_for_stream(self, preferred_account_index: int | None = None) -> tuple[int, str]:
        account_count = len(self._accounts)
        if account_count <= 0:
            raise RuntimeError("没有可用的 GLM 账号")

        import time as _t
        now = _t.time()

        # Batch find-usable + get-cached-token under one lock acquisition
        with self._lock:
            # Get starting index (EWMA-weighted or preferred)
            if preferred_account_index is None:
                _strategy = __import__("os").environ.get("GLM_ACCOUNT_STRATEGY", self.config.glm_account_strategy)
                start = None
                if _strategy == "fill_first":
                    start = self.get_fill_first_account()
                if start is None:
                    start = self._select_ewma_account()
            else:
                start = preferred_account_index % account_count

            # Find a usable account and get cached token
            for offset in range(account_count):
                idx = (start + offset) % account_count
                # Inline is_account_usable check
                if idx in self._rate_limited_accounts:
                    cooldown = self._rate_limit_cooldowns.get(idx, self._rate_limit_cooldown)
                    if now - self._rate_limited_accounts[idx] < cooldown:
                        continue
                if idx in self._silent_accounts:
                    if now - self._silent_accounts[idx] < self._silence_cooldown:
                        continue
                # Check cached token
                account = self._accounts[idx]
                if account.cached_token and now < account.cached_token.expires_at - 60:
                    return idx, account.cached_token.access_token
                # Token needs refresh — do it outside lock
                account_index = idx
                break
            else:
                account_index = None

        # Token refresh (or fallback) happens outside the lock
        if account_index is not None:
            token = self.get_access_token_for_account(account_index)
            return account_index, token

        if self.config.glm_use_guest_refresh_token:
            try:
                fresh_idx = self.spawn_fresh_guest_account()
                self.logger.info("所有账号不可用，动态创建新账号 index=%s", fresh_idx)
                self.reset_account_cycle()
                token = self.get_access_token_for_account(fresh_idx)
                return fresh_idx, token
            except Exception as exc:
                self.logger.warning("动态创建账号失败: %s", exc)

        self.force_refresh_all_guest_tokens()
        try:
            return 0, self.get_access_token_for_account(0)
        except Exception as exc:
            raise RuntimeError(f"无可用账号: {exc}") from exc

    def record_latency(self, account_index: int, latency_ms: float) -> None:
        """Record request latency for gradient failover (EWMA)."""
        with self._lock:
            if account_index < len(self._accounts):
                a = self._accounts[account_index]
                if a.ewma_latency == 0.0:
                    a.ewma_latency = latency_ms
                else:
                    a.ewma_latency = (1 - a.ewma_alpha) * a.ewma_latency + a.ewma_alpha * latency_ms

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._last_success_at = time.monotonic()

    def clear_silent_account(self, account_index: int) -> None:
        with self._lock:
            self._silent_accounts.pop(account_index, None)

    # ------------------------------------------------------------------ #
    #  Reliability: Watchdog, Aggressive Recovery, Subsystem Status
    # ------------------------------------------------------------------ #

    def _start_watchdog(self) -> None:
        """Background watchdog thread.

        Checks every 60s:
          - If no successful request in 300s (watchdog timeout), triggers
            aggressive recovery (reset circuit breaker, force token refresh).
          - If the pool is starved (all accounts exhausted), triggers
            immediate recovery without waiting for the 300s timer.
        """
        def _loop():
            import time as _t
            _watchdog_timeout = 300.0
            while True:
                _t.sleep(60)
                try:
                    now = _t.monotonic()
                    with self._lock:
                        last_ok = self._last_success_at
                        starved = self.is_starved()
                    if starved:
                        self.logger.warning("看门狗: 检测到账号池完全枯竭，触发激进恢复")
                        self._aggressive_recovery()
                    elif last_ok > 0 and now - last_ok > _watchdog_timeout:
                        self.logger.warning(
                            "看门狗: %.0fs 无成功请求 (阈值 %ds)，触发激进恢复",
                            now - last_ok, _watchdog_timeout,
                        )
                        self._aggressive_recovery()
                except Exception:
                    pass

        threading.Thread(target=_loop, daemon=True, name="watchdog").start()
        self.logger.info("看门狗已启动 (检查间隔=60s, 超时阈值=300s)")

    def _aggressive_recovery(self) -> None:
        """Aggressive recovery from total starvation.

        Resets the circuit breaker, clears all rate limits and silence
        cooldowns, and forces a full refresh of all guest tokens.
        This is the nuclear option — called by the watchdog when the
        system has been completely stuck for too long.
        """
        import time as _t
        with self._lock:
            self._circuit_open = False
            self._circuit_failures = 0
            self._circuit_opened_at = 0.0
            self._circuit_probe_in_progress = False
            self._consecutive_failures = 0
            self._rate_limited_accounts.clear()
            self._rate_limit_cooldowns.clear()
            self._silent_accounts.clear()
            self._account_fail_count.clear()
            # Allow immediate force refresh
            self._last_full_refresh_at = 0.0
            self.logger.warning("激进恢复: 电路关闭，限速/静默已清除，准备刷新 token")
        self.force_refresh_all_guest_tokens()
        self.logger.warning("激进恢复完成 — 系统已重置")

    def get_subsystem_status(self) -> dict[str, object]:
        """Return a snapshot of auth subsystem health for the /health endpoint.

        Shows token pool state, circuit breaker state, queue depth,
        and starvation status.
        """
        import time as _t
        now = _t.time()
        with self._lock:
            total = len(self._accounts)
            active = sum(1 for i, a in enumerate(self._accounts)
                         if a.cached_token is not None
                         and time.time() < a.cached_token.expires_at - 60
                         and i not in self._rate_limited_accounts
                         and i not in self._silent_accounts)
            cached = sum(1 for a in self._accounts if a.cached_token is not None)
            rate_limited = len(self._rate_limited_accounts)
            silent = len(self._silent_accounts)
            circuit_open = self._circuit_open
            circuit_failures = self._circuit_failures
            if circuit_open:
                remaining = max(0.0, self._circuit_cooldown - (_t.monotonic() - self._circuit_opened_at))
            else:
                remaining = 0.0
            consecutive_failures = self._consecutive_failures
            starved = self.is_starved()
            last_success = self._last_success_at
            warm_spares = len(self._warm_spares)

        return {
            "token_pool": {
                "total_slots": total,
                "cached_tokens": cached,
                "active_usable": active,
                "rate_limited": rate_limited,
                "silent": silent,
                "starved": starved,
                "warm_spares": warm_spares,
            },
            "circuit_breaker": {
                "open": circuit_open,
                "failures": circuit_failures,
                "remaining_cooldown_seconds": round(remaining, 1),
                "threshold": self._circuit_threshold,
            },
            "consecutive_failures": consecutive_failures,
            "last_success_monotonic_seconds_ago": round(
                _t.monotonic() - last_success, 1
            ) if last_success > 0 else None,
            "account_strategy": __import__("os").environ.get(
                "GLM_ACCOUNT_STRATEGY", self.config.glm_account_strategy
            ),
        }

    def track_account_failure(self, account_index: int) -> bool:
        """Track a failure for circuit breaker. Returns True if threshold reached."""
        with self._lock:
            count = self._account_fail_count.get(account_index, 0) + 1
            self._account_fail_count[account_index] = count
            if count >= self._account_fail_threshold:
                self.logger.warning("账号熔断触发 index=%s fails=%s", account_index, count)
                self._rate_limited_accounts[account_index] = __import__("time").time()
                self._accounts[account_index].cached_token = None
                del self._account_fail_count[account_index]
                return True
            self.logger.debug("账号失败计数 index=%s count=%s", account_index, count)
            return False

    def clear_account_failures(self, account_index: int) -> None:
        with self._lock:
            self._account_fail_count.pop(account_index, None)

    def record_empty_response(self, account_index: int) -> None:
        """Mark account as exhausted (empty upstream response = quota used up).
        Invalidates token so next request fetches a fresh one with new quota.
        Also marks as rate-limited to prevent immediate reuse."""
        import time as _t
        with self._lock:
            if 0 <= account_index < len(self._accounts):
                self._accounts[account_index].cached_token = None
                self._rate_limited_accounts[account_index] = _t.time()
                self.logger.info("账号 quota 耗尽 index=%s — 已清空 token，下次请求将刷新", account_index)
