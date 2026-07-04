from __future__ import annotations

import hashlib
import http.client
import gzip
import json
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


SIGN_SECRET = "8a1317a7468aa3ad86e997d08f3f31cb"
ACCESS_TOKEN_EXPIRES_SECONDS = 3600


def build_sign() -> tuple[str, str, str]:
    now = str(int(time.time() * 1000))
    digits = [int(char) for char in now]
    checksum = (sum(digits) - digits[-2]) % 10
    timestamp = now[:-2] + str(checksum) + now[-1]
    nonce = uuid.uuid4().hex
    sign = hashlib.md5(f"{timestamp}-{nonce}-{SIGN_SECRET}".encode("utf-8")).hexdigest()
    return timestamp, nonce, sign


def build_random_x_forwarded_for() -> str:
    while True:
        first_octet = random.randint(1, 223)
        if first_octet in {10, 127, 169, 172, 192}:
            continue
        octets = [first_octet]
        for _ in range(3):
            octets.append(random.randint(0, 255))
        return ".".join(str(octet) for octet in octets)


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
        self._lock = threading.Lock()
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
        self._init_guest_pool(config.glm_refresh_tokens)
        self._start_sliding_refresh()
        self._warm_connection()
        self._start_heartbeat()

    def _warm_connection(self) -> None:
        """Warm TLS session + WAF cookie by fetching homepage first.
        Uses one-shot requests to avoid polluting the cached Session pool."""
        from .http_client import do_request_oneshot
        # Warm the direct connection
        try:
            do_request_oneshot("GET", "https://chatglm.cn/", headers={
                "User-Agent": self.config.glm_user_agent,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "zh-CN,zh;q=0.9",
                "Origin": "https://chatglm.cn",
                "Referer": "https://chatglm.cn/",
            }, timeout=5, logger=self.logger)
            self.logger.info("Connection warmed (TLS session + WAF cookie established)")
        except Exception:
            pass  # non-fatal
        # Also warm a few proxy connections to establish WAF cookies on proxy IPs
        try:
            from .glm2api_proxy import get_pool
            pool = get_pool()
            warmed = 0
            for _ in range(3):
                proxy_url = pool.get_unique()
                if proxy_url:
                    try:
                        do_request_oneshot("GET", "https://chatglm.cn/", headers={
                            "User-Agent": self.config.glm_user_agent,
                            "Accept": "text/html,application/xhtml+xml",
                            "Accept-Language": "zh-CN,zh;q=0.9",
                        }, proxy_url=proxy_url, timeout=5, logger=self.logger)
                        warmed += 1
                    except Exception:
                        pass
            if warmed:
                self.logger.info("Warmed %s proxy connections with WAF cookies", warmed)
        except Exception:
            pass

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
        if self._circuit_failures > 0:
            self._circuit_failures = 0
            self.logger.info("上游恢复 — 重置失败计数")
        if self._circuit_open:
            self._circuit_open = False
            self.logger.info("断路器关闭 — 上游已恢复")

    def _do_upstream_request(self, request: urllib.request.Request, use_unique_proxy: bool = False) -> tuple[int, dict]:
        """Execute upstream request via curl_cffi with browser TLS fingerprinting.
        Routes through proxy pool when available."""
        from .glm2api_proxy import get_pool
        from .http_client import do_json_request
        pool = get_pool()
        proxy_url = pool.get_unique() if use_unique_proxy else pool.get_next()
        if not proxy_url and pool._proxies:
            proxy_url = next(iter(pool._proxies))
        try:
            status_code, payload = do_json_request(
                method=request.get_method() if hasattr(request, 'get_method') else 'POST',
                url=str(request.full_url) if hasattr(request, 'full_url') else str(request),
                headers=dict(request.header_items()),
                data=request.data if hasattr(request, 'data') and request.data else None,
                proxy_url=proxy_url,
                timeout=self.config.request_timeout,
                logger=self.logger,
            )
            if proxy_url:
                pool.report_success(proxy_url, 0)
            return (status_code, payload)
        except Exception as exc:
            if proxy_url:
                pool.report_failure(proxy_url)
            raise urllib.error.URLError(str(exc)) from exc

    @staticmethod
    def _is_jwt_guest_token(token: str) -> bool:
        """Check if a JWT token is a guest token by inspecting the payload."""
        if token == GUEST_REFRESH_TOKEN_MARKER:
            return True
        try:
            import base64 as _b64, json as _json
            parts = token.strip().split(".")
            if len(parts) == 3:
                payload = parts[1]
                # Add padding for base64 decode
                payload += "=" * (4 - len(payload) % 4)
                decoded = _b64.urlsafe_b64decode(payload)
                data = _json.loads(decoded)
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
        """Background thread: pre-warms guest access tokens from pre-created refresh_tokens.
        Uses staggered 3s intervals to avoid triggering the Tengine CDN per-IP rate limit."""
        import time as _t
        for idx, acct in enumerate(self._accounts):
            if not acct.is_guest:
                continue
            if acct.cached_token is not None:
                continue  # already warm
            try:
                acct.cached_token = self._fetch_guest_access_token(idx, rate_limit=True)
                if acct.cached_token:
                    self.logger.info("后台预刷新 access_token 成功 index=%s", idx)
                else:
                    self.logger.warning("后台预刷新 access_token 失败 index=%s", idx)
            except Exception as exc:
                self.logger.warning("后台预刷新 access_token 失败 index=%s error=%s", idx, exc)
            # Rate-limit friendly sleep between attempts
            _t.sleep(3)
        active = sum(1 for a in self._accounts if a.cached_token is not None)
        self.logger.info("后台预刷新完成 active=%s/%s", active, len(self._accounts))

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
    
    def hard_swap_account(self, failed_index: int) -> int:
        """Replace a failed guest account with a fresh one. Returns new index."""
        refreshed = AccountState(refresh_token="", is_guest=True)
        with self._lock:
            self._accounts[failed_index] = refreshed
    
        self.logger.info("强制刷新游客账号 index=%s", failed_index)
        try:
            refreshed.cached_token = self._fetch_guest_access_token(failed_index)
        except Exception as exc:
            self.logger.warning("强制刷新游客账号失败 index=%s error=%s", failed_index, exc)
            raise
        self.logger.info("游客账号已刷新 index=%s 获取新配额", failed_index)
        return failed_index
    
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
                    # We'll fetch outside the lock to avoid blocking
                    pass
                else:
                    continue
                break
            else:
                return None
        try:
            a.cached_token = self._fetch_guest_access_token(idx)
            self.logger.info("提前预热游客账号 index=%s", idx)
            return idx
        except Exception as exc:
            self.logger.warning("预热游客账号失败 index=%s error=%s", idx, exc)
            return None
    

    # ponytail: cache static headers to avoid rebuilding dict on every request
    _STATIC_HEADERS_CACHE: dict[str, dict[str, str]] = {}

    def get_browser_headers(self, app_fr: str = "browser_extension") -> dict[str, str]:
        cache_key = f"{app_fr}:{self.config.glm_user_agent}:{self.config.glm_use_guest_refresh_token}"
        if cache_key not in self._STATIC_HEADERS_CACHE:
            self._STATIC_HEADERS_CACHE[cache_key] = {
                "Accept": "application/json, text/plain, */*" if app_fr == "default" else "text/event-stream",
                "Accept-Encoding": "gzip, deflate" if app_fr == "default" else "identity",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
                "App-Name": "chatglm",
                "Cache-Control": "no-cache",
                "Content-Type": "application/json",
                "Origin": "https://chatglm.cn",
                "Pragma": "no-cache",
                "Priority": "u=1, i",
                "Sec-Ch-Ua": '"Microsoft Edge";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-origin",
                "User-Agent": self.config.glm_user_agent,
                "X-App-Fr": app_fr,
                "X-App-Platform": "pc",
                "X-App-Version": "0.0.1",
                "X-Device-Brand": "",
                "X-Device-Model": "",
                "X-Lang": "en" if self.config.glm_use_guest_refresh_token else "zh",
            }
        headers = dict(self._STATIC_HEADERS_CACHE[cache_key])
        headers["X-Forwarded-For"] = build_random_x_forwarded_for()
        return headers

    def read_json_response(self, response) -> dict[str, object]:
        try:
            raw_body = response.read()
            content_encoding = response.headers.get("Content-Encoding", "").lower()

            if content_encoding == "gzip":
                raw_body = gzip.decompress(raw_body)

            debug_dump(self.logger, self.config.debug_dump_all, "GLM 原始 JSON 响应体", raw_body)
            payload = json.loads(raw_body.decode("utf-8"))
        except gzip.BadGzipFile as exc:
            raise RuntimeError("GLM 响应 gzip 解压失败") from exc
        except UnicodeDecodeError as exc:
            raise RuntimeError("GLM 响应不是合法 UTF-8") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"GLM 响应不是合法 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"GLM 响应格式异常，期望 JSON 对象，实际是: {type(payload).__name__}")
        return payload

    def get_access_token(self) -> str:
        with self._lock:
            return self._get_access_token_for_index(self._current_index)

    def get_account_count(self) -> int:
        return len(self._accounts)

    def get_current_account_index(self) -> int:
        with self._lock:
            return self._current_index

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
        # ponytail: circuit breaker — fail-fast when upstream is down
        if self._circuit_open:
            raise UpstreamBlockedError(
                f"上游断路保护 — 最近 {self._circuit_threshold} 次连续失败，暂停 {self._circuit_cooldown}s，请稍后重试。"
            )
        with self._lock:
            return self._get_access_token_for_index(account_index)

    def has_cached_token(self, account_index: int) -> bool:
        """Check if account has a valid cached token without triggering a refresh."""
        if account_index < 0 or account_index >= len(self._accounts):
            return False
        account = self._accounts[account_index]
        return bool(account.cached_token and time.time() < account.cached_token.expires_at - 60)


    def get_next_account_index(self) -> int:
        import time as _t
        with self._lock:
            now = _t.time()
            expired = [i for i, ts in self._rate_limited_accounts.items() if now - ts > self._rate_limit_cooldown]
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
            import random as _rnd
            c1 = _rnd.choice(usable)
            c2 = _rnd.choice(usable)
            a1 = self._accounts[c1]
            a2 = self._accounts[c2]
            # If either has no EWMA data, prefer the one without (fresh accounts)
            if a1.ewma_latency == 0.0 and a2.ewma_latency > 0.0:
                return c1
            if a2.ewma_latency == 0.0 and a1.ewma_latency > 0.0:
                return c2
            # Add jitter to avoid stampede
            jitter1 = a1.ewma_latency * (1 + _rnd.uniform(-0.1, 0.1))
            jitter2 = a2.ewma_latency * (1 + _rnd.uniform(-0.1, 0.1))
            return c1 if jitter1 <= jitter2 else c2

    def mark_rate_limited(self, account_index: int) -> None:
        import time as _t
        with self._lock:
            self._rate_limited_accounts[account_index] = _t.time()
            if account_index < len(self._accounts):
                self._accounts[account_index].cached_token = None
            self.logger.warning("账号限速 index=%s 冷却=%.0fs", account_index, self._rate_limit_cooldown)

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
        """Prewarm guest tokens in parallel. ponytail: 60 accounts × 2s sequential = 120s;
        10 concurrent workers = ~6s total."""
        guests = [i for i, a in enumerate(self._accounts) if a.is_guest and not a.cached_token]
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

    def get_rate_limited_count(self) -> int:
        import time as _t
        with self._lock:
            now = _t.time()
            return sum(1 for ts in self._rate_limited_accounts.values() if now - ts < self._rate_limit_cooldown)

    def _get_access_token_for_index(self, account_index: int) -> str:
        account = self._accounts[account_index]
        if account.cached_token and time.time() < account.cached_token.expires_at - 60:
            self.logger.debug("使用缓存 access_token account=%s 剩余=%.0fs", account_index, account.cached_token.expires_at - time.time())
            return account.cached_token.access_token
        account.cached_token = self._refresh_access_token(account_index)
        return account.cached_token.access_token

    def _refresh_access_token(self, account_index: int) -> AccessToken:
        account = self._accounts[account_index]
        if account.is_guest or not account.refresh_token:
            return self._fetch_guest_access_token(account_index)
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

    def _fetch_guest_access_token(self, account_index: int, rate_limit: bool = True) -> AccessToken:
        import time as _t
        if rate_limit:
            with self._lock:
                elapsed = _t.time() - self._last_guest_fetch
                if elapsed < self._guest_fetch_interval:
                    import random as _r
                    _t.sleep(self._guest_fetch_interval - elapsed + _r.uniform(0, 0.5))
                self._last_guest_fetch = _t.time()
        account = self._accounts[account_index]
        timestamp, nonce, sign = build_sign()
        request_id = uuid.uuid4().hex
        # ponytail: match JS — X-Device-Id is persistent per account (not random each call)
        if not account._device_id:
            account._device_id = uuid.uuid4().hex
        device_id = account._device_id
        request = urllib.request.Request(
            self.config.guest_refresh_url,
            data=b"",
            method="POST",
            headers={
                # Exact headers from website JS: Content-Type with charset, app version, no X-Forwarded-For
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
                "X-Nonce": nonce,
                "X-Request-Id": request_id,
                "X-Sign": sign,
                "X-Timestamp": timestamp,
                "X-Lang": "en",
            },
        )
        debug_dump(self.logger, self.config.debug_dump_all, f"GLM 游客 token 请求头 account={account_index}", dict(request.header_items()))
        debug_dump(self.logger, self.config.debug_dump_all, f"GLM 游客 token 请求体 account={account_index}", b"")

        # ponytail: retry with exponential backoff on upstream errors
        max_attempts = 3
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
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
                self.record_upstream_success()
                return AccessToken(
                    access_token=str(access_token),
                    refresh_token=str(refresh_token),
                    expires_at=time.time() + ACCESS_TOKEN_EXPIRES_SECONDS - random.randint(10, 30),
                )
            except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, OSError) as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    delay = min(30, (2 ** attempt) * 5)  # 5s, 10s, 20s cap at 30
                    self.logger.warning(
                        "获取游客 token 失败 attempt=%s/%s account=%s delay=%ss error=%s",
                        attempt + 1, max_attempts, account_index, delay, exc,
                    )
                    _t.sleep(delay)
                    # Generate fresh sign and device for each retry
                    timestamp, nonce, sign = build_sign()
                    request.headers["X-Timestamp"] = timestamp
                    request.headers["X-Nonce"] = nonce
                    request.headers["X-Sign"] = sign
                    request.headers["X-Device-Id"] = uuid.uuid4().hex
                    request.headers["X-Request-Id"] = uuid.uuid4().hex
                else:
                    self.logger.error(
                        "获取游客 token 最终失败 account=%s after %s attempts: %s",
                        account_index, max_attempts, exc,
                    )

        # ponytail: circuit breaker — record persistent failure/success
        if last_exc is not None:
            self.record_upstream_failure()
        raise last_exc  # type: ignore[misc]

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

    def is_silent(self, account_index: int) -> bool:
        import time as _t
        with self._lock:
            ts = self._silent_accounts.get(account_index)
            if ts is None:
                return False
            if _t.time() - ts > self._silence_cooldown:
                del self._silent_accounts[account_index]
                return False
            return True

    def is_starved(self) -> bool:
        with self._lock:
            active = sum(1 for i, a in enumerate(self._accounts)
                         if a.cached_token is not None
                         and i not in self._rate_limited_accounts
                         and i not in self._silent_accounts)
            return active == 0 and len(self._accounts) > 0

    def get_consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    def is_account_usable(self, account_index: int) -> bool:
        with self._lock:
            if account_index in self._rate_limited_accounts:
                import time as _t
                if _t.time() - self._rate_limited_accounts[account_index] < self._rate_limit_cooldown:
                    return False
            if account_index in self._silent_accounts:
                import time as _t
                if _t.time() - self._silent_accounts[account_index] < self._silence_cooldown:
                    return False
            if account_index >= len(self._accounts):
                return False
            return True

    def acquire_account_for_stream(self, preferred_account_index: int | None = None) -> tuple[int, str]:
        account_count = self.get_account_count()
        if account_count <= 0:
            raise RuntimeError("没有可用的 GLM 账号")

        start = preferred_account_index % account_count if preferred_account_index is not None else self.get_next_account_index()

        for offset in range(account_count):
            idx = (start + offset) % account_count
            if self.is_account_usable(idx):
                token = self.get_access_token_for_account(idx)
                return idx, token

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

    def get_best_account(self) -> int | None:
        """Power-of-two-choices with jitter: pick 2 random usable accounts,
        select the one with lower EWMA latency. Prevents stampedes."""
        with self._lock:
            import random as _rnd
            usable = [i for i in range(len(self._accounts)) if self.is_account_usable(i)]
            if not usable:
                return 0 if self._accounts else None
            if len(usable) == 1:
                return usable[0]
            # Pick 2 random choices
            c1 = _rnd.choice(usable)
            c2 = _rnd.choice(usable)
            a1 = self._accounts[c1]
            a2 = self._accounts[c2]
            # Add ±10% jitter to EWMA for comparison
            jitter1 = a1.ewma_latency * (1 + _rnd.uniform(-0.1, 0.1))
            jitter2 = a2.ewma_latency * (1 + _rnd.uniform(-0.1, 0.1))
            return c1 if jitter1 <= jitter2 else c2

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def clear_silent_account(self, account_index: int) -> None:
        with self._lock:
            self._silent_accounts.pop(account_index, None)

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
