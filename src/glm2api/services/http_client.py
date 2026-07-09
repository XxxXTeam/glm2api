"""Unified HTTP client using curl_cffi for browser-perfect TLS fingerprinting.

Replaces the dual httpx+urllib client systems in glm_auth.py and glm_client.py.
curl_cffi impersonates Chrome's JA3/JA4 TLS fingerprint, bypassing Alibaba Cloud
WAF bot detection that blocks Python's default TLS fingerprint.

Proxy strategy:
- All connections (direct, local VPN, remote SOCKS5): curl_cffi Session
  (cached per proxy URL, Chrome 120 TLS impersonation)
- Fallback: httpx[socks] only when curl_cffi is not installed

Tested with curl_cffi 0.15.0+ — works through remote SOCKS5 proxies with
all proxy dict formats (separate http/https, 'all' key, socks5h, env vars)
and multiple impersonate targets (chrome120, chrome131, safari15_3).
"""
from __future__ import annotations

import gzip
import json
import threading
import time
from io import BytesIO
from logging import Logger
from urllib.response import addinfourl

IMPERSONATE_TARGET = "chrome120"  # curl_cffi 0.15.0 supports up to chrome131, chrome120 is stable

# Overridable curl options via env for tuning without code changes
_OVERRIDE_CURL_OPTS = None
def _get_curl_opts() -> dict[int, int]:
    global _OVERRIDE_CURL_OPTS
    if _OVERRIDE_CURL_OPTS is None:
        _OVERRIDE_CURL_OPTS = {}
        raw = __import__("os").environ.get("GLM_CURL_OPTIONS", "").strip()
        if raw:
            for pair in raw.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    try:
                        _OVERRIDE_CURL_OPTS[int(k)] = int(v)
                    except ValueError:
                        pass
    return _OVERRIDE_CURL_OPTS


class SessionPool:
    """Thread-safe pool of curl_cffi Sessions per proxy URL.
    
    Each proxy URL gets up to POOL_SIZE cached sessions. Each session has a 
    per-session Lock for exclusive access (curl_cffi Sessions are not thread-safe).
    Warmup GET for WAF cookies happens on first creation per session.
    
    Protocol optimizations (all sessions):
      - HTTP/2 over TLS (CurlHttpVersion.V2TLS) for multiplexed streams
      - TCP_NODELAY — disable Nagle's algorithm for lower-latency SSE
      - TCP_FASTOPEN — reduce 1-RTT on connection setup
      - DNS cache TTL 300s — avoid redundant DNS lookups
    """
    
    POOL_SIZE = 10  # increased from 5 for higher concurrent reuse
    IMPERSONATE_TARGET = "chrome120"
    
    def __init__(self):
        self._pools: dict[str, list[tuple[object, threading.Lock]]] = {}
        self._lock = threading.Lock()
    
    def acquire(self, proxy_url: str | None, logger: Logger | None = None) -> tuple[object, threading.Lock]:
        """Get a session from the pool. Creates+warms if needed."""
        key = proxy_url or "__direct__"
        with self._lock:
            if key not in self._pools:
                self._pools[key] = []
            pool = self._pools[key]
            # Find unlocked session
            for i, (session, slock) in enumerate(pool):
                if slock.acquire(blocking=False):
                    return session, slock
            # Create new if pool not full
            if len(pool) < self.POOL_SIZE:
                session = self._create_session(proxy_url, logger)
                slock = threading.Lock()
                slock.acquire()
                pool.append((session, slock))
                return session, slock
        
        # Pool full — create temp session (not pooled, discarded after use)
        session = self._create_session(proxy_url, logger)
        temp_lock = threading.Lock()
        temp_lock.acquire()
        return session, temp_lock
    
    def release(self, proxy_url: str | None, session: object, lock: threading.Lock):
        """Return session to pool."""
        lock.release()
    
    def _create_session(self, proxy_url: str | None, logger: Logger | None = None) -> object:
        """Create a new curl_cffi Session with HTTP/2, TCP optimizations, DNS caching, and WAF warmup."""
        from curl_cffi import requests as cffi_requests
        from curl_cffi.const import CurlHttpVersion
        proxies = {"https": proxy_url, "http": proxy_url} if proxy_url else None
        base_opts = {
            121: 1,    # CURLOPT_TCP_NODELAY — disable Nagle for low-latency SSE
            92: 300,   # CURLOPT_DNS_CACHE_TIMEOUT — cache DNS for 5 minutes
            244: 1,    # CURLOPT_TCP_FASTOPEN — save 1 RTT on new connections
        }
        base_opts.update(_get_curl_opts())
        session = cffi_requests.Session(
            impersonate=self.IMPERSONATE_TARGET,
            proxies=proxies,
            timeout=(15, 120),  # (connect_timeout, read_timeout) — fast fail on connect hang
            allow_redirects=True,
            http_version=CurlHttpVersion.V2TLS,  # HTTP/2 multiplexing over TLS
            curl_options=base_opts,
        )
        # Warmup: GET chatglm.cn for WAF cookies (acw_tc, cdn_sec_tc)
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        for attempt in range(2):
            try:
                session.get(
                    "https://chatglm.cn/",
                    headers={
                        "User-Agent": user_agent,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    },
                    timeout=5,
                )
                break
            except Exception as warm_exc:
                if logger:
                    logger.debug("session warmup attempt %d failed: %s", attempt + 1, warm_exc)
                continue
        return session
    
    def close_all(self):
        """Close all cached sessions."""
        with self._lock:
            for key, pool in list(self._pools.items()):
                for session, slock in pool:
                    try:
                        session.close()
                    except Exception:
                        pass
                del self._pools[key]


_session_pool = SessionPool()


def get_client(proxy_url: str | None = None, logger: Logger | None = None) -> tuple[object, threading.Lock | None]:
    """Get a pooled curl_cffi Session for the given proxy.
    
    Returns (session, lock). Caller MUST call return_client() after use.
    Direct and SOCKS5 proxies: curl_cffi Chrome 120 TLS (WAF bypass).
    HTTP CONNECT proxies: httpx fallback (returns None lock, caller uses httpx).
    """
    # HTTP CONNECT or direct: httpx (curl_cffi doesn't support HTTP CONNECT)
    if not proxy_url or proxy_url.startswith("http://"):
        return _get_httpx_client(proxy_url), None
    
    try:
        from curl_cffi import requests as cffi_requests
        _HAS_CURL_CFFI = True
    except ImportError:
        _HAS_CURL_CFFI = False
    
    if not _HAS_CURL_CFFI:
        if logger:
            logger.warning("curl_cffi not installed, falling back to httpx for SOCKS5")
        return _get_httpx_client(proxy_url), None
    
    return _session_pool.acquire(proxy_url, logger)


def return_client(proxy_url: str | None, session: object, lock: threading.Lock | None):
    """Return a pooled session after use. No-op for httpx clients (lock is None)."""
    if lock is not None:
        _session_pool.release(proxy_url, session, lock)


# httpx client cache per proxy URL — ensures connection reuse across requests.
# httpx Clients are thread-safe for concurrent use, so no lock needed per client,
# just a simple dict with a lock for creation.
_httpx_clients: dict[str | None, object] = {}
_httpx_clients_lock = threading.Lock()


def _get_httpx_client(proxy_url: str | None = None):
    """Get a cached httpx Client for the given proxy.
    
    httpx Clients are thread-safe and maintain their own connection pool,
    so we cache one per proxy URL for connection reuse.
    
    Uses HTTP/2 (h2 available) with TLS 1.2 forced for WAF compatibility.
    """
    import httpx
    import ssl

    with _httpx_clients_lock:
        cached = _httpx_clients.get(proxy_url)
        if cached is not None:
            return cached

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    limits = httpx.Limits(max_keepalive_connections=100, max_connections=200, keepalive_expiry=120)
    kwargs = {
        "verify": ctx,
        "http2": True,     # HTTP/2 multiplexing via h2 (verified available)
        "limits": limits,
        "timeout": httpx.Timeout(300, connect=10),
        "follow_redirects": False,
    }
    if proxy_url:
        kwargs["proxy"] = proxy_url
    client = httpx.Client(**kwargs)

    with _httpx_clients_lock:
        # Double-check: another thread may have created a client while we were building ours
        cached = _httpx_clients.get(proxy_url)
        if cached is not None:
            try:
                client.close()
            except Exception:
                pass
            return cached
        _httpx_clients[proxy_url] = client

    return client


class StreamingResponseWrapper:
    """Wraps a curl_cffi streaming response to provide urllib-compatible read().

    Reads complete lines from iter_lines() and returns them immediately.
    The _iter_sse_events method in glm_client.py expects read(n) to return
    whatever bytes are available without waiting to fill the full n buffer.
    """
    def __init__(self, curl_response, proxy_url: str | None = None, session_lock: threading.Lock | None = None):
        self._curl_resp = curl_response
        self._proxy_url = proxy_url
        self._session_lock = session_lock
        self._buffer = bytearray()
        self._line_iter = curl_response.iter_lines()
        self._read_deadline: float | None = None
        self.headers = curl_response.headers
        self.code = curl_response.status_code
        self.status_code = curl_response.status_code
        self.msg = getattr(curl_response, "reason", "")
        self.fp = None

    def read(self, n: int = -1) -> bytes:
        """Read available bytes from the streaming response.

        Returns whatever is available — does NOT wait to fill n bytes.
        This matches the expectation of _iter_sse_events which reads
        chunks and splits on \\n\\n to find complete SSE events.
        """
        if n < 0:
            chunks = [bytes(self._buffer)]
            self._buffer.clear()
            try:
                for line in self._line_iter:
                    if isinstance(line, bytes):
                        chunks.append(line + b"\n")
                    else:
                        chunks.append(line.encode("utf-8", errors="replace") + b"\n")
            except Exception as exc:
                __import__("logging").getLogger(__name__).warning(
                    "StreamingResponseWrapper.read: line iteration failed: %s", exc
                )
            return b"".join(chunks)

        # If buffer has data, return it immediately without waiting for more
        if self._buffer:
            buf_len = len(self._buffer)
            if buf_len <= n:
                result = bytes(self._buffer)
                self._buffer.clear()
            else:
                result = bytes(self._buffer[:n])
                self._buffer = bytearray(self._buffer[n:])
            return result

        # Read exactly ONE line from iter_lines and return it (with \n restored)
        # _iter_sse_events will accumulate lines in its pending buffer
        try:
            line = next(self._line_iter)
            if self._read_deadline is None:
                self._read_deadline = time.monotonic() + 300
            elif time.monotonic() > self._read_deadline:
                return b""
            if isinstance(line, bytes):
                line_bytes = line + b"\n"
            else:
                line_bytes = line.encode("utf-8", errors="replace") + b"\n"
            line_len = len(line_bytes)
            if line_len <= n:
                return line_bytes
            self._buffer = bytearray(line_bytes[n:])
            return line_bytes[:n]
        except StopIteration:
            return b""
        except Exception as exc:
            __import__("logging").getLogger(__name__).warning(
                "StreamingResponseWrapper.read: next() failed: %s", exc
            )
            return b""

    def close(self):
        try:
            self._curl_resp.close()
        except Exception:
            pass
        if self._session_lock is not None:
            _session_pool.release(self._proxy_url, self._curl_resp, self._session_lock)
            self._session_lock = None


def _is_httpx_client(client) -> bool:
    """Detect if client is an httpx Client (vs curl_cffi Session)."""
    return type(client).__module__.startswith("httpx")


def do_request(
    method: str,
    url: str,
    headers: dict[str, str],
    data: bytes | None = None,
    proxy_url: str | None = None,
    timeout: float = 300,
    logger: Logger | None = None,
    stream: bool = False,
):
    """Execute an HTTP request with browser fingerprinting."""
    client, session_lock = get_client(proxy_url, logger=logger)
    is_httpx = _is_httpx_client(client)
    
    if is_httpx:
        # httpx Client: stream or non-stream
        if stream:
            import httpx as _httpx
            stream_timeout = _httpx.Timeout(timeout, connect=10, read=timeout)
            request = client.build_request(method, url, headers=headers, content=data or b"", timeout=stream_timeout)
            httpx_response = client.send(request, stream=True)
            return _HttpxStreamWrapper(httpx_response)
        else:
            response = client.request(method, url, headers=headers, content=data or b"", timeout=timeout)
            try:
                content = response.content
                content_encoding = response.headers.get("Content-Encoding", "").lower()
                if content_encoding == "gzip":
                    try:
                        content = gzip.decompress(content)
                    except Exception:
                        pass
                    # After gzip decompression, remove the Content-Encoding header so downstream
                    # code doesn't try to decompress again
                    resp_headers = dict(response.headers)
                    resp_headers.pop('Content-Encoding', None)
                else:
                    resp_headers = response.headers
                result = addinfourl(BytesIO(content), resp_headers, url, code=response.status_code)
                result.msg = response.reason_phrase if hasattr(response, "reason_phrase") else ""
                result.status_code = response.status_code
                return result
            finally:
                response.close()
    
    # curl_cffi Session
    try:
        response = client.request(
            method=method,
            url=url,
            headers=headers,
            data=data if data is not None else None,
            stream=stream,
            timeout=timeout,
        )
    except Exception:
        if session_lock is not None:
            _session_pool.release(proxy_url, client, session_lock)
        raise
    
    if stream:
        # Streaming: wrapper holds the lock, releases on close()
        return StreamingResponseWrapper(response, proxy_url, session_lock)
    
    # Buffered mode
    try:
        content = response.content
        content_encoding = ""
        try:
            content_encoding = response.headers.get("Content-Encoding", "").lower()
        except Exception:
            pass
        if content_encoding == "gzip":
            try:
                content = gzip.decompress(content)
            except Exception:
                pass
            # After gzip decompression, remove the Content-Encoding header so downstream
            # code doesn't try to decompress again
            resp_headers = dict(response.headers)
            resp_headers.pop('Content-Encoding', None)
        else:
            resp_headers = response.headers
        result = addinfourl(BytesIO(content), resp_headers, url, code=response.status_code)
        result.msg = response.reason if hasattr(response, "reason") else ""
        result.status_code = response.status_code
        return result
    finally:
        try:
            response.close()
        except Exception:
            pass
        if session_lock is not None:
            _session_pool.release(proxy_url, client, session_lock)


class _HttpxStreamWrapper:
    """Wraps an httpx streaming response to provide urllib-compatible incremental read().

    Provides iter_lines() and read() interface compatible with StreamingResponseWrapper
    and _iter_sse_events. Reads chunks incrementally from the httpx streaming response.

    Uses line-buffered reads to deliver SSE events immediately (matching
    StreamingResponseWrapper behavior) rather than coalescing many events into
    the large chunks that httpcore's internal read buffer (up to 64 KB) would
    otherwise produce.
    """
    def __init__(self, httpx_response):
        self._response = httpx_response
        self._raw_iter = httpx_response.iter_bytes()
        self._buf = bytearray()
        self._eof = False
        self.headers = httpx_response.headers
        self.status_code = httpx_response.status_code
        self.code = httpx_response.status_code
        self.msg = httpx_response.reason_phrase if hasattr(httpx_response, "reason_phrase") else ""
        self.fp = None

    def _fill(self) -> bool:
        """Read the next chunk from the upstream iterator into *self._buf*.

        Returns ``True`` if data was added, ``False`` on EOF.
        """
        if self._eof:
            return False
        try:
            chunk = next(self._raw_iter)
            self._buf.extend(chunk)
            return True
        except StopIteration:
            self._eof = True
            return bool(self._buf)
        except Exception as exc:
            __import__("logging").getLogger(__name__).warning(
                "_HttpxStreamWrapper._fill: %s", exc
            )
            return False

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            chunks = [bytes(self._buf)]
            self._buf.clear()
            for chunk in self._raw_iter:
                chunks.append(chunk)
            self._eof = True
            return b"".join(chunks)

        # Ensure we have at least one byte in the buffer.
        if not self._buf and not self._fill():
            return b""

        # Return at most one \n-delimited line (up to *n* bytes).
        # This mirrors StreamingResponseWrapper's line-at-a-time delivery,
        # preventing httpcore's large internal read buffer from coalescing
        # many SSE events into a single read() call.
        nl_pos = self._buf.find(b"\n")
        if nl_pos >= 0:
            end = nl_pos + 1  # include the \n
            if end <= n:
                result = bytes(self._buf[:end])
                self._buf = self._buf[end:]
                return result
            # Line is longer than *n* — return what fits.
            result = bytes(self._buf[:n])
            self._buf = self._buf[n:]
            return result

        # No newline in buffer — return everything available (partial line).
        buf_len = len(self._buf)
        if buf_len <= n:
            result = bytes(self._buf)
            self._buf.clear()
            return result
        result = bytes(self._buf[:n])
        self._buf = self._buf[n:]
        return result

    def close(self):
        try:
            self._response.close()
        except Exception as exc:
            __import__("logging").getLogger(__name__).warning(
                "_HttpxStreamWrapper.close: %s", exc
            )


def do_json_request(
    method: str,
    url: str,
    headers: dict[str, str],
    data: bytes | None = None,
    proxy_url: str | None = None,
    timeout: float = 300,
    logger: Logger | None = None,
) -> tuple[int, dict]:
    """Execute request and return (status_code, parsed_json_payload).

    Convenience wrapper for non-streaming requests that expect JSON responses.
    """
    result = do_request(method, url, headers, data, proxy_url, timeout, logger)
    raw = result.read()
    try:
        payload = json.loads(raw if isinstance(raw, bytes) else raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {"message": raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)}
    return (result.status_code, payload if isinstance(payload, dict) else {"data": payload})


def close_all() -> None:
    """Close all pooled curl_cffi sessions and cached httpx clients."""
    _session_pool.close_all()
    with _httpx_clients_lock:
        for proxy_url, client in list(_httpx_clients.items()):
            try:
                client.close()
            except Exception:
                pass
        _httpx_clients.clear()
