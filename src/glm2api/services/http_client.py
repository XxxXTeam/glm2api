"""Unified HTTP client using curl_cffi for browser-perfect TLS fingerprinting.

Replaces the dual httpx+urllib client systems in glm_auth.py and glm_client.py.
curl_cffi impersonates Chrome's JA3/JA4 TLS fingerprint, bypassing Alibaba Cloud
WAF bot detection that blocks Python's default TLS fingerprint.
"""
from __future__ import annotations

import gzip
import json
import threading
from io import BytesIO
from logging import Logger
from urllib.response import addinfourl


# Singleton client pool: keyed by proxy URL
_clients: dict[str, object] = {}
_clients_lock = threading.Lock()

IMPERSONATE_TARGET = "chrome120"  # Match GLM's Edge 143 / Chrome 143


def get_client(proxy_url: str | None = None, logger: Logger | None = None) -> object:
    """Get or create a cached curl_cffi Session for the given proxy.

    Each proxy gets its own persistent Session for connection reuse.
    Direct connections use key "__direct__".

    Returns a curl_cffi.requests.Session with Chrome 120 impersonation.
    """
    key = proxy_url or "__direct__"
    if key in _clients:
        return _clients[key]
    with _clients_lock:
        if key in _clients:
            return _clients[key]
        try:
            from curl_cffi import requests as cffi_requests
        except ImportError:
            if logger:
                logger.warning("curl_cffi not installed, falling back to httpx")
            return _get_httpx_fallback(proxy_url)

        proxies = {"https": proxy_url, "http": proxy_url} if proxy_url else None
        client = cffi_requests.Session(
            impersonate=IMPERSONATE_TARGET,
            proxies=proxies,
            timeout=120,
            allow_redirects=True,
        )
        # No warmup GET here — Session state (cookies) would interfere with
        # subsequent POST requests (guest token endpoint returns 405 otherwise).
        # The warmup is done separately via _warm_connection in glm_auth.py.
        _clients[key] = client
        return client


def _get_httpx_fallback(proxy_url: str | None = None):
    """Fallback httpx client if curl_cffi is unavailable."""
    import httpx
    import ssl

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    limits = httpx.Limits(max_keepalive_connections=100, max_connections=200, keepalive_expiry=120)
    kwargs = {"verify": ctx, "http2": False, "limits": limits, "timeout": httpx.Timeout(300, connect=10), "follow_redirects": False}
    if proxy_url:
        kwargs["proxy"] = proxy_url
    return httpx.Client(**kwargs)


class StreamingResponseWrapper:
    """Wraps a curl_cffi streaming response to provide urllib-compatible read().

    Reads complete lines from iter_lines() and returns them immediately.
    The _iter_sse_events method in glm_client.py expects read(n) to return
    whatever bytes are available without waiting to fill the full n buffer.
    """
    def __init__(self, curl_response):
        self._curl_resp = curl_response
        self._buffer = b""
        self._line_iter = curl_response.iter_lines()
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
            chunks = [self._buffer]
            self._buffer = b""
            try:
                for line in self._line_iter:
                    if isinstance(line, bytes):
                        chunks.append(line + b"\n")
                    else:
                        chunks.append(line.encode("utf-8", errors="replace") + b"\n")
            except Exception:
                pass
            return b"".join(chunks)

        # If buffer has data, return it immediately without waiting for more
        if self._buffer:
            if len(self._buffer) <= n:
                result = self._buffer
                self._buffer = b""
            else:
                result = self._buffer[:n]
                self._buffer = self._buffer[n:]
            return result

        # Read exactly ONE line from iter_lines and return it (with \n restored)
        # _iter_sse_events will accumulate lines in its pending buffer
        try:
            line = next(self._line_iter)
            if isinstance(line, bytes):
                self._buffer = line + b"\n"
            else:
                self._buffer = line.encode("utf-8", errors="replace") + b"\n"
            if len(self._buffer) <= n:
                result = self._buffer
                self._buffer = b""
            else:
                result = self._buffer[:n]
                self._buffer = self._buffer[n:]
            return result
        except StopIteration:
            return b""
        except Exception:
            return b""

    def close(self):
        try:
            self._curl_resp.close()
        except Exception:
            pass


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
    """Execute an HTTP request via curl_cffi with browser fingerprinting.

    When stream=True, returns a StreamingResponseWrapper that supports incremental
    read() calls for SSE parsing. When stream=False (default), returns a urllib-
    compatible addinfourl with fully buffered content.
    """
    client = get_client(proxy_url, logger=logger)

    response = client.request(
        method=method,
        url=url,
        headers=headers,
        data=data if data else None,
        stream=stream,
        timeout=timeout,
    )

    if stream:
        # Return streaming wrapper - caller reads chunks incrementally
        return StreamingResponseWrapper(response)

    # Buffered mode: read all content at once
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

    result = addinfourl(BytesIO(content), response.headers, url, code=response.status_code)
    result.msg = response.reason if hasattr(response, "reason") else ""
    # addinfourl has .code/.status but not .status_code; add compat alias
    result.status_code = response.status_code
    return result


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
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {"message": raw.decode("utf-8", errors="ignore")}
    return (result.status_code, payload if isinstance(payload, dict) else {"data": payload})


def do_request_oneshot(
    method: str,
    url: str,
    headers: dict[str, str],
    data: bytes | None = None,
    proxy_url: str | None = None,
    timeout: float = 300,
    logger: Logger | None = None,
):
    """Execute a one-shot HTTP request without caching the Session.
    
    Use for warmup/heartbeat requests that should not pollute the cached Session
    state (cookies, headers) used by main API calls.
    """
    try:
        from curl_cffi import requests as cffi_requests
        proxies = {"https": proxy_url, "http": proxy_url} if proxy_url else None
        client = cffi_requests.Session(
            impersonate=IMPERSONATE_TARGET,
            proxies=proxies,
            timeout=timeout,
            allow_redirects=True,
        )
        resp = client.request(method=method, url=url, headers=headers, data=data if data else None)
        client.close()
        content = resp.content
        content_encoding = ""
        try:
            content_encoding = resp.headers.get("Content-Encoding", "").lower()
        except Exception:
            pass
        if content_encoding == "gzip":
            try:
                content = gzip.decompress(content)
            except Exception:
                pass
        result = addinfourl(BytesIO(content), resp.headers, url, code=resp.status_code)
        result.msg = resp.reason if hasattr(resp, "reason") else ""
        result.status_code = resp.status_code
        return result
    except ImportError:
        if logger:
            logger.warning("curl_cffi not installed for one-shot request")
        client = _get_httpx_fallback(None)
        resp = client.request(method=method, url=url, headers=headers, content=data)
        client.close()
        content = resp.content
        result = addinfourl(BytesIO(content), resp.headers, url, code=resp.status_code)
        result.status_code = resp.status_code
        return result


def close_all() -> None:
    """Close all cached clients. Called on application shutdown."""
    with _clients_lock:
        for client in _clients.values():
            try:
                client.close()
            except Exception:
                pass
        _clients.clear()
