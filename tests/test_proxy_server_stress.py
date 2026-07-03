"""Comprehensive stress/unit tests for SmartProxyPool and server auth/validation layer.

Tests cover:
  - SmartProxyPool: init, population, scoring, blacklisting, health checks,
    concurrency, thread safety, auto-detection
  - Server auth/validation: _authorize, content-length, JSON parsing,
    unknown paths, CORS, SSE errors, safe_http_status
  - Stress/concurrency: 50+ concurrent operations on the proxy pool
"""

from __future__ import annotations

import json
import os
import threading
import time
from http import HTTPStatus
from io import BytesIO
from typing import Any

import pytest

from glm2api.services.glm2api_proxy import ProxyScore, SmartProxyPool
from glm2api.server import GLM2APIServer


# =========================================================================
#  Fixtures
# =========================================================================


@pytest.fixture
def clean_env():
    """Remove SmartProxyPool-relevant env vars before each test, then restore."""
    saved = {}
    keys = (
        "GLM_PROXY_LIST",
        "GLM_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
    ) + tuple(f"GLM_PROXY_{i}" for i in range(10))
    for k in keys:
        saved[k] = os.environ.pop(k, None)
    yield
    for k in keys:
        if saved[k] is not None:
            os.environ[k] = saved[k]
        else:
            os.environ.pop(k, None)


@pytest.fixture
def empty_pool(clean_env, monkeypatch):
    """Pool with no proxies configured (no env vars, no localhost ports)."""
    # Prevent socket-based auto-detection
    import socket as _s

    def _no_connect(*args, **kwargs):
        raise OSError("No proxy found")

    monkeypatch.setattr(_s, "create_connection", _no_connect)
    pool = SmartProxyPool()
    # Pool is populated in __init__; _populate() ran. Let's force clear.
    pool._proxies.clear()
    pool._current = ""
    return pool


@pytest.fixture
def pool_with_proxies(empty_pool):
    """Pool with two pre-configured proxies."""
    pool = empty_pool
    p1 = ProxyScore(url="http://proxy-a:8080")
    p2 = ProxyScore(url="http://proxy-b:8080")
    pool._proxies = {"http://proxy-a:8080": p1, "http://proxy-b:8080": p2}
    pool._current = "http://proxy-a:8080"
    return pool


@pytest.fixture
def mock_handler(monkeypatch):
    """Build a minimal RequestHandler-like object to test server helper methods."""
    from types import SimpleNamespace

    config = SimpleNamespace(
        host="127.0.0.1",
        port=0,
        api_prefix="/v1",
        cors_allow_origin="*",
        server_api_keys=[],
        debug_dump_all=False,
        exposed_models=["glm-4"],
    )
    logger = _FakeLogger()

    class FakeGLM:
        pass

    server = GLM2APIServer(config, FakeGLM(), logger)
    handler_cls = server._build_handler()

    handler = handler_cls.__new__(handler_cls)
    handler.server = server._server
    handler.command = "POST"
    handler.path = "/v1/chat/completions"
    handler.headers = _Headers({})
    handler.rfile = BytesIO()
    handler.wfile = BytesIO()
    handler.client_address = ("127.0.0.1", 54321)
    handler.request = _FakeSocket()
    handler.close_connection = False
    return {"handler": handler, "config": config, "logger": logger}


# =========================================================================
#  Helpers
# =========================================================================


class _FakeLogger:
    def debug(self, *a, **kw): pass
    def info(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass


class _Headers(dict):
    """Mock for self.headers that supports .get() with default."""
    def get(self, key, default=""):
        return super().get(key, default)


class _FakeSocket:
    """Minimal fake socket for handler metadata."""
    def getsockname(self):
        return ("127.0.0.1", 8000)
    def recv(self, bufsize, flags=0):
        raise BlockingIOError
    def settimeout(self, t): pass
    def setsockopt(self, *a): pass
    def close(self): pass


# =========================================================================
#  1. SmartProxyPool Tests
# =========================================================================


class TestSmartProxyPoolInit:
    """Pool initialisation edge-cases."""

    def test_initialize_empty_pool_returns_none(self, empty_pool):
        assert empty_pool.get_best() is None

    def test_get_status_empty(self, empty_pool):
        status = empty_pool.get_status()
        assert status["proxies"] == []

    def test_init_from_env_list(self, clean_env, monkeypatch):
        """GLM_PROXY_LIST populates the pool."""
        monkeypatch.setitem(os.environ, "GLM_PROXY_LIST", "http://a:1, http://b:2")
        pool = SmartProxyPool()
        assert len(pool._proxies) == 2
        assert "http://a:1" in pool._proxies
        assert "http://b:2" in pool._proxies

    def test_init_from_env_individual(self, clean_env, monkeypatch):
        """GLM_PROXY_0..GLM_PROXY_9 populate the pool."""
        monkeypatch.setitem(os.environ, "GLM_PROXY_0", "http://zero:0")
        monkeypatch.setitem(os.environ, "GLM_PROXY_1", "http://one:1")
        monkeypatch.setitem(os.environ, "GLM_PROXY_2", "http://two:2")
        pool = SmartProxyPool()
        assert len(pool._proxies) == 3
        assert "http://zero:0" in pool._proxies

    def test_init_from_fallback_env(self, clean_env, monkeypatch):
        """GLM_PROXY / HTTPS_PROXY / HTTP_PROXY fallback."""
        monkeypatch.setitem(os.environ, "HTTPS_PROXY", "http://fallback:3128")
        pool = SmartProxyPool()
        assert len(pool._proxies) == 1
        assert "http://fallback:3128" in pool._proxies

    def test_current_is_first_url(self, clean_env, monkeypatch):
        monkeypatch.setitem(os.environ, "GLM_PROXY_LIST", "http://a:1,http://b:2")
        pool = SmartProxyPool()
        assert pool._current == "http://a:1"

    def test_auto_detection_no_env_returns_empty(self, empty_pool):
        """When no env vars and no local proxies, pool stays empty."""
        assert empty_pool.get_best() is None

    def test_fetch_public_socks5_returns_proxies(self):
        """Fetch from public SOCKS5 lists returns socks5://ip:port URLs."""
        pool = SmartProxyPool.__new__(SmartProxyPool)
        urls = pool._fetch_public_socks5()
        assert len(urls) > 100  # expect at least 100 proxies
        # All should be socks5://ip:port format
        for url in urls[:10]:
            assert url.startswith("socks5://")
            assert ":" in url.replace("socks5://", "")
        # No duplicates
        assert len(urls) == len(set(urls))


class TestSmartProxyPoolScoring:
    """Score calculation and ranking."""

    def test_score_untested_is_100(self):
        p = ProxyScore(url="http://x:1")
        assert p.score == 100.0

    def test_score_formula_perfect(self):
        p = ProxyScore(url="http://x:1", successes=10, failures=0, total_calls=10, latency_ms=0, consec_failures=0)
        # rate=1.0 * 50 - 0 - 0 = 50
        assert p.score == 50.0

    def test_score_formula_with_latency(self):
        p = ProxyScore(url="http://x:1", successes=8, failures=2, total_calls=10, latency_ms=5000, consec_failures=1)
        # rate=0.8 * 50 = 40, lat_penalty=5, fail_penalty=5 => 30
        assert p.score == pytest.approx(30.0)

    def test_score_formula_with_consec_failures(self):
        p = ProxyScore(url="http://x:1", successes=5, failures=5, total_calls=10, latency_ms=0, consec_failures=3)
        # rate=0.5 * 50 = 25, lat_penalty=0, fail_penalty=15 => 10
        assert p.score == pytest.approx(10.0)

    def test_score_latency_penalty_capped_at_10(self):
        p = ProxyScore(url="http://x:1", successes=10, failures=0, total_calls=10, latency_ms=20_000, consec_failures=0)
        # lat_penalty = min(20, 10) = 10 => 40
        assert p.score == pytest.approx(40.0)

    def test_get_best_returns_highest_score(self, pool_with_proxies):
        pool = pool_with_proxies
        p_a = pool._proxies["http://proxy-a:8080"]
        p_b = pool._proxies["http://proxy-b:8080"]
        p_a.successes, p_a.total_calls = 1, 2
        p_a._score_cache = p_a._recompute_score()
        p_b.successes, p_b.total_calls = 5, 5
        p_b.latency_ms = 0
        p_b._score_cache = p_b._recompute_score()
        best = pool.get_best()
        assert best == "http://proxy-b:8080"

    def test_get_best_with_zero_successes_alive(self, pool_with_proxies):
        pool = pool_with_proxies
        p_a = pool._proxies["http://proxy-a:8080"]
        p_a.successes, p_a.failures, p_a.total_calls = 0, 3, 3
        p_a._score_cache = p_a._recompute_score()
        p_a.alive = True
        p_a.blacklisted_until = 0
        p_b = pool._proxies["http://proxy-b:8080"]
        p_b.successes, p_b.failures, p_b.total_calls = 1, 0, 1
        p_b._score_cache = p_b._recompute_score()
        p_b.alive = True
        p_b.blacklisted_until = 0
        best = pool.get_best()
        assert best == "http://proxy-b:8080"


class TestSmartProxyPoolReporting:
    """report_success / report_failure / report_rate_limited behaviour."""

    def test_report_success_updates_counters(self, pool_with_proxies):
        pool = pool_with_proxies
        pool.report_success("http://proxy-a:8080", latency=123.0)
        p = pool._proxies["http://proxy-a:8080"]
        assert p.successes == 1
        assert p.total_calls == 1
        assert p.consec_failures == 0
        assert p.latency_ms == 123.0
        assert p.alive is True

    def test_report_success_unknown_url_no_error(self, pool_with_proxies):
        pool = pool_with_proxies
        pool.report_success("http://nonexistent:9999")
        # Should not raise, just silently ignore

    def test_report_failure_increments_consec(self, pool_with_proxies):
        pool = pool_with_proxies
        pool.report_failure("http://proxy-a:8080")
        p = pool._proxies["http://proxy-a:8080"]
        assert p.failures == 1
        assert p.total_calls == 1
        assert p.consec_failures == 1
        assert p.alive is True  # Not blacklisted until >= 3

    def test_report_failure_triggers_blacklist_after_3(self, pool_with_proxies):
        pool = pool_with_proxies
        for _ in range(3):
            pool.report_failure("http://proxy-a:8080")
        p = pool._proxies["http://proxy-a:8080"]
        assert p.consec_failures == 3
        assert p.alive is False
        assert p.blacklisted_until > 0

    def test_report_rate_limited_blacklists_immediately(self, pool_with_proxies):
        pool = pool_with_proxies
        pool.report_rate_limited("http://proxy-a:8080")
        p = pool._proxies["http://proxy-a:8080"]
        assert p.alive is False
        assert p.blacklisted_until > time.monotonic()
        assert p.consec_failures >= 3

    def test_report_rate_limited_cooldown_doubles(self, pool_with_proxies):
        pool = pool_with_proxies
        pool.report_rate_limited("http://proxy-a:8080")
        p = pool._proxies["http://proxy-a:8080"]
        first_cooldown = p.blacklisted_until - time.monotonic()
        pool.report_rate_limited("http://proxy-a:8080")
        second_cooldown = p.blacklisted_until - time.monotonic()
        assert second_cooldown >= first_cooldown

    def test_report_success_resets_consec_failures(self, pool_with_proxies):
        pool = pool_with_proxies
        for _ in range(2):
            pool.report_failure("http://proxy-a:8080")
        pool.report_success("http://proxy-a:8080")
        p = pool._proxies["http://proxy-a:8080"]
        assert p.consec_failures == 0
        assert p.alive is True

    def test_report_failure_unknown_url_no_error(self, pool_with_proxies):
        pool = pool_with_proxies
        pool.report_failure("http://ghost:9")

    def test_report_rate_limited_unknown_url_no_error(self, pool_with_proxies):
        pool = pool_with_proxies
        pool.report_rate_limited("http://ghost:9")


class TestSmartProxyPoolBlacklist:
    """Blacklist and recovery logic."""

    def test_all_blacklisted_returns_soonest_to_recover(self, pool_with_proxies):
        pool = pool_with_proxies
        now = time.monotonic()
        p_a = pool._proxies["http://proxy-a:8080"]
        p_b = pool._proxies["http://proxy-b:8080"]
        p_a.alive = False; p_a.blacklisted_until = now + 100
        p_b.alive = False; p_b.blacklisted_until = now + 10
        best = pool.get_best()
        assert best == "http://proxy-b:8080"

    def test_get_best_revives_expired_blacklist(self, pool_with_proxies):
        pool = pool_with_proxies
        now = time.monotonic()
        p_a = pool._proxies["http://proxy-a:8080"]
        p_a.alive = False
        p_a.blacklisted_until = now - 1
        p_a.consec_failures = 5
        best = pool.get_best()
        assert best == "http://proxy-a:8080"
        assert p_a.alive is True
        assert p_a.blacklisted_until == 0
        assert p_a.consec_failures == 0

    def test_get_best_skips_blacklisted_alive_proxy(self, pool_with_proxies):
        pool = pool_with_proxies
        now = time.monotonic()
        p_a = pool._proxies["http://proxy-a:8080"]
        p_b = pool._proxies["http://proxy-b:8080"]
        p_a.alive = True
        p_a.blacklisted_until = now + 999
        p_b.alive = True
        p_b.blacklisted_until = 0
        p_b.successes, p_b.total_calls = 1, 1
        best = pool.get_best()
        assert best == "http://proxy-b:8080"

    def test_empty_pool_blacklist_edge(self, empty_pool):
        assert empty_pool.get_best() is None


class TestSmartProxyPoolHealthCheck:
    """Internal health check logic."""

    def test_health_check_alive_proxy_stays_alive(self, pool_with_proxies, monkeypatch):
        pool = pool_with_proxies
        import socket as _s

        def _ok(*args, **kwargs):
            return _FakeSocket()

        monkeypatch.setattr(_s, "create_connection", _ok)
        p = pool._proxies["http://proxy-a:8080"]
        p.consec_failures = 1
        p.alive = True
        pool._check_all()
        assert p.alive is True
        assert p.consec_failures == 0

    def test_health_check_dead_gets_blacklisted(self, pool_with_proxies, monkeypatch):
        pool = pool_with_proxies
        import socket as _s

        def _fail(*args, **kwargs):
            raise OSError("timeout")

        monkeypatch.setattr(_s, "create_connection", _fail)
        p = pool._proxies["http://proxy-a:8080"]
        p.consec_failures = 1
        p.alive = True
        pool._check_all()
        assert p.consec_failures >= 2
        if p.consec_failures >= 2:
            assert p.alive is False
            assert p.blacklisted_until > 0

    def test_health_check_skips_blacklisted(self, pool_with_proxies, monkeypatch):
        pool = pool_with_proxies
        import socket as _s
        calls = []

        def _track(*args, **kwargs):
            calls.append(args)
            raise OSError("fail")

        monkeypatch.setattr(_s, "create_connection", _track)
        # Blacklist BOTH proxies so neither gets checked
        now = time.monotonic()
        for p in pool._proxies.values():
            p.blacklisted_until = now + 3600
        pool._check_all()
        assert len(calls) == 0


# =========================================================================
#  2. Server auth/validation Tests
# =========================================================================


class TestServerAuthorize:
    """_authorize method under various configurations."""

    def test_authorize_no_keys_returns_true(self, mock_handler):
        config = mock_handler["config"]
        config.server_api_keys = []
        handler = mock_handler["handler"]
        assert handler._authorize() is True

    def test_authorize_bearer_matching(self, mock_handler):
        config = mock_handler["config"]
        config.server_api_keys = ["sk-abc123"]
        handler = mock_handler["handler"]
        handler.headers = _Headers({"Authorization": "Bearer sk-abc123"})
        assert handler._authorize() is True

    def test_authorize_bearer_not_matching(self, mock_handler):
        config = mock_handler["config"]
        config.server_api_keys = ["sk-abc123"]
        handler = mock_handler["handler"]
        handler.headers = _Headers({"Authorization": "Bearer sk-evil"})
        assert handler._authorize() is False

    def test_authorize_x_api_key_matching(self, mock_handler):
        config = mock_handler["config"]
        config.server_api_keys = ["sk-abc123"]
        handler = mock_handler["handler"]
        handler.headers = _Headers({"x-api-key": "sk-abc123"})
        assert handler._authorize() is True

    def test_authorize_x_api_key_not_matching(self, mock_handler):
        config = mock_handler["config"]
        config.server_api_keys = ["sk-abc123"]
        handler = mock_handler["handler"]
        handler.headers = _Headers({"x-api-key": "sk-evil"})
        assert handler._authorize() is False

    def test_authorize_bearer_wins_over_x_api_key(self, mock_handler):
        config = mock_handler["config"]
        config.server_api_keys = ["sk-bearer-good"]
        handler = mock_handler["handler"]
        handler.headers = _Headers({
            "Authorization": "Bearer sk-bearer-good",
            "x-api-key": "sk-api-key-bad",
        })
        assert handler._authorize() is True

    def test_authorize_x_api_key_when_bearer_missing(self, mock_handler):
        config = mock_handler["config"]
        config.server_api_keys = ["sk-abc123"]
        handler = mock_handler["handler"]
        handler.headers = _Headers({"x-api-key": "sk-abc123"})
        assert handler._authorize() is True

    def test_authorize_both_bad_returns_false(self, mock_handler):
        config = mock_handler["config"]
        config.server_api_keys = ["sk-abc123"]
        handler = mock_handler["handler"]
        handler.headers = _Headers({
            "Authorization": "Bearer sk-wrong",
            "x-api-key": "sk-also-wrong",
        })
        assert handler._authorize() is False

    def test_authorize_case_sensitive(self, mock_handler):
        config = mock_handler["config"]
        config.server_api_keys = ["Sk-Abc123"]
        handler = mock_handler["handler"]
        handler.headers = _Headers({"Authorization": "Bearer sk-abc123"})
        assert handler._authorize() is False

    def test_authorize_with_multiple_keys(self, mock_handler):
        config = mock_handler["config"]
        config.server_api_keys = ["key-one", "key-two", "key-three"]
        handler = mock_handler["handler"]
        handler.headers = _Headers({"Authorization": "Bearer key-two"})
        assert handler._authorize() is True
        handler.headers = _Headers({"Authorization": "Bearer key-four"})
        assert handler._authorize() is False


class TestServerValidation:
    """Request body validation and error responses."""

    def _do_post(self, mock_handler, body_bytes, path=None):
        handler = mock_handler["handler"]
        if path is not None:
            handler.path = path
        handler.rfile = BytesIO(body_bytes)
        handler.headers = _Headers({
            "Content-Length": str(len(body_bytes)),
            "Content-Type": "application/json",
        })
        handler.wfile = BytesIO()
        handler.close_connection = False
        handler.do_POST()
        return handler.wfile.getvalue()

    def _parse_response(self, raw: bytes):
        parts = raw.split(b"\r\n\r\n", 1)
        if len(parts) < 2:
            return raw, b""
        status_header = parts[0].split(b"\r\n")[0]
        body = parts[1]
        return status_header, body

    def test_invalid_json_body_400(self, mock_handler):
        raw = self._do_post(mock_handler, b"not json at all")
        assert b"400" in raw
        payload = json.loads(raw.split(b"\r\n\r\n", 1)[1].decode("utf-8"))
        assert "invalid_json" in payload.get("error", {}).get("type", "")

    def test_non_unicode_body_400(self, mock_handler):
        raw = self._do_post(mock_handler, b"\xff\xfe\x00\x01")
        assert b"400" in raw

    def test_body_not_dict_400(self, mock_handler):
        raw = self._do_post(mock_handler, b'"just a string"')
        assert b"400" in raw

    def test_missing_content_length_defaults_zero(self, mock_handler):
        handler = mock_handler["handler"]
        handler.path = "/v1/chat/completions"
        handler.headers = _Headers({"Content-Type": "application/json"})
        handler.rfile = BytesIO(b'{"model":"glm-4"}')
        handler.wfile = BytesIO()
        handler.close_connection = False
        handler.do_POST()
        raw = handler.wfile.getvalue()
        assert b"400" in raw
        assert b"messages" in raw

    def test_negative_content_length_400(self, mock_handler):
        handler = mock_handler["handler"]
        handler.path = "/v1/chat/completions"
        handler.headers = _Headers({"Content-Length": "-50", "Content-Type": "application/json"})
        handler.rfile = BytesIO(b'{"model":"glm-4","messages":[{"role":"user","content":"hi"}]}')
        handler.wfile = BytesIO()
        handler.close_connection = False
        handler.do_POST()
        raw = handler.wfile.getvalue()
        assert b"400" in raw
        assert b"Content-Length" in raw

    def test_missing_messages_field_400(self, mock_handler):
        raw = self._do_post(mock_handler, json.dumps({"model": "glm-4"}).encode("utf-8"))
        assert b"400" in raw

    def test_missing_model_field_400(self, mock_handler):
        raw = self._do_post(mock_handler, json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode("utf-8"))
        assert b"400" in raw

    def test_unknown_path_404(self, mock_handler):
        handler = mock_handler["handler"]
        handler.path = "/v1/unknown/endpoint"
        handler.headers = _Headers({"Content-Type": "application/json"})
        handler.rfile = BytesIO(b"{}")
        handler.wfile = BytesIO()
        handler.do_POST()
        raw = handler.wfile.getvalue()
        assert b"404" in raw

    def test_unknown_get_path_404(self, mock_handler):
        handler = mock_handler["handler"]
        handler.command = "GET"
        handler.path = "/v1/neverland"
        handler.headers = _Headers({})
        handler.rfile = BytesIO(b"")
        handler.wfile = BytesIO()
        handler.do_GET()
        raw = handler.wfile.getvalue()
        assert b"404" in raw


class TestServerCORS:
    """CORS headers on OPTIONS and common responses."""

    def test_options_returns_cors_headers(self, mock_handler):
        handler = mock_handler["handler"]
        handler.command = "OPTIONS"
        handler.headers = _Headers({})
        handler.rfile = BytesIO(b"")
        handler.wfile = BytesIO()
        handler.close_connection = False
        handler.do_OPTIONS()
        raw = handler.wfile.getvalue()
        assert b"204" in raw
        assert b"Access-Control-Allow-Origin" in raw
        assert b"Access-Control-Allow-Headers" in raw
        assert b"Access-Control-Allow-Methods" in raw

    def test_cors_origin_from_config(self, mock_handler):
        mock_handler["config"].cors_allow_origin = "https://example.com"
        handler = mock_handler["handler"]
        handler.command = "OPTIONS"
        handler.headers = _Headers({})
        handler.rfile = BytesIO(b"")
        handler.wfile = BytesIO()
        handler.close_connection = False
        handler.do_OPTIONS()
        raw = handler.wfile.getvalue()
        assert b"example.com" in raw

    def test_post_contains_cors_headers(self, mock_handler):
        handler = mock_handler["handler"]
        handler.command = "GET"
        handler.path = "/v1/models"
        handler.headers = _Headers({})
        handler.rfile = BytesIO(b"")
        handler.wfile = BytesIO()
        handler.close_connection = False
        handler.do_GET()
        raw = handler.wfile.getvalue()
        assert b"Access-Control-Allow-Origin" in raw


class TestServerMisc:
    """Miscellaneous helper methods."""

    def test_sse_error_writes_to_wfile(self, mock_handler):
        handler = mock_handler["handler"]
        handler.wfile = BytesIO()
        handler._write_sse_error("Something broke", "test_error")
        raw = handler.wfile.getvalue()
        assert b"data:" in raw
        assert b"Something broke" in raw
        assert b"test_error" in raw
        assert raw.endswith(b"\n\n")

    def test_sse_error_disconnect_does_not_raise(self, mock_handler):
        handler = mock_handler["handler"]

        class _BrokenWfile:
            def write(self, _b):
                raise BrokenPipeError("disconnected")
            def flush(self):
                pass

        handler.wfile = _BrokenWfile()
        handler._write_sse_error("err", "type")

    def test_safe_http_status_valid(self, mock_handler):
        handler = mock_handler["handler"]
        result = handler._safe_http_status(200, fallback=HTTPStatus.BAD_GATEWAY)
        assert result == HTTPStatus.OK

    def test_safe_http_status_valid_502(self, mock_handler):
        handler = mock_handler["handler"]
        result = handler._safe_http_status(502, fallback=HTTPStatus.BAD_GATEWAY)
        assert result == HTTPStatus.BAD_GATEWAY

    def test_safe_http_status_invalid_returns_fallback(self, mock_handler):
        handler = mock_handler["handler"]
        result = handler._safe_http_status(999, fallback=HTTPStatus.INTERNAL_SERVER_ERROR)
        assert result == HTTPStatus.INTERNAL_SERVER_ERROR

    def test_path_without_query_strips_params(self, mock_handler):
        handler = mock_handler["handler"]
        handler.path = "/v1/chat/completions?model=glm-4"
        assert handler._path_without_query() == "/v1/chat/completions"

    def test_path_without_query_no_params(self, mock_handler):
        handler = mock_handler["handler"]
        handler.path = "/v1/chat/completions"
        assert handler._path_without_query() == "/v1/chat/completions"

    def test_health_endpoint_returns_200(self, mock_handler):
        handler = mock_handler["handler"]
        handler.command = "GET"
        handler.path = "/health"
        handler.headers = _Headers({})
        handler.rfile = BytesIO(b"")
        handler.wfile = BytesIO()
        handler.close_connection = False
        handler.do_GET()
        raw = handler.wfile.getvalue()
        assert b"200" in raw
        assert b"status" in raw


# =========================================================================
#  3. Stress / Concurrency Tests
# =========================================================================


class TestStress:
    """High-concurrency stress scenarios."""

    def test_50_concurrent_report_success(self, pool_with_proxies):
        pool = pool_with_proxies
        n = 50
        barrier = threading.Barrier(n)
        errors = []

        def _work():
            barrier.wait()
            try:
                pool.report_success("http://proxy-a:8080", latency=1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_work) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        p = pool._proxies["http://proxy-a:8080"]
        assert p.successes == n
        assert p.total_calls == n

    def test_50_concurrent_get_best(self, pool_with_proxies):
        pool = pool_with_proxies
        n = 50
        barrier = threading.Barrier(n)
        results = []
        errors = []

        def _work():
            barrier.wait()
            try:
                r = pool.get_best()
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_work) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(results) == n
        assert all(r is not None for r in results)

    def test_100_threads_mixed_read_write(self, pool_with_proxies):
        pool = pool_with_proxies
        n = 100
        barrier = threading.Barrier(n)
        errors = []
        results = []

        def _worker(i):
            barrier.wait()
            try:
                for _ in range(20):
                    if i % 4 == 0:
                        pool.report_success("http://proxy-a:8080", latency=2.0)
                    elif i % 4 == 1:
                        pool.report_failure("http://proxy-b:8080")
                    elif i % 4 == 2:
                        pool.report_rate_limited("http://proxy-a:8080")
                    else:
                        r = pool.get_best()
                        results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0
        p_a = pool._proxies["http://proxy-a:8080"]
        p_b = pool._proxies["http://proxy-b:8080"]
        assert p_a.total_calls + p_b.total_calls > 0
        assert isinstance(p_a.score, float)
        assert isinstance(p_b.score, float)
        assert len(pool._proxies) == 2

    def test_50_concurrent_report_rate_limited(self, pool_with_proxies):
        pool = pool_with_proxies
        n = 50
        barrier = threading.Barrier(n)
        errors = []

        def _work():
            barrier.wait()
            try:
                pool.report_rate_limited("http://proxy-a:8080")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_work) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        p = pool._proxies["http://proxy-a:8080"]
        assert p.alive is False
        assert p.blacklisted_until > time.monotonic()

    def test_pool_survives_high_contention_with_get_status(self, pool_with_proxies):
        pool = pool_with_proxies
        n = 30
        barrier = threading.Barrier(n)
        errors = []

        def _writer():
            barrier.wait()
            for _ in range(50):
                pool.report_success("http://proxy-a:8080", latency=1.0)
                pool.report_failure("http://proxy-b:8080")

        def _reader():
            barrier.wait()
            for _ in range(50):
                try:
                    s = pool.get_status()
                    assert "proxies" in s
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=_writer) for _ in range(n // 2)]
        threads += [threading.Thread(target=_reader) for _ in range(n // 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0

    def test_stress_report_failure_accumulates_correctly(self, pool_with_proxies):
        pool = pool_with_proxies
        n = 50
        iterations = 10
        barrier = threading.Barrier(n)
        errors = []

        def _work():
            barrier.wait()
            for _ in range(iterations):
                try:
                    pool.report_failure("http://proxy-a:8080")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=_work) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0
        p = pool._proxies["http://proxy-a:8080"]
        expected_failures = n * iterations
        assert p.failures == expected_failures
        assert p.consec_failures == expected_failures
