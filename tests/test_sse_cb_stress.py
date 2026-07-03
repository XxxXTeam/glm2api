"""Stress/unit tests for SSE parser (_iter_sse_events) and circuit breaker (GLMAccessTokenManager).

Covers:
- Normal and edge-case SSE parsing (frame splitting, timeouts, partial reads, buffer boundaries)
- Circuit breaker open/close/reset behavior
- Rate limiting, EWMA latency routing, account failure tracking
- Starvation detection, guest token prewarm/spawn, parallel refresh
- Boundary conditions: empty account pool, out-of-range indices, silence cooldown
"""

from __future__ import annotations

import errno
import http.client
import json
import socket
import threading
import time
import uuid
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from unittest.mock import MagicMock, patch, call, ANY

import pytest

from glm2api.services.glm_client import GLMWebClient, UpstreamAPIError, STREAM_READ_TIMEOUT
from glm2api.services.glm_auth import (
    GLMAccessTokenManager,
    UpstreamBlockedError,
    AccountState,
    AccessToken,
    build_sign,
)
from glm2api.config import GUEST_REFRESH_TOKEN_MARKER


# ===================================================================
# Helpers
# ===================================================================


class FakeLogger:
    """Silent logger that implements the Logger protocol."""
    def debug(self, *args: Any, **kwargs: Any) -> None: pass
    def info(self, *args: Any, **kwargs: Any) -> None: pass
    def warning(self, *args: Any, **kwargs: Any) -> None: pass
    def error(self, *args: Any, **kwargs: Any) -> None: pass


def make_config(
    glm_use_guest_refresh_token: bool = False,
    glm_max_concurrency: int = 3,
    glm_refresh_tokens: list[str] | None = None,
) -> object:
    """Build a minimal config SimpleNamespace for testing."""
    import types
    return types.SimpleNamespace(
        glm_user_agent="Mozilla/5.0",
        glm_use_guest_refresh_token=glm_use_guest_refresh_token,
        glm_refresh_tokens=glm_refresh_tokens or ["tok1", "tok2", "tok3"],
        glm_max_concurrency=glm_max_concurrency,
        refresh_url="https://chatglm.cn/refresh",
        guest_refresh_url="https://chatglm.cn/guest",
        request_timeout=30,
        debug_dump_all=False,
        token_file_path=None,
        env_file_path=None,
    )


def _make_manager(
    accounts: list[AccountState] | None = None,
    guest_mode: bool = False,
) -> GLMAccessTokenManager:
    """Construct a GLMAccessTokenManager via __new__ with full field init.

    Avoids calling __init__ so we don't trigger network calls or file I/O.
    """
    mgr = GLMAccessTokenManager.__new__(GLMAccessTokenManager)
    mgr.config = make_config(glm_use_guest_refresh_token=guest_mode)
    mgr.logger = FakeLogger()
    mgr._accounts = accounts if accounts is not None else [
        AccountState(refresh_token="tok1"),
        AccountState(refresh_token="tok2"),
    ]
    mgr._current_index = 0
    mgr._lock = threading.Lock()
    mgr._persist_lock = threading.Lock()
    mgr._round_robin_counter = 0
    mgr._rate_limited_accounts = {}
    mgr._rate_limit_cooldown = 60.0
    mgr._last_guest_fetch = 0.0
    mgr._guest_fetch_interval = 2.0
    mgr._consecutive_failures = 0
    mgr._starvation_threshold = 6
    mgr._silent_accounts = {}
    mgr._silence_cooldown = 120.0
    mgr._last_full_refresh_at = 0.0
    mgr._circuit_open = False
    mgr._circuit_opened_at = 0.0
    mgr._circuit_failures = 0
    mgr._circuit_threshold = 6
    mgr._circuit_cooldown = 30.0
    mgr._full_refresh_cooldown = 300.0
    mgr._account_fail_count = {}
    mgr._account_fail_threshold = 3
    return mgr


def _make_client() -> GLMWebClient:
    """Construct a GLMWebClient via __new__ with minimal setup."""
    c = GLMWebClient.__new__(GLMWebClient)
    c.config = make_config()
    c.logger = FakeLogger()
    return c


def _make_http_response(
    read_side_effect: list[bytes] | Callable | None = None,
    set_sock: bool = True,
) -> MagicMock:
    """Build a mock HTTPResponse with configurable .read() behaviour."""
    resp = MagicMock()
    if read_side_effect is not None:
        resp.read = MagicMock(side_effect=read_side_effect)
    if set_sock:
        resp.fp.raw._sock = MagicMock()
    return resp


def _collect_events(
    client: GLMWebClient,
    response: MagicMock,
    stream_timeout: int = 0,
) -> list[dict[str, Any]]:
    """Helper: iterate _iter_sse_events and collect non-None results."""
    return list(client._iter_sse_events(response, stream_timeout=stream_timeout))


# ===================================================================
# PART 1 — _iter_sse_events
# ===================================================================


class TestSseNormalParsing:
    """Basic SSE parsing scenarios."""

    @pytest.fixture
    def client(self) -> GLMWebClient:
        return _make_client()

    def test_single_event(self, client: GLMWebClient) -> None:
        resp = _make_http_response([b'data: {"content": "hello"}\n\n', b""])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == [{"content": "hello"}]

    def test_multiple_events_one_chunk(self, client: GLMWebClient) -> None:
        chunk = b"data: {}\n\ndata: {}\n\ndata: {}\n\n"
        resp = _make_http_response([chunk, b""])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert len(events) == 3

    def test_events_split_across_reads(self, client: GLMWebClient) -> None:
        """SSE frame boundary in the middle of a JSON token."""
        resp = _make_http_response([
            b'data: {"content": "hel',
            b'lo"}\n\n',
            b"",
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == [{"content": "hello"}]

    def test_multiple_splits_multi_event(self, client: GLMWebClient) -> None:
        resp = _make_http_response([
            b'data: {"i": 0}\n\ndata: {"i": 1}\n\ndata: {"i": ',
            b'2}\n\ndata: {"i": 3}\n\n',
            b"",
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert [e["i"] for e in events] == [0, 1, 2, 3]

    def test_many_fragments(self, client: GLMWebClient) -> None:
        parts = [b'data: {"n": "']
        for i in range(50):
            parts.append(str(i).encode())
        parts.append(b'"}\n\n')
        parts.append(b"")
        resp = _make_http_response(parts)
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert len(events) == 1
        assert events[0]["n"] == "012345678910111213141516171819202122232425262728293031323334353637383940414243444546474849"

    def test_pending_flush_at_end(self, client: GLMWebClient) -> None:
        """When connection drops without trailing \\n\\n, flush remaining data: lines."""
        resp = _make_http_response([
            b'data: {"a": 1}\n\ndata: {"b": 2}\n\ndata: {"c": 3}',
            b"",
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert len(events) == 3
        assert events[2] == {"c": 3}

    def test_pending_flush_with_done(self, client: GLMWebClient) -> None:
        data = b'data: {"a": 1}\n\ndata: [DONE]'
        resp = _make_http_response([data, b""])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == [{"a": 1}]

    def test_single_line_flush(self, client: GLMWebClient) -> None:
        resp = _make_http_response([b'data: {"x": 1}', b""])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == [{"x": 1}]


class TestSseEdgeCases:
    """Edge cases: sentinel, timeouts, malformed data, buffer boundary, carriage returns."""

    @pytest.fixture
    def client(self) -> GLMWebClient:
        return _make_client()

    def test_done_sentinel_stops_iteration(self, client: GLMWebClient) -> None:
        resp = _make_http_response([
            b'data: {"a": 1}\n\ndata: [DONE]\n\n'
            b'data: {"a": 2}\n\n',
            b"",
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert len(events) == 1
        assert events[0] == {"a": 1}

    def test_socket_timeout_raises_504(self, client: GLMWebClient) -> None:
        resp = _make_http_response()
        resp.read = MagicMock(side_effect=[
            b'data: {"a": 1}\n\n',
            socket.timeout("timed out"),
        ])
        with patch("select.select", return_value=([1], [], [])):
            with pytest.raises(UpstreamAPIError) as exc:
                _collect_events(client, resp)
        assert exc.value.status_code == 504
        assert "超时" in str(exc.value)

    def test_select_timeout_raises_504(self, client: GLMWebClient) -> None:
        """select() returns empty list => upstream silence."""
        resp = _make_http_response([b'data: {"a": 1}\n\n'])
        with patch("select.select", return_value=([], [], [])):
            with pytest.raises(UpstreamAPIError) as exc:
                _collect_events(client, resp)
        assert exc.value.status_code == 504

    def test_incomplete_read_recovers_partial(self, client: GLMWebClient) -> None:
        resp = _make_http_response()
        resp.read = MagicMock(side_effect=[
            b'data: {"a": 1}\n\ndata: {"b": 2}\n\n',
            http.client.IncompleteRead(b'data: {"c": 3}\n\n'),
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert len(events) == 3
        assert events[2] == {"c": 3}

    def test_incomplete_read_zero_partial(self, client: GLMWebClient) -> None:
        resp = _make_http_response()
        resp.read = MagicMock(side_effect=[http.client.IncompleteRead(b"")])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == []

    def test_incomplete_read_partial_then_stop(self, client: GLMWebClient) -> None:
        resp = _make_http_response()
        resp.read = MagicMock(side_effect=[
            b'data: {"a": 1}\n\n',
            http.client.IncompleteRead(b'data: {"b": 2}'),
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert len(events) == 2
        assert events[1] == {"b": 2}

    def test_invalid_json_skipped(self, client: GLMWebClient) -> None:
        resp = _make_http_response([
            b"data: not valid json\n\n"
            b'data: {"ok": true}\n\n',
            b"",
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == [{"ok": True}]

    def test_empty_data_lines_skipped(self, client: GLMWebClient) -> None:
        resp = _make_http_response([
            b"data: \n\ndata:\n\n" b'data: {"x": 1}\n\n',
            b"",
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == [{"x": 1}]

    def test_non_data_lines_skipped(self, client: GLMWebClient) -> None:
        """event:, retry:, :comment, id: lines should be ignored."""
        resp = _make_http_response([
            b"event: custom\nretry: 1000\n:comment\nid: 42\n"
            b'data: {"a": 1}\n\n'
            b'data: {"b": 2}\n\n',
            b"",
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == [{"a": 1}, {"b": 2}]

    def test_carriage_returns(self, client: GLMWebClient) -> None:
        resp = _make_http_response([
            b'data: {"a": 1}\r\n\r\n'
            b'data: {"b": 2}\r\n\r\n',
            b"",
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == [{"a": 1}, {"b": 2}]

    def test_mixed_line_endings(self, client: GLMWebClient) -> None:
        resp = _make_http_response([
            b'data: {"a": 1}\r\n\r\n'
            b'data: {"b": 2}\n\n'
            b'data: {"c": 3}\r\n\r\n',
            b"",
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert len(events) == 3

    def test_data_prefix_extra_whitespace(self, client: GLMWebClient) -> None:
        resp = _make_http_response([
            b"data:  {\"a\": 1}\n\n"
            b"data:  \n\n"
            b'data:{"b": 2}\n\n',
            b"",
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == [{"a": 1}, {"b": 2}]

    def test_4kb_buffer_boundary(self, client: GLMWebClient) -> None:
        """JSON payload large enough to span or equal the 4096-byte read size."""
        inner = "x" * 4075
        payload = f'{{"content": "{inner}"}}'
        data_bytes = f"data: {payload}\n\n".encode("utf-8")
        assert len(data_bytes) > 4096
        resp = _make_http_response([data_bytes, b""])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert len(events) == 1
        assert events[0] == json.loads(payload)

    def test_large_payload_100kb(self, client: GLMWebClient) -> None:
        inner = "x" * 100_000
        chunk1 = b'data: {"content": "'
        chunk2 = inner.encode("utf-8")
        chunk3 = b'"}\n\n'
        resp = _make_http_response([chunk1, chunk2, chunk3, b""])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert len(events) == 1
        assert events[0]["content"] == inner

    def test_pending_non_utf8_bytes(self, client: GLMWebClient) -> None:
        """Trailing invalid UTF-8 should not crash."""
        resp = _make_http_response([
            b'data: {"valid": 1}\n\n',
            b"\xff\xfe",
            b"", b"", b"",
        ])
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == [{"valid": 1}]

    def test_stream_timeout_applied_to_socket(self, client: GLMWebClient) -> None:
        sock = MagicMock()
        resp = _make_http_response([b"data: {}\n\n", b""])
        resp.fp.raw._sock = sock
        with patch("select.select", return_value=([1], [], [])):
            _collect_events(client, resp, stream_timeout=99)
        sock.settimeout.assert_called_once_with(99)

    def test_no_socket_path_works(self, client: GLMWebClient) -> None:
        """When no socket path is found, parsing should still work."""
        resp = MagicMock()
        resp.read = MagicMock(side_effect=[b'data: {"a": 1}\n\n', b""])
        del resp.fp  # no fp at all
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == [{"a": 1}]

    def test_socket_alt_path_buf_raw(self, client: GLMWebClient) -> None:
        """Exercises the second attr_path: ('fp', 'buf', 'raw', '_sock')."""
        sock = MagicMock()
        resp = MagicMock()
        resp.read = MagicMock(side_effect=[b"data: {}\n\n", b""])
        # Delete fp.raw to force fallthrough to fp.buf.raw path
        resp.fp.buf.raw._sock = sock
        del resp.fp.raw
        with patch("select.select", return_value=([1], [], [])):
            _collect_events(client, resp, stream_timeout=42)
        sock.settimeout.assert_called_once_with(42)

    def test_socket_alt_path_fp_sock(self, client: GLMWebClient) -> None:
        """Exercises the third attr_path: ('fp', '_sock')."""
        sock = MagicMock()
        resp = MagicMock()
        resp.read = MagicMock(side_effect=[b"data: {}\n\n", b""])
        # Delete fp.raw and fp.buf to force fallthrough to fp._sock path
        resp.fp._sock = sock
        del resp.fp.raw
        del resp.fp.buf
        with patch("select.select", return_value=([1], [], [])):
            _collect_events(client, resp, stream_timeout=42)
        sock.settimeout.assert_called_once_with(42)

    def test_select_oserror_falls_through(self, client: GLMWebClient) -> None:
        """When fileno raises OSError, select is skipped and read still works."""
        resp = _make_http_response(
            [b'data: {"a": 1}\n\n', b""],
            set_sock=False,
        )
        resp.fp = MagicMock()
        resp.fp.fileno.side_effect = OSError("not a socket")
        del resp.fp.raw  # ensure fileno discovery fails
        with patch("select.select", return_value=([1], [], [])):
            events = _collect_events(client, resp)
        assert events == [{"a": 1}]

    def test_select_called_with_fileno(self, client: GLMWebClient) -> None:
        resp = _make_http_response([b"data: {}\n\n", b""])
        resp.fileno.return_value = 42
        with patch("select.select", return_value=([1], [], [])) as sel:
            _collect_events(client, resp)
        sel.assert_called()
        args = sel.call_args[0]
        assert len(args[0]) == 1  # one fileno in the rlist


# ===================================================================
# PART 2 — Circuit Breaker
# ===================================================================


class TestCircuitBreaker:

    def test_opens_after_threshold_failures(self) -> None:
        mgr = _make_manager()
        for _ in range(mgr._circuit_threshold):
            assert not mgr._circuit_open
            mgr.record_upstream_failure()
        assert mgr._circuit_open
        assert mgr._circuit_failures == mgr._circuit_threshold

    def test_resets_on_success(self) -> None:
        mgr = _make_manager()
        for _ in range(mgr._circuit_threshold):
            mgr.record_upstream_failure()
        assert mgr._circuit_open

        mgr.record_upstream_success()
        assert not mgr._circuit_open
        assert mgr._circuit_failures == 0

    def test_upstream_blocked_error_raised_when_open(self) -> None:
        mgr = _make_manager()
        for _ in range(mgr._circuit_threshold):
            mgr.record_upstream_failure()
        mgr._accounts[0].cached_token = AccessToken("tok", "rtok", time.time() + 3600)

        with pytest.raises(UpstreamBlockedError):
            mgr.get_access_token_for_account(0)

    def test_normal_access_when_closed(self) -> None:
        mgr = _make_manager()
        mgr._accounts[0].cached_token = AccessToken("tok", "rtok", time.time() + 3600)
        token = mgr.get_access_token_for_account(0)
        assert token == "tok"

    def test_failure_not_reaching_threshold(self) -> None:
        mgr = _make_manager()
        for _ in range(mgr._circuit_threshold - 1):
            mgr.record_upstream_failure()
        assert not mgr._circuit_open
        assert mgr._circuit_failures == mgr._circuit_threshold - 1

    def test_success_after_partial_failures(self) -> None:
        mgr = _make_manager()
        mgr._circuit_failures = 3
        mgr.record_upstream_success()
        assert mgr._circuit_failures == 0
        assert not mgr._circuit_open

    def test_circuit_does_not_auto_close(self) -> None:
        """Circuit stays open until record_upstream_success is called."""
        mgr = _make_manager()
        for _ in range(mgr._circuit_threshold):
            mgr.record_upstream_failure()
        assert mgr._circuit_open
        # Even if cooldown has elapsed, circuit stays open
        mgr._circuit_opened_at = time.monotonic() - mgr._circuit_cooldown - 10
        assert mgr._circuit_open
        # Only a success resets it
        mgr.record_upstream_success()
        assert not mgr._circuit_open


# ===================================================================
# PART 3 — Rate Limiting & Cooldown
# ===================================================================


class TestRateLimiting:

    def test_mark_rate_limited(self) -> None:
        mgr = _make_manager()
        mgr.mark_rate_limited(1)
        assert 1 in mgr._rate_limited_accounts
        assert mgr._accounts[1].cached_token is None

    def test_get_next_account_index_skips_rate_limited(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
        ])
        mgr.mark_rate_limited(0)
        for _ in range(50):
            assert mgr.get_next_account_index() == 1

    def test_get_next_account_index_all_rate_limited(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
        ])
        mgr.mark_rate_limited(0)
        mgr.mark_rate_limited(1)
        idx = mgr.get_next_account_index()
        assert idx in (0, 1)

    def test_rate_limited_cleared_after_cooldown(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
        ])
        mgr.mark_rate_limited(0)
        mgr._rate_limited_accounts[0] = time.time() - 120  # past cooldown
        mgr.get_next_account_index()  # triggers pruning
        assert 0 not in mgr._rate_limited_accounts

    def test_get_rate_limited_count(self) -> None:
        mgr = _make_manager()
        assert mgr.get_rate_limited_count() == 0
        mgr.mark_rate_limited(0)
        assert mgr.get_rate_limited_count() == 1
        mgr.mark_rate_limited(1)
        assert mgr.get_rate_limited_count() == 2

    def test_is_account_usable_rate_limited(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        mgr.mark_rate_limited(0)
        assert not mgr.is_account_usable(0)

    def test_is_account_usable_out_of_range(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        assert not mgr.is_account_usable(99)


# ===================================================================
# PART 4 — EWMA Latency Routing
# ===================================================================


class TestEwma:

    def test_initial_value_is_first_sample(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        mgr.record_latency(0, 100.0)
        assert mgr._accounts[0].ewma_latency == 100.0

    def test_sliding_average(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        mgr.record_latency(0, 100.0)
        mgr.record_latency(0, 200.0)
        alpha = mgr._accounts[0].ewma_alpha
        expected = (1 - alpha) * 100.0 + alpha * 200.0
        assert mgr._accounts[0].ewma_latency == pytest.approx(expected)

    def test_multiple_samples(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        samples = [50.0, 100.0, 150.0, 200.0, 250.0]
        alpha = mgr._accounts[0].ewma_alpha
        ewma = 0.0
        for i, s in enumerate(samples):
            mgr.record_latency(0, s)
            ewma = s if i == 0 else (1 - alpha) * ewma + alpha * s
        assert mgr._accounts[0].ewma_latency == pytest.approx(ewma)

    def test_get_next_account_index_prefers_lower(self) -> None:
        """Probabilistic: verify the low-latency account wins overwhelmingly."""
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
            AccountState(refresh_token="tok3"),
        ])
        mgr._accounts[0].ewma_latency = 50.0
        mgr._accounts[1].ewma_latency = 500.0
        mgr._accounts[2].ewma_latency = 1000.0

        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(200):
            counts[mgr.get_next_account_index()] += 1
        # ponytail: power-of-two-choices with ±10% jitter means low-latency
        # wins more often than random, but not guaranteed >50% every run
        assert counts[0] >= counts[1] and counts[0] >= counts[2]  # low-latency should still be most common

    def test_fresh_account_preferred(self) -> None:
        """Account with ewma_latency == 0.0 (no data) is preferred over high-latency."""
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
        ])
        mgr._accounts[0].ewma_latency = 0.0  # fresh
        mgr._accounts[1].ewma_latency = 999.0
        counts = {0: 0, 1: 0}
        for _ in range(200):
            counts[mgr.get_next_account_index()] += 1
        assert counts[0] > 100


# ===================================================================
# PART 5 — Account Failure Tracking
# ===================================================================


class TestAccountFailureTracking:

    def test_track_failure_triggers_after_three(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
        ])
        mgr._accounts[0].cached_token = AccessToken("tok", "rtok", time.time() + 3600)

        assert not mgr.track_account_failure(0)
        assert mgr._account_fail_count[0] == 1

        assert not mgr.track_account_failure(0)
        assert mgr._account_fail_count[0] == 2

        result = mgr.track_account_failure(0)
        assert result  # threshold reached
        assert 0 in mgr._rate_limited_accounts
        assert mgr._accounts[0].cached_token is None
        assert 0 not in mgr._account_fail_count  # counter cleared

    def test_clear_account_failures(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
        ])
        mgr.track_account_failure(0)
        mgr.track_account_failure(0)
        assert mgr._account_fail_count[0] == 2

        mgr.clear_account_failures(0)
        assert 0 not in mgr._account_fail_count

    def test_clear_nonexistent_does_not_raise(self) -> None:
        mgr = _make_manager()
        mgr.clear_account_failures(99)
        mgr.clear_account_failures(0)  # key not present


# ===================================================================
# PART 6 — Starvation Detection
# ===================================================================


class TestStarvation:

    def test_not_starved_when_active(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
        ])
        mgr._accounts[0].cached_token = AccessToken("t", "r", time.time() + 3600)
        mgr._accounts[1].cached_token = AccessToken("t", "r", time.time() + 3600)
        assert not mgr.is_starved()

    def test_starved_when_all_tokens_none(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
        ])
        mgr._accounts[0].cached_token = None
        mgr._accounts[1].cached_token = None
        assert mgr.is_starved()

    def test_starved_when_all_rate_limited(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
        ])
        mgr._accounts[0].cached_token = AccessToken("t", "r", time.time() + 3600)
        mgr._accounts[1].cached_token = AccessToken("t", "r", time.time() + 3600)
        mgr._rate_limited_accounts[0] = time.time()
        mgr._rate_limited_accounts[1] = time.time()
        assert mgr.is_starved()

    def test_starved_when_all_silent(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
        ])
        mgr._accounts[0].cached_token = AccessToken("t", "r", time.time() + 3600)
        mgr._accounts[1].cached_token = AccessToken("t", "r", time.time() + 3600)
        mgr._silent_accounts[0] = time.time()
        mgr._silent_accounts[1] = time.time()
        assert mgr.is_starved()

    def test_not_starved_empty_pool(self) -> None:
        mgr = _make_manager(accounts=[])
        assert not mgr.is_starved()


# ===================================================================
# PART 7 — Guest Token Prewarm & Spawn
# ===================================================================


class TestPrewarmGuestTokens:

    def test_prewarm_next_guest_slot(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True),
            AccountState(refresh_token="", is_guest=True, cached_token=AccessToken("t", "r", time.time() + 3600)),
        ], guest_mode=True)
        token = AccessToken("prewarmed", "rtok", time.time() + 3600)
        with patch.object(mgr, "_fetch_guest_access_token", return_value=token):
            idx = mgr.prewarm_next_guest_slot()
        assert idx == 0
        assert mgr._accounts[0].cached_token is token

    def test_prewarm_no_lazy_slots(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True, cached_token=AccessToken("t", "r", time.time() + 3600)),
        ], guest_mode=True)
        with patch.object(mgr, "_fetch_guest_access_token") as mock:
            idx = mgr.prewarm_next_guest_slot()
        assert idx is None
        mock.assert_not_called()

    def test_prewarm_bulk(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True),
            AccountState(refresh_token="", is_guest=True, cached_token=AccessToken("t", "r", time.time() + 3600)),
        ], guest_mode=True)
        token = AccessToken("p", "r", time.time() + 3600)
        with patch.object(mgr, "_fetch_guest_access_token", return_value=token) as mock:
            mgr.prewarm_guest_tokens()
        mock.assert_called_once_with(0, False)
        assert mgr._accounts[0].cached_token is token

    def test_prewarm_skips_when_none_needed(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True, cached_token=AccessToken("t", "r", time.time() + 3600)),
        ], guest_mode=True)
        with patch.object(mgr, "_fetch_guest_access_token") as mock:
            mgr.prewarm_guest_tokens()
        mock.assert_not_called()

    def test_spare_guest_slot_available(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True),
        ], guest_mode=True)
        assert mgr.spare_guest_slot_available()

    def test_spare_guest_slot_not_available(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True, cached_token=AccessToken("t", "r", time.time() + 3600)),
        ], guest_mode=True)
        assert not mgr.spare_guest_slot_available()

    def test_spare_guest_slot_no_guests(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1", is_guest=False),
        ])
        assert not mgr.spare_guest_slot_available()


class TestSpawnFreshGuestAccount:

    def test_spawn_creates_and_fetches(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True),
        ], guest_mode=True)
        initial = len(mgr._accounts)
        mock_token = AccessToken("new_tok", "rtok", time.time() + 3600)

        with patch.object(mgr, "_fetch_guest_access_token", return_value=mock_token) as mock:
            idx = mgr.spawn_fresh_guest_account()
        assert idx == initial
        assert len(mgr._accounts) == initial + 1
        assert mgr._accounts[idx].is_guest
        assert mgr._accounts[idx].cached_token is mock_token
        mock.assert_called_once_with(idx)

    def test_spawn_fetch_failure_pops_account(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True),
        ], guest_mode=True)
        initial = len(mgr._accounts)

        with patch.object(mgr, "_fetch_guest_access_token", side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError):
                mgr.spawn_fresh_guest_account()
        assert len(mgr._accounts) == initial

    def test_hard_swap_account(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True),
        ], guest_mode=True)
        mock_token = AccessToken("swapped", "rtok", time.time() + 3600)
        with patch.object(mgr, "_fetch_guest_access_token", return_value=mock_token):
            idx = mgr.hard_swap_account(0)
        assert idx == 0
        assert mgr._accounts[0].cached_token is mock_token


class TestForceRefreshAllGuestTokens:

    def test_invalidates_guests_and_clears_state(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True),
            AccountState(refresh_token="", is_guest=True),
            AccountState(refresh_token="tok1", is_guest=False),
        ], guest_mode=True)
        mgr._accounts[0].cached_token = AccessToken("t1", "r", time.time() + 3600)
        mgr._accounts[1].cached_token = AccessToken("t2", "r", time.time() + 3600)
        mgr._accounts[2].cached_token = AccessToken("t3", "r", time.time() + 3600)
        mgr._rate_limited_accounts[0] = time.time()
        mgr._silent_accounts[1] = time.time()

        with patch.object(mgr, "_fetch_guest_access_token", return_value=None):
            mgr.force_refresh_all_guest_tokens()

        assert mgr._accounts[0].cached_token is None
        assert mgr._accounts[1].cached_token is None
        assert mgr._accounts[2].cached_token is not None  # not a guest
        assert mgr._rate_limited_accounts == {}
        assert mgr._silent_accounts == {}
        assert mgr._consecutive_failures == 0

    def test_respects_cooldown(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True),
        ], guest_mode=True)
        mgr._last_full_refresh_at = time.monotonic()
        with patch.object(mgr, "_fetch_guest_access_token") as mock:
            mgr.force_refresh_all_guest_tokens()
        mock.assert_not_called()

    def test_parallel_execution(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True),
            AccountState(refresh_token="", is_guest=True),
            AccountState(refresh_token="", is_guest=True),
        ], guest_mode=True)
        mgr._last_full_refresh_at = 0.0

        with patch.object(mgr, "_fetch_guest_access_token", return_value=None) as mock:
            mgr.force_refresh_all_guest_tokens()
        assert mock.call_count == 3


# ===================================================================
# PART 8 — Account State Helpers
# ===================================================================


class TestAccountStateHelpers:

    def test_record_tool_call(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        mgr.record_tool_call(0)
        assert mgr._accounts[0].tool_call_count == 1

    def test_record_search(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        mgr.record_search(0)
        assert mgr._accounts[0].search_count == 1

    def test_get_usage(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        mgr.record_tool_call(0)
        mgr.record_search(0)
        assert mgr.get_usage(0) == (1, 1)
        assert mgr.get_usage(99) == (0, 0)

    def test_is_near_quota_limit(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="", is_guest=True)])
        assert not mgr.is_near_quota_limit(0)
        mgr._accounts[0].tool_call_count = 7
        assert mgr.is_near_quota_limit(0)
        mgr._accounts[0].tool_call_count = 0
        mgr._accounts[0].search_count = 5
        assert mgr.is_near_quota_limit(0)

    def test_near_quota_limit_non_guest(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1", is_guest=False)])
        mgr._accounts[0].tool_call_count = 99
        assert not mgr.is_near_quota_limit(0)

    def test_advance_account(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
            AccountState(refresh_token="tok3"),
        ])
        mgr._current_index = 0
        assert mgr.advance_account(0, "test") == 1
        assert mgr._current_index == 1

    def test_advance_account_wraps(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
        ])
        mgr._current_index = 1
        assert mgr.advance_account(1, "test") == 0

    def test_advance_account_mismatch(self) -> None:
        """If failed_index does not match current, return current unchanged."""
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
        ])
        mgr._current_index = 1
        assert mgr.advance_account(0, "test") == 1
        assert mgr._current_index == 1

    def test_invalidate_account(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        mgr._accounts[0].cached_token = AccessToken("t", "r", time.time() + 3600)
        mgr.invalidate_account(0)
        assert mgr._accounts[0].cached_token is None

    def test_reset_account_cycle(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
        ])
        mgr._current_index = 10
        mgr.reset_account_cycle()
        assert mgr._current_index == 0

    def test_get_current_account_index(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        assert mgr.get_current_account_index() == 0

    def test_record_success(self) -> None:
        mgr = _make_manager()
        mgr._consecutive_failures = 5
        mgr.record_success()
        assert mgr._consecutive_failures == 0

    def test_get_consecutive_failures(self) -> None:
        mgr = _make_manager()
        mgr._consecutive_failures = 3
        assert mgr.get_consecutive_failures() == 3

    def test_is_guest_account(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True),
            AccountState(refresh_token="tok1", is_guest=False),
        ])
        assert mgr.is_guest_account(0)
        assert not mgr.is_guest_account(1)

    def test_get_account_count(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
        ])
        assert mgr.get_account_count() == 1
        mgr._accounts = []
        assert mgr.get_account_count() == 0


# ===================================================================
# PART 9 — Silence Tracking
# ===================================================================


class TestSilenceTracking:

    def test_record_and_is_silent(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        mgr._accounts[0].cached_token = AccessToken("t", "r", time.time() + 3600)
        mgr.record_silence(0)
        assert mgr.is_silent(0)
        assert mgr._accounts[0].cached_token is None

    def test_silence_expires(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        mgr.record_silence(0)
        mgr._silent_accounts[0] = time.time() - 200  # past cooldown
        assert not mgr.is_silent(0)
        assert 0 not in mgr._silent_accounts

    def test_clear_silent(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        mgr.record_silence(0)
        assert mgr.is_silent(0)
        mgr.clear_silent_account(0)
        assert not mgr.is_silent(0)

    def test_is_silent_nonexistent(self) -> None:
        mgr = _make_manager(accounts=[AccountState(refresh_token="tok1")])
        assert not mgr.is_silent(99)


# ===================================================================
# PART 10 — should_switch_account
# ===================================================================


import urllib.error
import urllib.request
class TestShouldSwitchAccount:

    @pytest.fixture
    def mgr(self) -> GLMAccessTokenManager:
        return _make_manager()

    def test_socket_timeout(self, mgr: GLMAccessTokenManager) -> None:
        assert mgr.should_switch_account(socket.timeout())

    def test_timeout_error(self, mgr: GLMAccessTokenManager) -> None:
        assert mgr.should_switch_account(TimeoutError())

    def test_http_401(self, mgr: GLMAccessTokenManager) -> None:
        exc = urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, None)
        assert mgr.should_switch_account(exc)

    def test_http_403(self, mgr: GLMAccessTokenManager) -> None:
        exc = urllib.error.HTTPError("http://x", 403, "Forbidden", {}, None)
        assert mgr.should_switch_account(exc)

    def test_http_500_all_http_errors_are_switched(self, mgr: GLMAccessTokenManager) -> None:
        """HTTPError is subclass of URLError, so all HTTP errors trigger switch."""
        exc = urllib.error.HTTPError("http://x", 500, "Server Error", {}, None)
        assert mgr.should_switch_account(exc)
    def test_incomplete_read(self, mgr: GLMAccessTokenManager) -> None:
        assert mgr.should_switch_account(http.client.IncompleteRead(b""))

    def test_connection_error(self, mgr: GLMAccessTokenManager) -> None:
        assert mgr.should_switch_account(ConnectionError())

    def test_os_error_etimedout(self, mgr: GLMAccessTokenManager) -> None:
        assert mgr.should_switch_account(OSError(errno.ETIMEDOUT, "timed out"))

    def test_os_error_econnrefused(self, mgr: GLMAccessTokenManager) -> None:
        assert mgr.should_switch_account(OSError(errno.ECONNREFUSED, "refused"))

    def test_os_error_unrelated(self, mgr: GLMAccessTokenManager) -> None:
        assert not mgr.should_switch_account(OSError(errno.EACCES, "permission"))

    def test_urlerror(self, mgr: GLMAccessTokenManager) -> None:
        exc = urllib.error.URLError("reason")
        assert mgr.should_switch_account(exc)

    def test_upstream_api_error_429(self, mgr: GLMAccessTokenManager) -> None:
        exc = UpstreamAPIError(429, "rate limited")
        assert mgr.should_switch_account(exc)

    def test_upstream_api_error_502(self, mgr: GLMAccessTokenManager) -> None:
        exc = UpstreamAPIError(502, "bad gateway")
        assert mgr.should_switch_account(exc)

    def test_upstream_api_error_503(self, mgr: GLMAccessTokenManager) -> None:
        exc = UpstreamAPIError(503, "service unavailable")
        assert mgr.should_switch_account(exc)

    def test_upstream_api_error_other(self, mgr: GLMAccessTokenManager) -> None:
        exc = UpstreamAPIError(400, "bad request")
        assert not mgr.should_switch_account(exc)

    def test_runtime_error_token_related(self, mgr: GLMAccessTokenManager) -> None:
        assert mgr.should_switch_account(RuntimeError("token expired"))

    def test_runtime_error_unrelated(self, mgr: GLMAccessTokenManager) -> None:
        assert not mgr.should_switch_account(RuntimeError("something else"))


# ===================================================================
# PART 11 — Empty Pool Edge Cases
# ===================================================================


class TestEmptyPoolEdgeCases:

    def test_acquire_account_for_stream_raises(self) -> None:
        mgr = _make_manager(accounts=[])
        with pytest.raises(RuntimeError, match="没有可用的 GLM 账号"):
            mgr.acquire_account_for_stream()

    def test_get_best_account_returns_none(self) -> None:
        mgr = _make_manager(accounts=[])
        assert mgr.get_best_account() is None

    def test_get_next_account_index_returns_default(self) -> None:
        mgr = _make_manager(accounts=[])
        assert mgr.get_next_account_index() == 0

    def test_advance_account_raises(self) -> None:
        mgr = _make_manager(accounts=[])
        with pytest.raises(ZeroDivisionError):
            mgr.advance_account(0, "empty")

    def test_get_account_count_zero(self) -> None:
        mgr = _make_manager(accounts=[])
        assert mgr.get_account_count() == 0


# ===================================================================
# PART 12 — Acquire Account For Stream
# ===================================================================


class TestAcquireAccountForStream:

    def test_returns_first_usable_account(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
        ])
        mgr._accounts[0].cached_token = AccessToken("t1", "r1", time.time() + 3600)
        mgr._rate_limited_accounts[0] = time.time()  # make account 0 unusable
        mgr._accounts[1].cached_token = AccessToken("t2", "r2", time.time() + 3600)
        idx, token = mgr.acquire_account_for_stream()
        assert idx == 1
        assert token == "t2"

    def test_preferred_index(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="tok1"),
            AccountState(refresh_token="tok2"),
            AccountState(refresh_token="tok3"),
        ])
        for a in mgr._accounts:
            a.cached_token = AccessToken("t", "r", time.time() + 3600)
        idx, _ = mgr.acquire_account_for_stream(preferred_account_index=1)
        assert idx == 1

    def test_spawns_guest_when_all_blocked(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True),
            AccountState(refresh_token="", is_guest=True),
        ], guest_mode=True)
        mgr._rate_limited_accounts[0] = time.time()
        mgr._rate_limited_accounts[1] = time.time()
        mock_token = AccessToken("fresh", "r", time.time() + 3600)
        with patch.object(mgr, "_fetch_guest_access_token", return_value=mock_token):
            idx, token = mgr.acquire_account_for_stream()
        assert idx == 2
        assert token == "fresh"

    def test_falls_back_to_force_refresh_when_spawn_fails(self) -> None:
        mgr = _make_manager(accounts=[
            AccountState(refresh_token="", is_guest=True),
        ], guest_mode=True)
        mgr._rate_limited_accounts[0] = time.time()

        with patch.object(mgr, "_fetch_guest_access_token", side_effect=RuntimeError("fail")):
            with pytest.raises(RuntimeError, match="无可用账号"):
                mgr.acquire_account_for_stream()


# ===================================================================
# PART 13 — _init_guest_pool
# ===================================================================


class TestInitGuestPool:

    def test_guest_mode_creates_lazy_slots(self) -> None:
        config = make_config(glm_use_guest_refresh_token=True, glm_max_concurrency=5)
        mgr = GLMAccessTokenManager.__new__(GLMAccessTokenManager)
        mgr.config = config
        mgr.logger = FakeLogger()
        mgr._lock = threading.Lock()
        mgr._persist_lock = threading.Lock()
        mgr._init_guest_pool([GUEST_REFRESH_TOKEN_MARKER])
        assert len(mgr._accounts) == 5
        assert all(a.is_guest for a in mgr._accounts)
        assert all(a.cached_token is None for a in mgr._accounts)

    def test_non_guest_mode(self) -> None:
        config = make_config()
        mgr = GLMAccessTokenManager.__new__(GLMAccessTokenManager)
        mgr.config = config
        mgr.logger = FakeLogger()
        mgr._lock = threading.Lock()
        mgr._persist_lock = threading.Lock()
        mgr._init_guest_pool(["rtok1", "rtok2"])
        assert len(mgr._accounts) == 2
        assert not mgr._accounts[0].is_guest
        assert mgr._accounts[0].refresh_token == "rtok1"
