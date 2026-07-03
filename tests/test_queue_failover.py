"""
Comprehensive stress/unit tests for ConcurrentRequestQueue and
GLMWebClient._call_with_account_failover.

Run:  python -m pytest tests/test_queue_failover.py -v

All mocks -- no real upstream calls.
"""

from __future__ import annotations

import logging
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock, call

import pytest

from glm2api.services.glm_client import (
    ConcurrentRequestQueue,
    QueueLease,
    QueueTimeoutError,
    UpstreamAPIError,
    GLMWebClient,
)
from glm2api.services.glm_auth import GLMAccessTokenManager, AccountState, AccessToken
from glm2api.config import AppConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logger() -> logging.Logger:
    logger = logging.getLogger("test_queue_failover")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _make_config(**overrides: object) -> AppConfig:
    """Build a minimal AppConfig with sensible test defaults."""
    from pathlib import Path
    defaults: dict[str, object] = dict(
        env_file_path=Path("/tmp/.env.test"),
        env_file_created=False,
        token_file_path=Path("/tmp/token.test.txt"),
        host="127.0.0.1",
        port=8000,
        api_prefix="/v1",
        log_level="DEBUG",
        debug_dump_all=False,
        request_timeout=30,
        glm_base_url="https://chatglm.cn/chatglm",
        glm_use_guest_refresh_token=True,
        glm_refresh_token="__glm_guest__",
        glm_refresh_tokens=["__glm_guest__", "__glm_guest__", "__glm_guest__"],
        glm_assistant_id="test_asst",
        glm_image_assistant_id="img_asst",
        glm_image_model_name="cogView-4-250304",
        glm_user_agent="test-agent",
        glm_delete_conversation=False,
        glm_max_concurrency=5,
        glm_queue_wait_timeout=10,
        glm_busy_max_retries=0,
        glm_busy_retry_interval=0.1,
        glm_guest_max_retries=0,
        blocked_tool_names=[],
        exposed_models=["glm-4-flash"],
        model_aliases={},
        server_api_keys=[],
        cors_allow_origin="*",
    )
    defaults.update(overrides)  # type: ignore[arg-type]
    return AppConfig(**defaults)  # type:ignore[arg-type]


def _make_client(logger: logging.Logger | None = None, config: AppConfig | None = None) -> GLMWebClient:
    """Return a GLMWebClient with a mocked auth manager."""
    logger = logger or _make_logger()
    config = config or _make_config()
    client = GLMWebClient(config=config, logger=logger)
    # Replace the real auth with a MagicMock so we never talk to upstream
    client.auth = MagicMock(spec=GLMAccessTokenManager)
    client.auth.get_account_count.return_value = 3
    client.auth.is_guest_account.return_value = True
    client.auth.get_access_token_for_account.return_value = "fake-token"
    client.auth.get_next_account_index.return_value = 0
    client.auth.should_switch_account.return_value = True
    client.auth.mark_rate_limited = MagicMock()
    client.auth.invalidate_account = MagicMock()
    client.auth.advance_account.return_value = 1
    client.auth.spawn_fresh_guest_account.return_value = 3
    client.auth.force_refresh_all_guest_tokens = MagicMock()
    client.auth.record_latency = MagicMock()
    client.auth.clear_account_failures = MagicMock()
    client.auth.track_account_failure.return_value = False
    client.auth.record_success = MagicMock()
    client.auth.reset_account_cycle = MagicMock()
    return client


# ===================================================================
# 1. ConcurrentRequestQueue
# ===================================================================

class TestConcurrentRequestQueue:
    """Tests for the per-account semaphore queue."""

    def make_queue(self, wait_timeout: int = 10, max_concurrency: int = 5) -> ConcurrentRequestQueue:
        return ConcurrentRequestQueue(
            logger=_make_logger(),
            wait_timeout=wait_timeout,
            max_concurrency=max_concurrency,
        )

    # ------------------------------------------------------------------
    # 1a. basic acquire + release
    # ------------------------------------------------------------------
    def test_acquire_release_basic(self) -> None:
        q = self.make_queue()
        lease = q.acquire("test-op", account_pool_size=3)
        assert isinstance(lease, QueueLease)
        assert not lease.released
        lease.release()
        assert lease.released
        # Releasing twice is a no-op
        lease.release()
        assert lease.released

    def test_acquire_returns_ticket_in_range(self) -> None:
        q = self.make_queue()
        q._ensure_accounts(3)
        lease = q.acquire("ticket-test", account_pool_size=3)
        assert 0 <= lease.ticket < len(q._sems)

    # ------------------------------------------------------------------
    # 1b. timeout raises QueueTimeoutError
    # ------------------------------------------------------------------
    def test_acquire_timeout_raises(self) -> None:
        q = self.make_queue(wait_timeout=0.3)
        # Exhaust all slots: 3 accounts × 3 slots = 9
        n_accounts = 3
        q._ensure_accounts(n_accounts)
        slots = n_accounts * ConcurrentRequestQueue.PER_ACCOUNT_LIMIT
        leases = [q.acquire("busy", account_pool_size=n_accounts) for _ in range(slots)]
        with pytest.raises(QueueTimeoutError, match="队列等待超时"):
            q.acquire("timeout-op", account_pool_size=n_accounts)
        # Cleanup
        for le in leases:
            le.release()

    def test_acquire_timeout_custom_timeout(self) -> None:
        q = self.make_queue(wait_timeout=0.2)
        n_accounts = 3
        q._ensure_accounts(n_accounts)
        slots = n_accounts * ConcurrentRequestQueue.PER_ACCOUNT_LIMIT
        leases = [q.acquire("busy", account_pool_size=n_accounts) for _ in range(slots)]
        t0 = time.monotonic()
        with pytest.raises(QueueTimeoutError):
            q.acquire("timeout-op", account_pool_size=n_accounts)
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0  # should fire well before 1s
        for le in leases:
            le.release()

    # ------------------------------------------------------------------
    # 1c. QueueTimeoutError raised correctly (also check it's a RuntimeError)
    # ------------------------------------------------------------------
    def test_queue_timeout_error_is_runtime_error(self) -> None:
        assert issubclass(QueueTimeoutError, RuntimeError)

    def test_queue_timeout_error_message(self) -> None:
        q = self.make_queue(wait_timeout=0.2)
        n_accounts = 3
        q._ensure_accounts(n_accounts)
        slots = n_accounts * ConcurrentRequestQueue.PER_ACCOUNT_LIMIT
        leases = [q.acquire("x", account_pool_size=n_accounts) for _ in range(slots)]
        with pytest.raises(QueueTimeoutError) as exc:
            q.acquire("y", account_pool_size=n_accounts)
        assert "队列等待超时" in str(exc.value)
        for le in leases:
            le.release()

    # ------------------------------------------------------------------
    # 1d. thread safety: multiple concurrent acquires
    # ------------------------------------------------------------------
    def test_concurrent_acquires_thread_safety(self) -> None:
        """Spawn N threads that all acquire+release. No deadlock, no exception."""
        q = self.make_queue(wait_timeout=5)
        n_threads = 10
        results: list[Exception | None] = [None] * n_threads
        lock = threading.Lock()

        def worker(idx: int) -> None:
            try:
                lease = q.acquire(f"worker-{idx}", account_pool_size=3)
                time.sleep(0.02)  # hold briefly
                lease.release()
            except Exception as exc:
                with lock:
                    results[idx] = exc

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        failures = [r for r in results if r is not None]
        assert not failures, f"{len(failures)} threads raised: {failures}"

    def test_concurrent_acquire_honours_per_account_limit(self) -> None:
        """Even with many threads, no account gets more than PER_ACCOUNT_LIMIT leases."""
        q = self.make_queue(wait_timeout=2)
        n_accounts = 3
        q._ensure_accounts(n_accounts)
        # Acquire all slots (3 per account = 9 total)
        leases: list[QueueLease] = []
        for _ in range(n_accounts * ConcurrentRequestQueue.PER_ACCOUNT_LIMIT):
            leases.append(q.acquire("saturation", account_pool_size=n_accounts))
        # Now all should be busy; this must timeout
        with pytest.raises(QueueTimeoutError):
            q.acquire("extra", account_pool_size=n_accounts)
        for le in leases:
            le.release()

    # ------------------------------------------------------------------
    # 1e. release from wrong thread does not corrupt state
    # ------------------------------------------------------------------
    def test_release_from_wrong_thread_safe(self) -> None:
        """A QueueLease acquired in one thread can be released from another."""
        q = self.make_queue(wait_timeout=2)
        lease = q.acquire("cross-thread", account_pool_size=3)
        released = False

        def releaser() -> None:
            nonlocal released
            lease.release()
            released = True

        t = threading.Thread(target=releaser)
        t.start()
        t.join(timeout=5)
        assert released
        assert lease.released
        # Queue state is sane: we can acquire again
        lease2 = q.acquire("after-cross", account_pool_size=3)
        lease2.release()

    # ------------------------------------------------------------------
    # 1f. _ensure_accounts grows dynamically
    # ------------------------------------------------------------------
    def test_ensure_accounts_dynamic_growth(self) -> None:
        q = self.make_queue()
        assert len(q._sems) == 0
        q._ensure_accounts(2)
        assert len(q._sems) == 2
        q._ensure_accounts(5)
        assert len(q._sems) == 5
        # Growing again to a smaller number does not shrink
        q._ensure_accounts(3)
        assert len(q._sems) == 5

    def test_ensure_accounts_minimum(self) -> None:
        """acquire with account_pool_size=0 still creates at least 3 semaphores."""
        q = self.make_queue()
        lease = q.acquire("min-ensure", account_pool_size=0)
        assert len(q._sems) >= 3
        lease.release()

    def test_ensure_accounts_thread_safe(self) -> None:
        """Race two threads calling _ensure_accounts concurrently."""
        q = self.make_queue()
        errors: list[Exception] = []

        def racer(target: int) -> None:
            try:
                q._ensure_accounts(target)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=racer, args=(3,))
        t2 = threading.Thread(target=racer, args=(5,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors
        assert len(q._sems) == 5

    # ------------------------------------------------------------------
    # 1g. PER_ACCOUNT_LIMIT = 3 is enforced
    # ------------------------------------------------------------------
    def test_per_account_limit_constant(self) -> None:
        assert ConcurrentRequestQueue.PER_ACCOUNT_LIMIT == 3

    def test_per_account_limit_enforced(self) -> None:
        """Each account semaphore allows exactly 3 acquisitions."""
        q = self.make_queue()
        q._ensure_accounts(1)
        sem = q._sems[0]
        acquires = [sem.acquire(blocking=False) for _ in range(10)]
        assert sum(1 for a in acquires if a) == 3

    # ------------------------------------------------------------------
    # 1h. Fair scheduling: best_idx picks account with most free slots
    # ------------------------------------------------------------------
    def test_fair_scheduling_most_free_slots(self) -> None:
        """After partially draining accounts, acquire picks the one with most capacity."""
        q = self.make_queue()
        q._ensure_accounts(3)

        # Drain account 0 completely (3 slots)
        for _ in range(3):
            q._sems[0].acquire(blocking=False)
        # Drain account 1 partially (2 slots)
        for _ in range(2):
            q._sems[1].acquire(blocking=False)
        # Account 2 untouched: 3 free slots

        # Next acquire should pick account 2 (index 2 has most free)
        lease = q.acquire("fairness", account_pool_size=3)
        assert lease.ticket == 2, f"expected account 2, got {lease.ticket}"
        lease.release()

    def test_fair_scheduling_all_equal_random(self) -> None:
        """When all accounts have same free slots, any is fine."""
        q = self.make_queue()
        q._ensure_accounts(3)
        lease = q.acquire("equal", account_pool_size=3)
        assert 0 <= lease.ticket <= 2
        lease.release()


# ===================================================================
# 2. GLMWebClient._call_with_account_failover
# ===================================================================

class TestCallWithAccountFailover:
    """Tests for the account failover logic."""

    # ------------------------------------------------------------------
    # 2a. Successful call on first try
    # ------------------------------------------------------------------
    def test_success_first_try(self) -> None:
        client = _make_client()
        operation = MagicMock(return_value={"ok": True})

        result = client._call_with_account_failover(
            "test-request", operation, preferred_account_index=0
        )

        assert result == {"ok": True}
        # Must have fetched token and recorded latency + success
        client.auth.get_access_token_for_account.assert_called_once_with(0)
        client.auth.record_latency.assert_called_once()
        client.auth.clear_account_failures.assert_called_once_with(0)
        client.auth.record_success.assert_called_once()
        # last_account_index must be set
        assert client._last_account_index == 0

    # ------------------------------------------------------------------
    # 2b. Failover to next account when first fails
    # ------------------------------------------------------------------
    def test_failover_to_next_account(self) -> None:
        client = _make_client()
        client.auth.get_account_count.return_value = 2
        client.auth.should_switch_account.return_value = True
        call_count: list[int] = []

        def failing_op(account_index: int, token: str) -> object:
            call_count.append(account_index)
            if account_index == 0:
                raise ConnectionError("upstream reset")
            return {"ok": True, "account": account_index}

        result = client._call_with_account_failover(
            "failover-test", failing_op, preferred_account_index=0
        )
        assert result == {"ok": True, "account": 1}
        # Guest retries first (default guest_retry_limit=1), then failover
        assert call_count == [0, 0, 1]  # 2x guest retry on 0, then 1
        # Account 0 invalidated, account 1 cleared

    def test_failover_non_guest_does_not_retry_same_account(self) -> None:
        """Non-guest accounts skip the inner retry loop (guest_retry_limit=0)."""
        client = _make_client()
        client.auth.get_account_count.return_value = 2
        client.auth.is_guest_account.return_value = False
        client.auth.should_switch_account.return_value = True
        call_count: list[int] = []

        def op(account: int, _: str) -> object:
            call_count.append(account)
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            client._call_with_account_failover("non-guest", op, 0)

        # Each account visited once only
        assert call_count == [0, 1]

    # ------------------------------------------------------------------
    # 2c. Exhaust all accounts then raise
    # ------------------------------------------------------------------
    def test_exhaust_all_accounts_raises(self) -> None:
        client = _make_client()
        client.auth.get_account_count.return_value = 2
        client.auth.should_switch_account.return_value = True
        client.config.glm_use_guest_refresh_token = False  # no fallback spawn

        def op(account: int, _: str) -> object:
            raise ConnectionError(f"fail-{account}")

        with pytest.raises(ConnectionError, match="fail-1"):
            client._call_with_account_failover("exhaust", op, 0)

    def test_exhaust_single_account_no_switch_raises(self) -> None:
        """Single account and should_switch=False => immediate raise."""
        client = _make_client()
        client.auth.get_account_count.return_value = 1
        client.auth.should_switch_account.return_value = False

        def op(account: int, _: str) -> object:
            raise RuntimeError("fatal")

        with pytest.raises(RuntimeError, match="fatal"):
            client._call_with_account_failover("single-fatal", op, 0)

    # ------------------------------------------------------------------
    # 2d. Guest token refresh on guest accounts
    # ------------------------------------------------------------------
    def test_guest_account_retry_on_token_failure(self) -> None:
        """Guest accounts retry with refreshed token (guest_retry_limit=1)."""
        client = _make_client()
        client.auth.get_account_count.return_value = 2
        client.auth.is_guest_account.return_value = True
        client.auth.should_switch_account.return_value = True
        client.config.glm_guest_max_retries = 1
        call_log: list[tuple[int, str]] = []

        def op(account: int, token: str) -> object:
            call_log.append((account, token))
            if len(call_log) == 1:
                raise RuntimeError("token expired")
            return {"ok": True}

        result = client._call_with_account_failover("guest-retry", op, 0)
        assert result == {"ok": True}
        # Same account called twice, then success on second attempt
        assert len(call_log) == 2
        assert call_log[0][0] == 0
        assert call_log[1][0] == 0

    def test_guest_account_exhaust_retries_then_failover(self) -> None:
        """Guest account exhausts retries then moves to next account."""
        client = _make_client()
        client.auth.get_account_count.return_value = 2
        client.auth.is_guest_account.return_value = True
        client.auth.should_switch_account.return_value = True
        client.config.glm_guest_max_retries = 1
        call_log: list[int] = []

        def op(account: int, _: str) -> object:
            call_log.append(account)
            raise RuntimeError("always fail")

        with pytest.raises(RuntimeError):
            client._call_with_account_failover("guest-exhaust", op, 0)

        # Account 0 tried 2 times (retry_limit=1 => 2 attempts), account 1 tried 2 times
        assert call_log == [0, 0, 1, 1]

    # ------------------------------------------------------------------
    # 2e. spawn_fresh_guest_account fallback when all fail
    # ------------------------------------------------------------------
    def test_spawn_fresh_fallback(self) -> None:
        """When all accounts exhausted and guest mode enabled, spawn a fresh one."""
        client = _make_client()
        client.auth.get_account_count.return_value = 2
        client.auth.should_switch_account.return_value = True
        client.config.glm_use_guest_refresh_token = True
        call_log: list[int] = []

        def op(account: int, _: str) -> object:
            call_log.append(account)
            if account < 3:
                raise ConnectionError(f"fail-{account}")
            return {"ok": True, "spawned": account}

        result = client._call_with_account_failover("spawn-test", op, 0)
        assert result == {"ok": True, "spawned": 3}
        client.auth.spawn_fresh_guest_account.assert_called_once()
        client.auth.reset_account_cycle.assert_called()

    def test_spawn_fresh_also_fails_raises_original(self) -> None:
        """When spawn_fresh_guest_account itself also fails, raise last original exception."""
        client = _make_client()
        client.auth.get_account_count.return_value = 1
        client.auth.should_switch_account.return_value = True
        client.config.glm_use_guest_refresh_token = True
        # Make spawn_fresh_guest_account raise
        client.auth.spawn_fresh_guest_account.side_effect = RuntimeError("spawn failed")
        last_error = RuntimeError("original fail")

        def op(account: int, _: str) -> object:
            raise last_error

        with pytest.raises(RuntimeError, match="original fail"):
            client._call_with_account_failover("spawn-fail", op, 0)

    # ------------------------------------------------------------------
    # 2f. force_refresh_all_guest_tokens fallback
    # ------------------------------------------------------------------
    def test_force_refresh_fallback(self) -> None:
        """When guest mode is off, fall back to force-refresh all tokens and retry."""
        client = _make_client()
        client.auth.get_account_count.return_value = 1
        client.auth.should_switch_account.return_value = True
        client.config.glm_use_guest_refresh_token = False
        call_log: list[int] = []

        def op(account: int, _: str) -> object:
            call_log.append(account)
            if not call_log or call_log.count(account) < 2:
                raise ConnectionError("fail")
            return {"ok": True}

        result = client._call_with_account_failover("force-refresh", op, 0)
        assert result == {"ok": True}
        client.auth.force_refresh_all_guest_tokens.assert_called_once()

    def test_force_refresh_also_fails_raises(self) -> None:
        """When force_refresh + retry also fails, raise last exception."""
        client = _make_client()
        client.auth.get_account_count.return_value = 1
        client.auth.should_switch_account.return_value = True
        client.config.glm_use_guest_refresh_token = False

        def op(account: int, _: str) -> object:
            raise ConnectionError("persistent fail")

        with pytest.raises(ConnectionError):
            client._call_with_account_failover("refresh-fail", op, 0)

    # ------------------------------------------------------------------
    # 2g. Circuit breaker integration
    # ------------------------------------------------------------------
    def test_track_failure_on_switch(self) -> None:
        """track_account_failure is called when should_switch_account returns True."""
        client = _make_client()
        client.auth.get_account_count.return_value = 2
        client.auth.should_switch_account.return_value = True

        def op(account: int, _: str) -> object:
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            client._call_with_account_failover("cb-test", op, 0)

        client.auth.track_account_failure.assert_called_once_with(0)

    def test_clear_failures_on_success(self) -> None:
        """clear_account_failures is called on successful result."""
        client = _make_client()
        client.auth.get_account_count.return_value = 2

        def op(account: int, _: str) -> object:
            return {"ok": True}

        client._call_with_account_failover("clear-test", op, 0)
        client.auth.clear_account_failures.assert_called_once_with(0)

    def test_mark_rate_limited_on_quota_error(self) -> None:
        """When the error looks like quota exhaustion, mark_rate_limited is called."""
        client = _make_client()
        client.auth.get_account_count.return_value = 2
        client.auth.should_switch_account.return_value = True

        def op(account: int, _: str) -> object:
            raise RuntimeError("rate limit exceeded")

        with pytest.raises(RuntimeError):
            client._call_with_account_failover("quota-test", op, 0)

        client.auth.mark_rate_limited.assert_called_once_with(0)

    # ------------------------------------------------------------------
    # 2h. EWMA latency recording
    # ------------------------------------------------------------------
    def test_record_latency_on_success(self) -> None:
        """record_latency is called with a positive float."""
        client = _make_client()

        def op(account: int, _: str) -> object:
            import time
            time.sleep(0.01)  # measurable delay
            return {"ok": True}

        client._call_with_account_failover("latency-test", op, 0)
        client.auth.record_latency.assert_called_once()
        args, _ = client.auth.record_latency.call_args
        assert args[0] == 0           # account index
        assert args[1] > 0            # latency > 0ms

    # ------------------------------------------------------------------
    # 2i. Empty content advance (via _call_with_account_failover caller)
    # ------------------------------------------------------------------
    def test_empty_content_advance_account(self) -> None:
        """Empty content in chat_completion triggers account advance and retry."""
        from glm2api.services.translator import GLMEventAccumulator

        client = _make_client()
        # We need a real auth for this test because _open_chat_stream calls auth methods
        client.auth = MagicMock(spec=GLMAccessTokenManager)
        client.auth.get_account_count.return_value = 2
        client.auth.is_guest_account.return_value = True
        client.auth.get_access_token_for_account.return_value = "tok"
        client.auth.get_best_account.return_value = 0
        client.auth.get_next_account_index.return_value = 0
        client.auth.should_switch_account.return_value = True
        client.auth.advance_account = MagicMock()
        client.auth.invalidate_account = MagicMock()
        client.auth.track_account_failure.return_value = False
        client.auth.mark_rate_limited = MagicMock()
        client.auth.record_latency = MagicMock()
        client.auth.clear_account_failures = MagicMock()
        client.auth.record_success = MagicMock()
        client.auth.spawn_fresh_guest_account = MagicMock()
        client.auth.force_refresh_all_guest_tokens = MagicMock()
        client.auth.reset_account_cycle = MagicMock()

        mock_response = MagicMock()
        client._open_chat_stream = MagicMock(return_value=(mock_response, "asst_1"))

        # The chat_completion method iterates over _iter_sse_events on the response.
        client._iter_sse_events = MagicMock(return_value=iter([
            {"conversation_id": "conv_1", "status": "finish", "parts": [
                {"logic_id": "1", "content": [{"type": "text", "text": ""}]}
            ]}
        ]))

        client.delete_conversation = MagicMock()
        client._record_usage = MagicMock()

        # The request queue needs to work for the lease
        payload = {"model": "glm-4", "messages": [{"role": "user", "content": "hello"}]}

        with pytest.raises(UpstreamAPIError, match="空内容"):
            client.chat_completion(payload)

        # The auth.advance_account must have been called with "empty_content"
        client.auth.advance_account.assert_called()
        # Get the call args
        advance_call_args = client.auth.advance_account.call_args
        assert advance_call_args is not None
        assert "empty_content" in str(advance_call_args)


# ===================================================================
# 3. Integration: _call_with_account_failover with real ConcurrentRequestQueue
# ===================================================================

class TestQueueFailoverIntegration:
    """End-to-end tests combining the queue and failover."""

    def test_full_flow_success(self) -> None:
        """Queue acquire -> failover -> operation -> release => happy path."""
        config = _make_config(glm_queue_wait_timeout=5)
        client = _make_client(config=config)
        # Give the queue real semaphores matching the account count
        client.request_queue._ensure_accounts(3)
        operation = MagicMock(return_value={"result": "ok"})

        result = client._call_with_account_failover(
            "integration", operation, preferred_account_index=0
        )
        assert result == {"result": "ok"}

    def test_full_flow_failover(self) -> None:
        """Queue acquire -> failover across two accounts -> success on second."""
        config = _make_config(glm_queue_wait_timeout=5)
        client = _make_client(config=config)
        client.request_queue._ensure_accounts(3)
        client.auth.get_account_count.return_value = 2
        client.auth.should_switch_account.return_value = True
        call_log: list[int] = []

        def op(account: int, token: str) -> object:
            call_log.append(account)
            if account == 0:
                raise ConnectionError("fail")
            return {"ok": True, "account": account}

        result = client._call_with_account_failover(
            "integration-failover", op, preferred_account_index=0
        )
        assert result == {"ok": True, "account": 1}
        assert call_log == [0, 1]
