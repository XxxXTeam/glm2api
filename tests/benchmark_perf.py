#!/usr/bin/env python3
"""Performance benchmarks for GLM2API critical code paths.

Measures:
  1. SSE parsing speed  (_iter_sse_events)
  2. JSON serialization (orjson vs stdlib)
  3. Message conversion (convert_messages)
  4. Proxy selection    (get_best)
  5. Queue acquire/release (ConcurrentRequestQueue)
  6. Header generation  (get_browser_headers)
  7. Token refresh flow (mocked)

Run:  python -m tests.benchmark_perf
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import threading
import time
from io import BytesIO, BufferedReader
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------
#  Ensure project root is on sys.path
# ----------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC = _PROJECT / "src"
sys.path.insert(0, str(_SRC))

# ----------------------------------------------------------------
#  Imports
# ----------------------------------------------------------------
import orjson

from glm2api.config import AppConfig
from glm2api.services.glm_auth import (
    GLMAccessTokenManager,
    build_sign,
    _HeaderPool,
    _random_ip,
)
from glm2api.services.glm_client import (
    ConcurrentRequestQueue,
    QueueLease,
    GLMWebClient,
)
from glm2api.services.glm2api_proxy import (
    SmartProxyPool,
    ProxyScore,
    get_pool as _get_proxy_pool,
)
from glm2api.services.translator import (
    convert_messages,
    _json_dumps,
    _json_loads,
    apply_context_strategy,
)
from glm2api.utils.tool_protocol import SERVER_SIDE_TOOL_NAMES

# ----------------------------------------------------------------
#  Benchmark harness
# ----------------------------------------------------------------

class Timer:
    """Precision timer for microbenchmarks."""

    __slots__ = ("_start", "_name")

    def __init__(self, name: str = ""):
        self._name = name
        self._start = 0.0

    def __enter__(self):
        gc.disable()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        gc.enable()
        # keep reference to avoid being optimized out
        _ = elapsed

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000


def bench(name: str, iterations: int | list[int], warmup: int = 2):
    """Decorator-like helper that prints result.  `iterations` can be an int
    (run that many iterations and report total/avg) or a list of iteration
    counts to try (auto-scaling)."""

    def deco(fn):
        def wrapper(*args, **kwargs):
            # Warmup
            for _ in range(warmup):
                fn(*args, **kwargs)

            if isinstance(iterations, int):
                count = iterations
                t0 = time.perf_counter()
                for _ in range(count):
                    fn(*args, **kwargs)
                total_ms = (time.perf_counter() - t0) * 1000
                avg_us = total_ms / count * 1000
                ops = count / (total_ms / 1000) if total_ms > 0 else float("inf")
                print(
                    f"  {name:45s}  {total_ms:>8.2f} ms total  "
                    f"{avg_us:>8.2f} us/op  {ops:>10.0f} ops/s"
                )
                return total_ms, avg_us, ops
            else:
                # auto-scale: find largest iteration count that runs in <3s
                results = []
                for n in iterations:
                    t0 = time.perf_counter()
                    for _ in range(n):
                        fn(*args, **kwargs)
                    elapsed = (time.perf_counter() - t0) * 1000
                    avg_us = elapsed / n * 1000
                    ops = n / (elapsed / 1000) if elapsed > 0 else float("inf")
                    print(
                        f"  {name:45s}  n={n:<8}  {avg_us:>8.2f} us/op  "
                        f"{ops:>10.0f} ops/s"
                    )
                    results.append((n, avg_us, ops))
                    if elapsed > 3000:
                        break
                return results

        return wrapper

    return deco


# ----------------------------------------------------------------
#  1. SSE parsing benchmark
# ----------------------------------------------------------------

def _make_sse_event(seq: int, text_len: int = 200) -> bytes:
    """Build a realistic SSE event payload."""
    text = f"Hello, world! This is event {seq}. " * (text_len // 30 + 1)
    payload = {
        "conversation_id": f"conv_{seq:06x}",
        "status": "finish" if seq % 5 == 0 else "",
        "parts": [
            {
                "logic_id": f"part_{seq}",
                "content": [{"type": "text", "text": text[:text_len]}],
            }
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    sse = f"data: {raw}\n\n".encode()
    return sse


def _sse_stream(event_count: int, text_len: int = 200) -> BufferedReader:
    """Build an SSE byte stream of *event_count* events."""
    chunks = bytearray()
    for i in range(event_count):
        chunks.extend(_make_sse_event(i, text_len))
    return BufferedReader(BytesIO(bytes(chunks)))


def bench_sse_parsing():
    """Benchmark _iter_sse_events with varying event counts."""
    print("\n" + "=" * 72)
    print("1. SSE PARSING SPEED  (_iter_sse_events)")
    print("=" * 72)
    import logging
    import types

    _log = logging.getLogger("bench_sse")

    class _FakeClient:
        config = type("cfg", (), {"debug_dump_all": False})()

    client = _FakeClient()
    client.logger = _log

    def _make_bound_iter(cl):
        # Bind the method to avoid decorator wrapping issues
        fn = GLMWebClient._iter_sse_events
        if hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        return types.MethodType(fn, cl)

    bound_iter = _make_bound_iter(client)

    for event_count in [10, 100, 1000]:
        stream = _sse_stream(event_count, text_len=200)
        # warmup
        for _ in range(3):
            list(bound_iter(stream, stream_timeout=5))
            stream.seek(0)

        t0 = time.perf_counter()
        stream.seek(0)
        events = list(bound_iter(stream, stream_timeout=5))
        elapsed = (time.perf_counter() - t0) * 1000
        avg_us = elapsed / event_count * 1000
        ops = event_count / (elapsed / 1000)
        print(
            f"  SSE parse ({event_count:>5} events):  "
            f"{elapsed:>8.2f} ms total  "
            f"{avg_us:>8.2f} us/event  {ops:>10.0f} events/s"
        )
    return


# ----------------------------------------------------------------
#  2. JSON serialization benchmark
# ----------------------------------------------------------------

_SAMPLE_PAYLOAD = {
    "id": "conv_abc123",
    "object": "chat.completion.chunk",
    "created": 1712345678,
    "model": "glm-4-flash",
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": (
                    "Hello! I am a large language model. I can help you with "
                    "various tasks such as answering questions, writing code, "
                    "and more. Let me know what you need!" * 5
                ),
            },
            "finish_reason": None,
        }
    ],
    "usage": {"prompt_tokens": 150, "completion_tokens": 42, "total_tokens": 192},
}

_LARGE_PAYLOAD = {
    "conversation_id": "conv_large_001",
    "parts": [
        {
            "logic_id": f"part_{i}",
            "content": [
                {"type": "text", "text": "This is a moderately long text block that simulates realistic GLM upstream responses." * 3},
                {"type": "think", "think": "Let me reason through this step by step..." * 5},
            ],
        }
        for i in range(100)
    ],
}


def bench_json():
    """Compare orjson vs stdlib json.dumps / json.loads."""
    print("\n" + "=" * 72)
    print("2. JSON SERIALIZATION  (orjson vs stdlib)")
    print("=" * 72)

    # --- Serialization: small payload ---
    for label, payload in [("small", _SAMPLE_PAYLOAD), ("large (100 parts)", _LARGE_PAYLOAD)]:
        # orjson
        _ = orjson.dumps(payload)
        n = 5000 if label.startswith("small") else 500
        t0 = time.perf_counter()
        for _ in range(n):
            _ = orjson.dumps(payload)
        t1 = time.perf_counter()
        orjson_us = (t1 - t0) / n * 1e6
        ops = n / (t1 - t0)
        print(f"  orjson dumps ({label:20s}):  {orjson_us:>8.2f} us  {ops:>10.0f} ops/s")

        # stdlib
        _ = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        t0 = time.perf_counter()
        for _ in range(n):
            _ = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        t1 = time.perf_counter()
        stdlib_us = (t1 - t0) / n * 1e6
        ops = n / (t1 - t0)
        print(f"  stdlib dumps ({label:20s}):  {stdlib_us:>8.2f} us  {ops:>10.0f} ops/s")
        print(f"  -> speedup: {stdlib_us / orjson_us:.1f}x")

    # --- Deserialization ---
    for label, payload in [("small", _SAMPLE_PAYLOAD), ("large (100 parts)", _LARGE_PAYLOAD)]:
        raw_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        raw_bytes = raw_json.encode("utf-8")

        _ = orjson.loads(raw_bytes)
        n = 5000 if label.startswith("small") else 500
        t0 = time.perf_counter()
        for _ in range(n):
            _ = orjson.loads(raw_bytes)
        t1 = time.perf_counter()
        orjson_us = (t1 - t0) / n * 1e6
        ops = n / (t1 - t0)
        print(f"  orjson loads ({label:20s}):  {orjson_us:>8.2f} us  {ops:>10.0f} ops/s")

        _ = json.loads(raw_json)
        t0 = time.perf_counter()
        for _ in range(n):
            _ = json.loads(raw_json)
        t1 = time.perf_counter()
        stdlib_us = (t1 - t0) / n * 1e6
        ops = n / (t1 - t0)
        print(f"  stdlib loads ({label:20s}):  {stdlib_us:>8.2f} us  {ops:>10.0f} ops/s")
        print(f"  -> speedup: {stdlib_us / orjson_us:.1f}x")

    # --- _chunk_json (SSE frame builder used in GLMEventAccumulator) ---
    # We'll measure the internal _chunk_json logic
    print()
    from glm2api.services.translator import GLMEventAccumulator

    acc = GLMEventAccumulator(model="glm-4-flash")
    patch = {
        "choices": [
            {
                "index": 0,
                "delta": {"content": "Hello world!"},
                "finish_reason": None,
            }
        ]
    }
    _ = acc._chunk_json(patch)
    n = 10000
    t0 = time.perf_counter()
    for _ in range(n):
        _ = acc._chunk_json(patch)
    elapsed = (time.perf_counter() - t0) * 1000
    avg_us = elapsed / n * 1000
    ops = n / (elapsed / 1000)
    print(f"  GLMEventAccumulator._chunk_json:  {avg_us:>8.2f} us  {ops:>10.0f} ops/s")

    # --- translator _json_dumps helper ---
    _ = _json_dumps(_SAMPLE_PAYLOAD)
    n = 10000
    t0 = time.perf_counter()
    for _ in range(n):
        _ = _json_dumps(_SAMPLE_PAYLOAD)
    elapsed = (time.perf_counter() - t0) * 1000
    avg_us = elapsed / n * 1000
    print(f"  translator._json_dumps:  {avg_us:>8.2f} us  {n / (elapsed / 1000):>10.0f} ops/s")


# ----------------------------------------------------------------
#  3. Message conversion benchmark
# ----------------------------------------------------------------

def _make_conversation(message_count: int, tool_count: int = 3) -> dict:
    """Build a realistic conversation for convert_messages."""
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    for i in range(message_count):
        msgs.append(
            {
                "role": "user",
                "content": f"What is the capital of France? Tell me more about it. " * 2,
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": f"The capital of France is Paris. It is known for landmarks like the Eiffel Tower. " * 3,
            }
        )
    tools = [
        {"type": "function", "function": {"name": f"tool_{j}", "description": f"A test tool {j}", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}
        for j in range(tool_count)
    ]
    return {"messages": msgs, "tools": tools}


def bench_convert_messages():
    """Benchmark convert_messages at various conversation depths."""
    print("\n" + "=" * 72)
    print("3. MESSAGE CONVERSION  (convert_messages)")
    print("=" * 72)

    for msg_count in [2, 10, 50, 100]:
        ctx = _make_conversation(msg_count)
        _ = convert_messages(
            messages=ctx["messages"],
            tools=ctx["tools"],
            blocked_tool_names=set(),
            server_side_tool_names=SERVER_SIDE_TOOL_NAMES,
            context_strategy="sliding",
            context_limit=30,
        )
        n = max(1, 2000 // max(msg_count, 1))
        t0 = time.perf_counter()
        for _ in range(n):
            _ = convert_messages(
                messages=ctx["messages"],
                tools=ctx["tools"],
                blocked_tool_names=set(),
                server_side_tool_names=SERVER_SIDE_TOOL_NAMES,
                context_strategy="sliding",
                context_limit=30,
            )
        elapsed = (time.perf_counter() - t0) * 1000
        avg_us = elapsed / n * 1000
        ops = n / (elapsed / 1000) if elapsed > 0 else 0
        print(
            f"  convert_messages ({msg_count:>3} msgs, {len(ctx['tools'])} tools):  "
            f"{avg_us:>8.2f} us/call  {ops:>10.0f} calls/s"
        )

    # With tool calls in messages (simulating real tool use)
    print()
    msgs_with_tools = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Search the web for AI news."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query": "AI news 2025"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_001", "name": "web_search", "content": "Search results here..."},
        {"role": "assistant", "content": "Here's what I found about AI news in 2025."},
    ]
    _ = convert_messages(
        messages=msgs_with_tools,
        tools=[{"type": "function", "function": {"name": "web_search", "description": "Search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}],
        blocked_tool_names=set(),
        server_side_tool_names=SERVER_SIDE_TOOL_NAMES,
    )
    n = 5000
    t0 = time.perf_counter()
    for _ in range(n):
        _ = convert_messages(
            messages=msgs_with_tools,
            tools=[{"type": "function", "function": {"name": "web_search", "description": "Search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}],
            blocked_tool_names=set(),
            server_side_tool_names=SERVER_SIDE_TOOL_NAMES,
        )
    elapsed = (time.perf_counter() - t0) * 1000
    avg_us = elapsed / n * 1000
    print(f"  convert_messages (tool calls in history):  {avg_us:>8.2f} us/call")


# ----------------------------------------------------------------
#  4. Proxy selection benchmark
# ----------------------------------------------------------------

def bench_proxy_selection():
    """Benchmark SmartProxyPool.get_best() with varying pool sizes."""
    print("\n" + "=" * 72)
    print("4. PROXY SELECTION  (SmartProxyPool.get_best)")
    print("=" * 72)

    for pool_size in [10, 100, 500]:
        pool = SmartProxyPool.__new__(SmartProxyPool)
        # Minimal init to avoid network I/O
        pool._lock = threading.RLock()
        pool._proxies = {}
        pool._current = ""
        pool._hot_pool = []
        pool._last_hot_refresh = time.monotonic()
        pool._hot_refresh_interval = 99999
        pool._hot_pool_size = 100
        pool._guest_rr_index = 0
        pool._main_rr_index = 0
        pool._last_refresh = time.monotonic()
        pool._last_health_check = 0.0

        for i in range(pool_size):
            ps = ProxyScore(url=f"socks5://proxy{i:04d}.example.com:1080")
            ps.alive = True
            ps._score_cache = ps._recompute_score()
            pool._proxies[ps.url] = ps

        # Warmup hot pool
        pool._refresh_hot_pool()

        # warmup
        for _ in range(10):
            pool.get_best()

        n = max(1, 20000 // max(pool_size, 1))
        t0 = time.perf_counter()
        for _ in range(n):
            pool.get_best()
        elapsed = (time.perf_counter() - t0) * 1000
        avg_us = elapsed / n * 1000
        ops = n / (elapsed / 1000) if elapsed > 0 else 0
        print(
            f"  get_best (pool={pool_size:>4}):  "
            f"{avg_us:>8.2f} us/call  {ops:>10.0f} calls/s"
        )

    # Benchmark get_unique and get_next too
    pool = SmartProxyPool.__new__(SmartProxyPool)
    pool._lock = threading.RLock()
    pool._proxies = {}
    pool._current = ""
    pool._hot_pool = []
    pool._last_hot_refresh = time.monotonic()
    pool._hot_refresh_interval = 99999
    pool._guest_rr_index = 0
    pool._main_rr_index = 0
    pool._last_refresh = time.monotonic()
    pool._last_health_check = 0.0

    for i in range(50):
        ps = ProxyScore(url=f"socks5://proxy{i:04d}.example.com:1080")
        ps.alive = True
        pool._proxies[ps.url] = ps

    for name, method in [("get_unique", pool.get_unique), ("get_next", pool.get_next)]:
        for _ in range(5):
            method()
        n = 10000
        t0 = time.perf_counter()
        for _ in range(n):
            method()
        elapsed = (time.perf_counter() - t0) * 1000
        avg_us = elapsed / n * 1000
        print(f"  {name:45s}  {avg_us:>8.2f} us/call")


# ----------------------------------------------------------------
#  5. Queue acquire/release benchmark
# ----------------------------------------------------------------

def bench_queue():
    """Benchmark ConcurrentRequestQueue acquire/release with contention."""
    print("\n" + "=" * 72)
    print("5. QUEUE ACQUIRE/RELEASE  (ConcurrentRequestQueue)")
    print("=" * 72)

    import logging

    logger = logging.getLogger("bench_queue")
    logger.setLevel(logging.WARNING)

    queue = ConcurrentRequestQueue(logger=logger, wait_timeout=30, max_concurrency=3)
    queue._ensure_accounts(3)

    # Single-thread acquire+release
    lease = queue.acquire("bench_single")
    lease.release()
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        lease = queue.acquire("bench_multi")
        lease.release()
    elapsed = (time.perf_counter() - t0) * 1000
    avg_us = elapsed / n * 1000
    ops = n / (elapsed / 1000)
    print(f"  acquire+release (single thread):  {avg_us:>8.2f} us/pair  {ops:>10.0f} pairs/s")

    # Multi-thread contention: N threads fighting for slots
    for num_threads in [2, 5, 10]:
        completed = [0]
        lock = threading.Lock()
        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()
            for _ in range(2000):
                try:
                    lease = queue.acquire(f"worker_{threading.get_ident()}", account_pool_size=3)
                    # Simulate tiny work
                    time.sleep(0.0001)
                    lease.release()
                    with lock:
                        completed[0] += 1
                except Exception:
                    pass

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = (time.perf_counter() - t0) * 1000
        ops = completed[0] / (elapsed / 1000) if elapsed > 0 else 0
        per_thread_ops = ops / num_threads
        print(
            f"  acquire+release ({num_threads:>2} threads, {completed[0]:>5} ops):  "
            f"{elapsed:>8.2f} ms  {ops:>8.0f} total ops/s  "
            f"{per_thread_ops:>8.0f} ops/s/thread"
        )


# ----------------------------------------------------------------
#  6. Header generation benchmark
# ----------------------------------------------------------------

def bench_headers():
    """Benchmark GLMAccessTokenManager.get_browser_headers."""
    print("\n" + "=" * 72)
    print("6. HEADER GENERATION  (get_browser_headers)")
    print("=" * 72)

    import logging

    logger = logging.getLogger("bench_headers")
    logger.setLevel(logging.WARNING)

    # Build a minimal config and token manager
    config = AppConfig(
        env_file_path=Path("/nonexistent"),
        env_file_created=False,
        token_file_path=Path("/nonexistent"),
        host="127.0.0.1",
        port=8000,
        api_prefix="/v1",
        log_level="WARNING",
        debug_dump_all=False,
        request_timeout=120,
        glm_base_url="https://chatglm.cn/chatglm",
        glm_use_guest_refresh_token=True,
        glm_refresh_token="__glm_guest__",
        glm_refresh_tokens=["__glm_guest__", "__glm_guest__", "__glm_guest__"],
        glm_assistant_id="65940acff94777010aa6b796",
        glm_image_assistant_id="65a232c082ff90a2ad2f15e2",
        glm_image_model_name="glm-image-1",
        glm_user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        glm_delete_conversation=False,
        glm_max_concurrency=3,
        glm_queue_wait_timeout=120,
        glm_busy_max_retries=5,
        glm_busy_retry_interval=2.0,
        glm_guest_max_retries=3,
        glm_proxy_max_retries=2,
        glm_account_strategy="ewma",
        blocked_tool_names=[],
        context_strategy="sliding",
        context_limit=30,
        exposed_models=["glm-4-flash"],
        model_aliases={"glm-4": "glm-4"},
        server_api_keys=[],
        cors_allow_origin="*",
    )

    # Create token manager but avoid network init
    auth = GLMAccessTokenManager.__new__(GLMAccessTokenManager)
    auth.config = config
    auth.logger = logger
    auth._accounts = []
    auth._current_index = 0
    auth._lock = threading.RLock()
    auth._persist_lock = threading.Lock()
    auth._round_robin_counter = 0
    auth._rate_limited_accounts = {}
    auth._rate_limit_cooldown = 3
    auth._last_guest_fetch = 0.0
    auth._guest_fetch_interval = 3.0
    auth._consecutive_failures = 0
    auth._starvation_threshold = 6
    auth._silent_accounts = {}
    auth._silence_cooldown = 120
    auth._last_full_refresh_at = 0.0
    auth._circuit_open = False
    auth._circuit_opened_at = 0.0
    auth._circuit_failures = 0
    auth._circuit_threshold = 6
    auth._circuit_cooldown = 30
    auth._full_refresh_cooldown = 300
    auth._account_fail_count = {}
    auth._account_fail_threshold = 3
    auth._fill_first_index = 0
    auth._warm_spares = []
    auth._warm_spare_target = 3

    # Clear the class-level cache
    GLMAccessTokenManager._STATIC_HEADERS_CACHE.clear()

    # Warmup
    _ = auth.get_browser_headers()

    n = 50000
    t0 = time.perf_counter()
    for _ in range(n):
        _ = auth.get_browser_headers()
    elapsed = (time.perf_counter() - t0) * 1000
    avg_us = elapsed / n * 1000
    ops = n / (elapsed / 1000)
    print(f"  get_browser_headers (cached pool):  {avg_us:>8.2f} us/call  {ops:>10.0f} calls/s")

    # Also benchmark _random_ip + dict copy (the non-cached version cost)
    base = {
        "Accept": "*/*", "Accept-Encoding": "gzip", "Accept-Language": "en",
        "App-Name": "chatglm", "Cache-Control": "no-cache", "Content-Type": "application/json",
        "Origin": "https://chatglm.cn", "Pragma": "no-cache",
        "User-Agent": config.glm_user_agent,
        "X-App-Fr": "browser_extension", "X-App-Platform": "pc", "X-App-Version": "1.0.83",
    }
    n = 100000
    t0 = time.perf_counter()
    for _ in range(n):
        h = base.copy()
        h["X-Forwarded-For"] = _random_ip()
    elapsed = (time.perf_counter() - t0) * 1000
    avg_us = elapsed / n * 1000
    print(f"  dict copy + random_ip (uncached):  {avg_us:>8.2f} us/header")
    _ = avg_us

    # Benchmark _random_ip alone
    n = 100000
    t0 = time.perf_counter()
    for _ in range(n):
        _random_ip()
    elapsed = (time.perf_counter() - t0) * 1000
    avg_us = elapsed / n * 1000
    print(f"  _random_ip alone:  {avg_us:>8.2f} us/call")

    # Benchmark build_sign
    n = 50000
    t0 = time.perf_counter()
    for _ in range(n):
        build_sign()
    elapsed = (time.perf_counter() - t0) * 1000
    avg_us = elapsed / n * 1000
    print(f"  build_sign:  {avg_us:>8.2f} us/call")


# ----------------------------------------------------------------
#  7. Token refresh flow (mocked)
# ----------------------------------------------------------------

def bench_token_refresh():
    """Benchmark token refresh logic overhead (mocked HTTP)."""
    print("\n" + "=" * 72)
    print("7. TOKEN REFRESH FLOW  (mocked overhead)")
    print("=" * 72)

    import logging

    logger = logging.getLogger("bench_token")
    logger.setLevel(logging.WARNING)

    # The actual _fetch_guest_access_token does HTTP.  Instead we benchmark
    # the overhead of the retry wrapper, sign building, and header assembly
    # that would wrap each token fetch.

    # Benchmark _exec_with_retry wrapper overhead
    config = AppConfig(
        env_file_path=Path("/nonexistent"),
        env_file_created=False,
        token_file_path=Path("/nonexistent"),
        host="127.0.0.1", port=8000, api_prefix="/v1",
        log_level="WARNING", debug_dump_all=False, request_timeout=120,
        glm_base_url="https://chatglm.cn/chatglm", glm_use_guest_refresh_token=True,
        glm_refresh_token="__glm_guest__",
        glm_refresh_tokens=["__glm_guest__", "__glm_guest__", "__glm_guest__"],
        glm_assistant_id="65940acff94777010aa6b796",
        glm_image_assistant_id="65a232c082ff90a2ad2f15e2",
        glm_image_model_name="glm-image-1",
        glm_user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        glm_delete_conversation=False, glm_max_concurrency=3,
        glm_queue_wait_timeout=120, glm_busy_max_retries=5,
        glm_busy_retry_interval=2.0, glm_guest_max_retries=3,
        glm_proxy_max_retries=2, glm_account_strategy="ewma",
        blocked_tool_names=[], context_strategy="sliding", context_limit=30,
        exposed_models=["glm-4-flash"], model_aliases={"glm-4": "glm-4"},
        server_api_keys=[], cors_allow_origin="*",
    )

    auth = GLMAccessTokenManager.__new__(GLMAccessTokenManager)
    auth.config = config
    auth.logger = logger
    auth._accounts = []
    auth._current_index = 0
    auth._lock = threading.RLock()
    auth._persist_lock = threading.Lock()
    auth._round_robin_counter = 0
    auth._rate_limited_accounts = {}
    auth._rate_limit_cooldown = 3
    auth._last_guest_fetch = 0.0
    auth._guest_fetch_interval = 3.0
    auth._consecutive_failures = 0
    auth._starvation_threshold = 6
    auth._silent_accounts = {}
    auth._silence_cooldown = 120
    auth._last_full_refresh_at = 0.0
    auth._circuit_open = False
    auth._circuit_opened_at = 0.0
    auth._circuit_failures = 0
    auth._circuit_threshold = 6
    auth._circuit_cooldown = 30
    auth._full_refresh_cooldown = 300
    auth._account_fail_count = {}
    auth._account_fail_threshold = 3
    auth._fill_first_index = 0
    auth._warm_spares = []
    auth._warm_spare_target = 3
    GLMAccessTokenManager._STATIC_HEADERS_CACHE.clear()

    # Benchmark: just build_sign + header assembly (no HTTP)
    n = 10000
    t0 = time.perf_counter()
    for _ in range(n):
        ts, nonce, sign = build_sign()
        _ = {
            **auth.get_browser_headers(),
            "Authorization": "Bearer mock_token",
            "X-Device-Id": "d" * 32,
            "X-Nonce": nonce,
            "X-Request-Id": "r" * 32,
            "X-Sign": sign,
            "X-Timestamp": ts,
        }
    elapsed = (time.perf_counter() - t0) * 1000
    avg_us = elapsed / n * 1000
    ops = n / (elapsed / 1000)
    print(f"  sign+header assembly (no HTTP):  {avg_us:>8.2f} us/request  {ops:>10.0f} ops/s")

    # Benchmark _get_access_token_for_index with valid cached token
    from glm2api.services.glm_auth import AccessToken
    import time as _time
    auth._accounts = [
        type("a", (), {
            "cached_token": AccessToken(
                access_token="tok_abc123",
                refresh_token="ref_abc123",
                expires_at=_time.time() + 3500,
            ),
            "is_guest": False,
            "refresh_token": "ref_abc123",
            "tool_call_count": 0,
            "search_count": 0,
            "ewma_latency": 0.0,
            "ewma_alpha": 0.125,
            "_device_id": None,
        })()
    ]
    _ = auth.get_access_token()
    n = 20000
    t0 = time.perf_counter()
    for _ in range(n):
        _ = auth.get_access_token()
    elapsed = (time.perf_counter() - t0) * 1000
    avg_us = elapsed / n * 1000
    print(f"  get_access_token (cached):  {avg_us:>8.2f} us/call")


# ----------------------------------------------------------------
#  Main
# ----------------------------------------------------------------

def main():
    print("=" * 72)
    print("  GLM2API  PERFORMANCE BENCHMARKS")
    print("  Python:", sys.version.split()[0])
    import platform
    print("  Platform:", platform.machine())
    print("=" * 72)

    bench_sse_parsing()
    bench_json()
    bench_convert_messages()
    bench_proxy_selection()
    bench_queue()
    bench_headers()
    bench_token_refresh()

    print("\n" + "=" * 72)
    print("  BENCHMARKS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
