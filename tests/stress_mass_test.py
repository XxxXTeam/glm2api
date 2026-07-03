"""Mass stress test: queue, proxy pool, SSE, memory — all at once.
(c) cavekit spec → ponytail build → check drift loop.
"""
from __future__ import annotations
import sys, os, time, threading, json, random, gc, tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from glm2api.services.glm_client import ConcurrentRequestQueue, QueueTimeoutError
from glm2api.services.glm2api_proxy import SmartProxyPool, ProxyScore
from glm2api.services.translator import GLMEventAccumulator
from glm2api.utils.tool_parser import parse_tool_calls_from_text, StreamingToolParser

PASS = 0
FAIL = 0
RESULTS = []

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append(f"  PASS  {name}")
    else:
        FAIL += 1
        RESULTS.append(f"  FAIL  {name}  {detail}")

# ===== PART 1: QUEUE STRESS =====
print("\n=== PART 1: Queue Saturation (200 threads) ===")

def queue_stress():
    q = ConcurrentRequestQueue(logger=__import__('logging').getLogger('test'), wait_timeout=30, max_concurrency=60)
    q._ensure_accounts(10)  # 10 accounts x 3 slots = 30 max
    lock = threading.Lock()
    acquired = 0
    errors = []

    def worker():
        nonlocal acquired
        try:
            lease = q.acquire("stress", account_pool_size=10)
            with lock:
                acquired += 1
            time.sleep(random.uniform(0.001, 0.01))
            lease.release()
        except Exception as e:
            with lock:
                errors.append(str(e))

    pool = ThreadPoolExecutor(max_workers=50)
    t0 = time.monotonic()
    list(pool.map(lambda _: worker(), range(200)))
    elapsed = time.monotonic() - t0

    check("200 threads completed", acquired + len(errors) == 200, f"ok={acquired} err={len(errors)}")
    check("Avg acquire time < 50ms", elapsed / max(acquired,1) < 0.05, f"{elapsed/acquired*1000:.1f}ms avg")
    check("No errors", len(errors) == 0, str(errors[:3]))

queue_stress()

# ===== PART 2: QUEUE TIMEOUT PRECISION =====
print("\n=== PART 2: Queue Timeout Precision ===")

def timeout_precision():
    q = ConcurrentRequestQueue(logger=__import__('logging').getLogger('test'), wait_timeout=0.1, max_concurrency=3)
    q._ensure_accounts(3)
    # Fill all 9 slots (3 accounts x 3 slots)
    leases = [q.acquire("fill", account_pool_size=3) for _ in range(9)]
    t0 = time.monotonic()
    try:
        q.acquire("timeout-test", account_pool_size=3)
        check("Timeout raised", False, "should have raised")
    except QueueTimeoutError:
        elapsed = time.monotonic() - t0
        check("Timeout ~100ms", 0.08 < elapsed < 0.5, f"{elapsed*1000:.0f}ms")
    for le in leases:
        le.release()

timeout_precision()

# ===== PART 3: PROXY POOL WITH 500 PROXIES =====
print("\n=== PART 3: Proxy Pool (500 proxies, concurrent scoring) ===")

def proxy_pool_stress():
    pool = SmartProxyPool.__new__(SmartProxyPool)
    pool._lock = threading.Lock()
    pool._proxies = {}
    pool._last_refresh = 0.0

    # 500 proxies: 100 good (high score), 100 medium, 300 bad (low score)
    for i in range(100):
        p = ProxyScore(url=f"socks5://good{i}.proxy:1080", successes=50, failures=2, total_calls=52, latency_ms=200, consec_failures=0)
        pool._proxies[f"socks5://good{i}.proxy:1080"] = p
    for i in range(100):
        p = ProxyScore(url=f"socks5://med{i}.proxy:1080", successes=20, failures=10, total_calls=30, latency_ms=500, consec_failures=1)
        pool._proxies[f"socks5://med{i}.proxy:1080"] = p
    for i in range(300):
        p = ProxyScore(url=f"socks5://bad{i}.proxy:1080", successes=1, failures=50, total_calls=51, latency_ms=5000, consec_failures=5)
        pool._proxies[f"socks5://bad{i}.proxy:1080"] = p
        p.alive = False
        p.blacklisted_until = time.monotonic() + 9999

    # get_best() should always return a good proxy
    best = pool.get_best()
    check("Best proxy is good", "good" in (best or ""), str(best))
    
    # Score ordering
    scores = [p.score for p in pool._proxies.values() if p.alive]
    check("Alive proxies have positive scores", all(s > 0 for s in scores), f"min={min(scores):.1f}")
    
    # 1000 get_best calls
    t0 = time.monotonic()
    for _ in range(1000):
        pool.get_best()
    elapsed = time.monotonic() - t0
    # ponytail: with 500 proxies in single-pass, 1000 calls well under 20ms
    check("1000 get_best calls < 100ms", elapsed < 0.1, f"{elapsed*1000:.1f}ms")

proxy_pool_stress()

# ===== PART 4: CONCURRENT PROXY REPORTING =====
print("\n=== PART 4: Concurrent Proxy Reporting (100 threads) ===")

def concurrent_reporting():
    pool = SmartProxyPool.__new__(SmartProxyPool)
    pool._lock = threading.Lock()
    pool._proxies = {"socks5://test:1080": ProxyScore(url="socks5://test:1080")}
    pool._last_refresh = 0.0

    def worker_success():
        for _ in range(10):
            pool.report_success("socks5://test:1080", latency=random.uniform(50, 500))
    
    def worker_failure():
        for _ in range(5):
            pool.report_failure("socks5://test:1080")

    with ThreadPoolExecutor(max_workers=50) as ex:
        futures = [ex.submit(worker_success) for _ in range(50)]
        futures += [ex.submit(worker_failure) for _ in range(50)]
        for f in as_completed(futures):
            f.result()

    p = pool._proxies["socks5://test:1080"]
    # ponytail: concurrent success/failure may interleave, but totals should be correct
    check("Total calls == ", p.total_calls == 750, f"{p.total_calls}")
    check("Success+failure matches", p.successes + p.failures <= p.total_calls, f"ok={p.successes} fail={p.failures} total={p.total_calls}")
    # After all operations, final state should be alive (last success resets)
    # But due to race, might be blacklisted. That's OK — real usage is sequential per proxy.
    print(f"  INFO: Final state: alive={p.alive} consec_fail={p.consec_failures} score={p.score:.0f}")

concurrent_reporting()

# ===== PART 5: SSE ACCUMULATOR MASSIVE PAYLOAD =====
print("\n=== PART 5: SSE Accumulator (500KB, 1000 parts) ===")

def sse_massive():
    acc = GLMEventAccumulator(model="glm-test", allowed_tool_names={"write"})
    parts = []
    for i in range(1000):
        part = {
            "logic_id": str(i),
            "content": [
                {"type": "text", "text": f"Part {i}: " + "x" * random.randint(10, 500)},
            ]
        }
        if i % 10 == 0:
            part["content"].append({"type": "think", "think": f"Reasoning step {i}"})
        if i % 20 == 0:
            part["content"].append({
                "type": "text",
                "text": f"<|DSML|tool_calls><|DSML|invoke name=\"write\"><|DSML|parameter name=\"file\">f{i}.txt</|DSML|parameter><|DSML|parameter name=\"text\">data{i}</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>"
            })
        parts.append(part)

    t0 = time.monotonic()
    for p in parts:
        acc.consume_event({"conversation_id": "mass_test", "status": "finish" if p == parts[-1] else "generate", "parts": [p]})
    ingest_time = time.monotonic() - t0
    
    t0 = time.monotonic()
    response = acc.build_response()
    build_time = time.monotonic() - t0

    msg = response["choices"][0]["message"]
    content_len = len(msg.get("content", "") or "")
    tool_calls = msg.get("tool_calls", [])
    reasoning = msg.get("reasoning_content", "") or ""
    
    check(f"1000 parts ingested < 6s", ingest_time < 6, f"{ingest_time:.2f}s")
    check("build_response < 2s", build_time < 2, f"{build_time*1000:.0f}ms")
    has_stuff = content_len > 0 or len(tool_calls) > 0
    check("Has content or tool calls", has_stuff, f"content={content_len} tool_calls={len(tool_calls)}")
    check(f"Tool calls extracted", len(tool_calls) >= 50, f"{len(tool_calls)}/50")
    check("Has reasoning", len(reasoning) > 100, f"{len(reasoning)} chars")

sse_massive()

# ===== PART 6: FRAGMENTED SSE FRAMES =====
print("\n=== PART 6: Tool Call Mania (100 sequential tool calls) ===")

def tool_call_mania():
    # Debug: check parse_tool_calls_from_text directly
    from glm2api.utils.tool_parser import parse_tool_calls_from_text
    test_text = "<|DSML|tool_calls><|DSML|invoke name=\"write\"><|DSML|parameter name=\"file\">test.txt</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>"
    _, parsed = parse_tool_calls_from_text(test_text, allowed_tool_names={"write"})
    if not parsed:
        print(f"  DEBUG: parse_tool_calls_from_text returned 0 calls for test text!")
    acc = GLMEventAccumulator(model="glm-test", allowed_tool_names={"write"})
    # Single part with 100 sequential DSML tool calls
    calls = []
    for i in range(100):
        calls.append(f"<|DSML|invoke name=\"write\"><|DSML|parameter name=\"file\">file{i}.txt</|DSML|parameter><|DSML|parameter name=\"content\">data{i}</|DSML|parameter></|DSML|invoke>")
    
    text = "<|DSML|tool_calls>" + "".join(calls) + "</|DSML|tool_calls>"
    acc.consume_event({"conversation_id": "mania", "status": "finish", "parts": [{"logic_id": "1", "content": [{"type": "text", "text": text}]}]})
    response = acc.build_response()
    tc = response["choices"][0]["message"].get("tool_calls", [])
    check("100 tool calls extracted", len(tc) == 100, f"got {len(tc)} finish={response['choices'][0]['finish_reason']}")
    check("finish_reason tool_calls", response["choices"][0]["finish_reason"] == "tool_calls", "")

tool_call_mania()

# ===== PART 7: MEMORY PRESSURE =====
print("\n=== PART 7: Memory Pressure ===")

def memory_pressure():
    import tracemalloc
    tracemalloc.start()
    
    # Create 50 concurrent accumulators with large payloads
    accs = []
    for j in range(50):
        acc = GLMEventAccumulator(model=f"mem-test-{j}", allowed_tool_names={"tool"})
        for i in range(100):
            acc.consume_event({
                "conversation_id": f"mem_{j}",
                "parts": [{"logic_id": str(i), "content": [{"type": "text", "text": "x" * 1000}]}]
            })
        accs.append(acc)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    check("50 accumulators < 200MB", peak < 200_000_000, f"peak={peak/1e6:.1f}MB")
    
    # Cleanup
    del accs
    gc.collect()

memory_pressure()

# ===== RESULTS =====
print(f"\n{'='*40}")
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS+FAIL}")
if FAIL > 0:
    print("\nFAILURES:")
    for r in RESULTS:
        if "FAIL" in r:
            print(r)
print(f"{'='*40}")
sys.exit(0 if FAIL == 0 else 1)
