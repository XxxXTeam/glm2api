# GLM2API Master Optimization Roadmap

**Created**: 2026-07-07
**Target**: Fastest, lowest-latency, highest-throughput GLM2API architecture

---

## Executive Summary

### Current State
- **Language**: Python 3.14+, stdlib-heavy (ponytail design philosophy)
- **Server**: `http.server.ThreadingHTTPServer` — thread-per-request
- **Upstream Client**: curl_cffi (Chrome 120 TLS impersonation) + httpx fallback
- **Proxy Pool**: SmartProxyPool with 6000+ SOCKS5 proxies, health-checked
- **Auth**: Guest token rotation with EWMA-weighted account selection
- **Queue**: Per-account semaphore queue (thread-based)
- **Streaming**: Blocking SSE parsing via `response.read()` + thread

### Current Bottlenecks (Ranked by Impact)

| # | Bottleneck | Impact | Current | Target |
|---|-----------|--------|---------|--------|
| 1 | ThreadingHTTPServer (GIL + stack overhead) | CRITICAL | ~280 threads = 2.2GB stacks | Async event loop = KB overhead |
| 2 | Per-call ThreadPoolExecutor in `_get_glm_opener` | HIGH | Creates threads for every stream | Reuse fixed pool |
| 3 | O(n) proxy pool scan on every `get_best()` | HIGH | 6000+ proxy scan per call | O(log n) hot pool |
| 4 | Sequential account failover in `_call_with_account_failover` | MEDIUM | Scans accounts serially | Parallel failover |
| 5 | Duplicate gzip decompression | MEDIUM | Decompressed in both http_client and glm_auth | Single decompress |
| 6 | urllib SSE parsing with select() + read() | MEDIUM | Double wait per chunk | Direct chunk read |
| 7 | Proxy routing duplicated in `_do_upstream_request` and `_get_glm_opener` | MEDIUM | Two independent fallback chains | Single unified chain |
| 8 | orjson optional dependency | LOW | Falls back to stdlib json | Make required |
| 9 | Per-request header dict copy in `get_browser_headers()` | LOW | New dict + gen X-Forwarded-For per call | Pre-generated headers |
| 10 | DDG search uses urllib (no connection pool) | LOW | New TCP connection per search | Reuse proxy pool |

### Expected Gains
- **Latency**: TTFT reduced 30-50% (async event loop, no thread switching)
- **Throughput**: 3-5x improvement (async I/O, 280 concurrent connections feasible)
- **Memory**: 60-80% reduction (no per-thread stacks)
- **Reliability**: Circuit breakers, exponential backoff, faster proxy rotation

---

## Tier 1: Quick Wins (Today)

### 1.1 Make orjson a Hard Dependency
- **Files**: `pyproject.toml`, `translator.py`, `tool_protocol.py`
- **Complexity**: Trivial
- **Expected**: ~2-3x JSON serialization speed
- **Risk**: None (orjson is pure C)

### 1.2 Pre-generate Full Headers Including Random IP
- **Files**: `glm_auth.py` — `get_browser_headers()`
- **Complexity**: Trivial
- **Expected**: ~0.1ms saved per request (1000s of requests = real savings)
- **Risk**: None

### 1.3 Remove Redundant select() in SSE Parser
- **Files**: `glm_client.py` — `_iter_sse_events()`
- **Complexity**: Low
- **Expected**: ~5-10ms per chunk reduced (no double syscall wait)
- **Risk**: Low (read() on streaming HTTP is already blocking)

### 1.4 Deduplicate Proxy Routing
- **Files**: `glm_auth.py` — `_do_upstream_request()`, `glm_client.py` — `_get_glm_opener()`
- **Complexity**: Low
- **Expected**: Single source of truth, fewer code paths
- **Risk**: Low (both functions do same thing, merge them)

### 1.5 Single gzip Decompression Point
- **Files**: `http_client.py` `do_request()`, `glm_auth.py` `read_json_response()`
- **Complexity**: Low
- **Expected**: No redundant decompression
- **Risk**: Low

---

## Tier 2: Architecture Improvements (This Week)

### 2.1 ThreadPoolExecutor Reuse
- **Files**: `glm_client.py` — `_get_glm_opener()`, `_upload_referenced_files()`
- **Complexity**: Medium
- **Expected**: No thread creation overhead per request
- **Risk**: Low

### 2.2 Hot/Cold Proxy Pool Split
- **Files**: `glm2api_proxy.py` — `SmartProxyPool.get_best()`
- **Complexity**: Medium
- **Expected**: O(1) proxy selection instead of O(n) scan
- **Risk**: Low

### 2.3 Parallel Account Failover
- **Files**: `glm_client.py` — `_call_with_account_failover()`
- **Complexity**: Medium
- **Expected**: ~3x faster failover (try 2-3 accounts concurrently)
- **Risk**: Low (each try is independent)

### 2.4 Streaming Response Parser Optimization
- **Files**: `glm_client.py` — `_iter_sse_events()`
- **Complexity**: Medium
- **Expected**: ~20% faster SSE parsing (use memoryview for byte scanning)
- **Risk**: Low

### 2.5 HTTP/2 for Upstream (curl_cffi supports it)
- **Files**: `http_client.py`
- **Complexity**: Medium
- **Expected**: Multiplexed connections, reduced connection count
- **Risk**: Medium (depends on GLM server supporting HTTP/2)

---

## Tier 3: Async Migration (Major Restructuring)

### 3.1 Replace ThreadingHTTPServer with aiohttp
- **Files**: `server.py`, `app.py`
- **Complexity**: High
- **Expected**: 280+ concurrent connections feasible, zero thread overhead
- **Risk**: High (requires testing all request paths)

### 3.2 Async Queue with asyncio.Semaphore
- **Files**: `glm_client.py` — `ConcurrentRequestQueue`
- **Complexity**: Medium
- **Expected**: No thread blocking, event-loop friendly
- **Risk**: Medium

### 3.3 Async Proxy Pool
- **Files**: `glm2api_proxy.py`
- **Complexity**: Medium
- **Expected**: No blocking locks in event loop
- **Risk**: Medium

---

## Implementation Order

```
Day 1: T1.1 → T1.2 → T1.3 → T1.4 → T1.5  (all quick wins)
Day 2: T2.1 → T2.2 → T2.3 → Benchmark
Day 3: T2.4 → T2.5 → T3.1 start (async server)
Day 4-5: T3.1 completion → T3.2 → T3.3 → Benchmark
```

## Rollback Strategy
Each change is atomic (single file, focused scope). Rollback = revert single commit.
All changes preserve API compatibility — no client-facing changes.
