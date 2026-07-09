# AI Subagent Handoff Document

## Project Overview

GLM2API is a reverse proxy that converts Zhipu Qingyan's (chatglm.cn) web API into an OpenAI-compatible Chat Completions API. It uses curl_cffi Chrome 120 TLS impersonation to bypass Alibaba Cloud WAF bot detection, manages a pool of thousands of SOCKS5 proxies for IP rotation, dynamically creates guest token sessions (each with ~5 searches / ~8 tool calls of free quota), and routes requests through a per-account semaphore queue with EWMA-weighted account selection. The architecture uses a sync server (stdlib `ThreadingHTTPServer`) as production workhorse and an async server (aiohttp) available via `--async` flag for incremental migration. The service layer consists of an auth/token manager, GLM web client, smart proxy pool, SSE streaming parser with DuckDuckGo fallback search, and unified HTTP transport via curl_cffi SessionPool with HTTP/2, TCP Fast Open, and DNS caching.

## Architecture

### Directory Structure

```
/home/uluru/ZCodeProject/glm2api/
├── src/glm2api/
│   ├── __init__.py                  # Package init, version
│   ├── __main__.py                  # Entry point (main()), parses --async flag
│   ├── app.py                       # Application factory (create_application, create_async_application)
│   ├── config.py                    # AppConfig dataclass, .env parsing, token file loading
│   ├── server.py                    # GLM2APIServer (sync, ThreadingHTTPServer, SSE streaming, backpressure)
│   ├── server_async.py              # AsyncGLM2APIServer (aiohttp, SSE streaming via run_in_executor)
│   ├── logging_utils.py             # TUI-colored logging setup, RotatingFileHandler for DEBUG
│   ├── model_profiles.py            # ModelProfile dataclass, MODEL_PROFILES dict
│   ├── model_variants.py            # split_model_features, expand_model_variants (think/search suffixes)
│   ├── services/
│   │   ├── __init__.py              # Empty, service layer marker
│   │   ├── glm_auth.py              # GLMAccessTokenManager: token refresh, circuit breaker, EWMA selection, watchdog, warm spares
│   │   ├── glm_client.py            # GLMWebClient: chat_completion, stream_chat_completion, SSE parser, queue, account failover
│   │   ├── glm2api_proxy.py         # SmartProxyPool: 6000+ SOCKS5 proxies, hot/cold pool, 3-phase verification, unified _try_proxies
│   │   ├── http_client.py           # Unified HTTP client: curl_cffi SessionPool, httpx fallback, streaming wrappers
│   │   ├── translator.py            # convert_messages, DDG search (regex parser + httpx), GLMEventAccumulator, context management
│   │   ├── anthropic_adapter.py     # anthropic_to_openai / openai_to_anthropic_response
│   │   └── responses_adapter.py     # responses_to_openai / openai_to_responses
│   └── utils/
│       ├── __init__.py
│       ├── tool_parser.py           # StreamingToolParser (XML DSML tool call extraction)
│       └── tool_protocol.py         # filter_tools, build_tool_call_instructions, serialize_tool_call_block
├── config/
│   ├── guest_tokens.txt             # Cached guest tokens
│   └── verified_proxies.txt         # Cached verified proxies
├── scripts/
│   ├── create_guest_tokens.py       # Pre-create guest tokens
│   ├── find_proxies.py              # Find public proxies
│   ├── find_chinese_proxies.py      # Find China-region proxies
│   ├── verify_proxies.py            # Verify proxies against chatglm.cn
│   └── vpn_watchdog.py              # External VPN watchdog
├── tests/
│   ├── benchmark_final.py           # Final benchmark
│   ├── benchmark_perf.py            # Performance benchmark (DeepSeek vs GLM comparison)
│   ├── benchmark_tps.py             # TPS benchmark
│   ├── stress_glm_live.py           # Live stress test
│   ├── stress_mass_test.py          # Mass concurrent stress test
│   └── test_*.py                    # Unit tests
├── docs/
│   ├── AI_SUBAGENT_HANDOFF.md       # This file
│   ├── OPTIMIZATION_CHANGELOG.md    # Optimization history
│   └── OPTIMIZATION_ROADMAP.md      # Optimization roadmap (historical reference)
├── pyproject.toml                   # Project config, dependencies (orjson, httpx, aiohttp, uvloop)
├── proxy_sources.py                 # Independent proxy source definitions
├── socks5_urls_python.py            # Python-parsed socks5 URLs
├── socks5_sources_report.txt        # Proxy source discovery report
└── README.md                        # Setup and usage instructions
```

### Key Components

| Component | File | Class/Function | Role |
|-----------|------|---------------|------|
| Entry Point | `/home/uluru/ZCodeProject/glm2api/src/glm2api/__main__.py` | `main()` | CLI entry, parses `--async` flag, creates app, runs it |
| Application Factory | `/home/uluru/ZCodeProject/glm2api/src/glm2api/app.py` | `Application`, `create_application()`, `create_async_application()` | Wires config, client, server; signal handling; graceful shutdown |
| Config | `/home/uluru/ZCodeProject/glm2api/src/glm2api/config.py` | `AppConfig`, `load_config()` | Loads .env + token file, guest mode detection, model alias expansion |
| Sync Server | `/home/uluru/ZCodeProject/glm2api/src/glm2api/server.py` | `GLM2APIServer`, `RequestHandler` | ThreadingHTTPServer, handles /health, /v1/chat/completions, /v1/images, /v1/messages, /v1/responses; CORS; API key auth; backpressure (503 at queue depth >100); TCP_NODELAY |
| Async Server | `/home/uluru/ZCodeProject/glm2api/src/glm2api/server_async.py` | `AsyncGLM2APIServer` | aiohttp-based, same endpoints, blocking calls via run_in_executor, SSE streaming, keep-alive heartbeat for Responses API |
| Auth Manager | `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm_auth.py` | `GLMAccessTokenManager` | Token refresh (guest + authenticated), EWMA-weighted account selection, fill-first strategy, circuit breaker, sliding refresh, warm spare pool, watchdog auto-recovery, heartbeat |
| Proxy Pool | `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm2api_proxy.py` | `SmartProxyPool`, `ProxyScore`, `_try_proxies()` | 6000+ SOCKS5 proxies via 50+ public sources, hot/cold pool (O(1) selection), 3-phase progressive verification (TCP pre-check + curl_cffi GET + POST), concurrent racing fallback chain, geo-location bonus, VPN reconnect |
| GLM Client | `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm_client.py` | `GLMWebClient`, `ConcurrentRequestQueue` | chat_completion, stream_chat_completion, generate_images; per-account semaphore queue (5 slots each); account failover with spawn; file upload executor; batch conversation deletion |
| HTTP Client | `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/http_client.py` | `SessionPool`, `StreamingResponseWrapper`, `_HttpxStreamWrapper` | curl_cffi SessionPool (10 per proxy URL, Chrome 120 JA3), HTTP/2 via CurlHttpVersion.V2TLS, TCP_NODELAY, TCP_FASTOPEN, DNS cache TTL 300s; httpx fallback with TLS 1.2 forced; gzip decompression |
| Translator | `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/translator.py` | `convert_messages()`, `GLMEventAccumulator`, `_run_web_search()` | OpenAI-to-GLM message conversion, DDG search with regex parser + httpx connection pool, streaming reasoning delta extraction, server-side tool call merging, search pre-fetch, context sliding window |
| Anthropic Adapter | `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/anthropic_adapter.py` | `anthropic_to_openai()`, `AnthropicStreamAccumulator` | Anthropic Messages -> OpenAI chat/completions conversion, stream accumulator |
| Responses Adapter | `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/responses_adapter.py` | `responses_to_openai()`, `ResponsesStreamAccumulator` | OpenAI Responses -> OpenAI chat/completions conversion, stream accumulator |

### Data Flow

```
┌─────────┐     POST /v1/chat/completions     ┌──────────────────────┐
│ Client  │ ────────────────────────────────►  │  GLM2APIServer       │
│ (OpenAI │     Authorization: Bearer XXX      │  (ThreadingHTTPServer)│
│  SDK)   │     {model, messages, stream, ...}  │  or AsyncGLM2APIServer│
└─────────┘                                    └───────┬──────────────┘
                                                        │
                                                        ▼
                                         ┌──────────────────────────┐
                                         │  GLMWebClient            │
                                         │  - auth check (API keys) │
                                         │  - backpressure check    │
                                         │  - ConcurrentRequestQueue│
                                         │    (per-account semaphore)│
                                         └───────────┬──────────────┘
                                                      │
                                                      ▼
                          ┌───────────────────────────────────────────┐
                          │  convert_messages()                       │
                          │  - context sliding window (30 msg limit)  │
                          │  - filter tools, sanitize tool calls      │
                          │  - merge into single text prompt          │
                          │  - pre-fetch DDG search (background)      │
                          └───────────────────┬───────────────────────┘
                                               │
                                               ▼
                          ┌───────────────────────────────────────────┐
                          │  _open_chat_stream()                      │
                          │  1. acquire account: EWMA/fill_first      │
                          │  2. get access token (refresh if expired) │
                          │  3. build GLM request (SSE stream POST)   │
                          │  4. send via _glm_opener                  │
                          └───────────────────┬───────────────────────┘
                                               │
                                               ▼
                     ┌───────────────────────────────────────────────────┐
                     │  _try_proxies() (concurrent racing)               │
                     │  1. Pool SOCKS5 (3x, weighted random)    ◄───────│ ───── SmartProxyPool
                     │  2. Env proxies (GLM_PROXY_LIST)           ●     │       (6000+ proxies)
                     │  3. Local VPN tunnels (127.0.0.1:7928-7932)●     │
                     │  4. Direct connection (brief 3s timeout)   ●──── │
                     │  5. Fallback: sequential pool retry        ●──── │
                     │  6. Emergency direct (full timeout)        ●──── │
                     └──────────────┬────────────────────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────────┐
                     │  http_client.do_request()         │
                     │  - curl_cffi Session (Chrome 120) │
                     │    HTTP/2 + TCP Fast Open + DNS   │
                     │  - or httpx fallback (TLS 1.2)    │
                     │  - gzip decompression             │
                     └──────────────┬───────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────────┐
                     │  chatglm.cn (upstream)            │
                     │  SSE stream: events with parts[]  │
                     └──────────────┬───────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────────────┐
                     │  _iter_sse_events()                    │
                     │  - read(4096) from streaming response  │
                     │  - split on \n\n for SSE frames        │
                     │  - orjson parse each data: payload     │
                     └──────────────┬────────────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────────────┐
                     │  GLMEventAccumulator.consume_event()    │
                     │  - accumulate parts by logic_id         │
                     │  - compute text/reasoning deltas        │
                     │  - extract tool calls from reasoning    │
                     │  - streaming tool call parsing (XML)    │
                     │  - background DDG search pre-fetch      │
                     └──────────────┬────────────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────────────┐
                     │  SSE chunk emitted as bytes             │
                     │  data: {"choices":[{"delta":{"content"  │
                     │  data: [DONE]                           │
                     └──────────────┬────────────────────────┘
                                    │
                                    ▼
                     ┌────────────────────────────────────────┐
                     │  Client receives SSE stream             │
                     └────────────────────────────────────────┘
```

## Completed Optimizations

Every optimization is listed with file path, approximate line number, and description.

1. **orjson required** (`/home/uluru/ZCodeProject/glm2api/pyproject.toml` line 14, `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/translator.py` line 63-68)
   - Promoted orjson from optional to hard dependency. 45x faster JSON dumps than stdlib json. Used everywhere: SSE framing, API responses, message conversion, configuration parsing.

2. **Pre-built header pool** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm_auth.py` lines 114-133, 557-605)
   - `_HeaderPool` class at line 114 pre-builds 128 header dict variants with different `X-Forwarded-For` IPs. `get_browser_headers()` at line 559 returns the next pre-built dict from the rotating pool -- zero dict copies on the hot path.

3. **SSE parser optimized** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm_client.py` lines 1009-1078)
   - `_READ_SIZE = 4096` (line 1014) instead of 32768, cuts TTFT up to 5s. SSE splitting uses `pending.split(b"\n\n")` for O(n) single-pass parsing instead of repeated `find()`. Earlier version removed redundant `select()` call.

4. **Unified proxy routing** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm2api_proxy.py` lines 1433-1531)
   - `_try_proxies()` at line 1433 races all proxy candidates concurrently (pool SOCKS5, env proxies, VPN tunnels, direct) via ThreadPoolExecutor + `wait(FIRST_COMPLETED)`. Falls back to sequential pool retry, then emergency direct connection. Replaces two independent fallback chains in `glm_auth._do_upstream_request` and `glm_client._get_glm_opener`.

5. **Gzip standardization** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/http_client.py` lines 354-363, 400-408; `glm_auth.py` lines 610-617; `glm_client.py` lines 889-893, 1212-1214)
   - Gzip decompression happens in one place (`do_request()` in http_client.py) and removes the `Content-Encoding` header so downstream code never double-decompresses. All callers now expect plain (decompressed) bytes.

6. **Shared ThreadPoolExecutors** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm_client.py` lines 59-70, `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm_auth.py` lines 100-111, `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm2api_proxy.py` lines 29-41)
   - Three shared executors with lazy init + shutdown detection: `_get_upload_executor()` (10 workers, file uploads), `_get_prewarm_executor()` (5 workers, token prewarm), `_get_proxy_executor()` (8 workers, proxy racing). Avoids per-call pool create/teardown. No atexit handlers (caused "cannot schedule new futures after interpreter shutdown" in AppImage Python).

7. **Hot/cold proxy pool** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm2api_proxy.py` lines 217-224, 1066-1123)
   - `SmartProxyPool` maintains a hot pool of top 100 proxies by score (line 219). `get_best()` at line 1085 does O(1) weighted random selection from hot pool. Background thread refreshes every 30s (line 1074). Fallback to O(n) full scan when hot pool is empty. Failures immediately evict from hot pool (line 1331-1332).

8. **EWMA selection deduplication** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm_auth.py` lines 708-741, 1135-1196)
   - `_select_ewma_account()` at line 708 uses power-of-two-choices: picks 2 random accounts from usable pool, compares EWMA latency with jitter. `acquire_account_for_stream()` at line 1135 integrates find-usable + get-cached-token in one lock acquisition. EWMA alpha = 0.125.

9. **Async server shell** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/server_async.py`, `/home/uluru/ZCodeProject/glm2api/src/glm2api/app.py`, `/home/uluru/ZCodeProject/glm2api/src/glm2api/__main__.py`)
   - `AsyncGLM2APIServer` at `/home/uluru/ZCodeProject/glm2api/src/glm2api/server_async.py` provides same endpoints via aiohttp. Blocking GLM client calls forwarded to `run_in_executor`. Launched via `glm2api --async`. Uses `uvloop.install()` in __main__.py line 63.

10. **Queue starvation fix** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm_client.py` lines 175-237)
    - `ConcurrentRequestQueue` uses threading.Condition with `notify()` (not `notify_all()`) to wake one waiter per release. Added `_available` slot tracking (line 188) instead of accessing private `Semaphore._value`. Supports 280 concurrent conversations in 0.14s (previously 75% timeout at 200).

11. **Streaming-time search pre-fetch** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/translator.py` lines 729-752, 919-921)
    - `_prefetch_search_from_tool_call()` at line 729 launches background DDG search cache fill when tool call is detected during streaming. `_cache_search_query()` at line 70 starts a daemon thread. Search results are ready by the time `finalize()` calls `_execute_retrieve_tool_calls()`.

12. **DDG regex parser** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/translator.py` lines 139-159)
    - `_DDG_LINK_RE` and `_DDG_SNIPPET_RE` at lines 139-140 parse DuckDuckGo Lite HTML with regex (~10x faster than stdlib HTMLParser). Fallback to `_DDGLiteParser` (HTMLParser subclass) when regex fails.

13. **HTTP/2 + TCP FastOpen + DNS cache** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/http_client.py` lines 96-133)
    - `SessionPool._create_session()` at line 96 enables `CurlHttpVersion.V2TLS` (HTTP/2 multiplexing), `CURLOPT_TCP_NODELAY` (line 102), `CURLOPT_DNS_CACHE_TIMEOUT=300` (line 103), `CURLOPT_TCP_FASTOPEN` (line 104). Session pool size = 10 per proxy URL (line 60). Separate connect (15s) and read (120s) timeouts (line 110).

14. **Circuit breaker improvements** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm_auth.py` lines 174-272, `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm2api_proxy.py` lines 1334-1347)
    - Half-open circuit breaker (probe request after cooldown, line 659-676 in glm_auth.py). Progressive backoff on rate limits (base * 2^failures, cap 30s, line 758-762). Proxy pool `report_rate_limited()` at line 1334 in glm2api_proxy.py with 3x consecutive failure penalty.

15. **Watchdog auto-recovery** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm_auth.py` lines 1221-1278)
    - `_start_watchdog()` at line 1221 runs every 60s (line 1234). If no success in 300s (line 1232) or pool is starved, triggers `_aggressive_recovery()` at line 1255 which resets circuit breaker, clears all rate limits/cooldowns, forces full guest token refresh.

16. **Backpressure** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/server.py` lines 193-210)
    - Before accepting any POST request, checks `request_queue.queue_depth >= 100`. Returns 503 with `type: "backpressure"` and current queue depth if overloaded. Also checks `auth.is_starved()` to reject when no accounts are usable.

17. **`_READ_SIZE 4096`** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm_client.py` line 1014)
    - Reduced from 32768 to 4096 bytes for SSE reads. Cuts TTFT by up to 5s because smaller reads from the streaming response deliver the first event faster.

18. **Proxy sources expanded** (`/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm2api_proxy.py` lines 50-120)
    - Added 50+ public SOCKS5/HTTP proxy source URLs (was ~19, now 70+). Sources include GitHub raw repos, proxyscrape v2/v3 APIs, CDN jsdelivr mirrors, region-specific endpoints. 3-phase progressive verification: TCP pre-check (200 workers, 3s timeout) -> curl_cffi GET -> curl_cffi POST.

## Remaining Bottlenecks / Future Work

1. **True async migration of GLMWebClient**: Currently `AsyncGLM2APIServer` forwards all blocking calls via `run_in_executor`. The GLM client (glm_client.py), auth manager (glm_auth.py), and SSE parser all use blocking I/O. True async would use `httpx.AsyncClient` for all upstream calls, eliminating the thread pool overhead for each request.

2. **Direct connection fallback**: When the proxy pool is completely starved (all proxies blacklisted or rate-limited), the system should attempt direct connections more aggressively rather than waiting for proxy replenishment. The `_try_proxies()` function already has an emergency direct connection path (line 1523-1528) but it's a last resort with limited timeout. Consider making this a first-class option with configurable priority.

3. **Proxy pool size scaling**: Auto-fetch target should be increased to 200+ verified proxies. Current `min_working=50` (line 642) is conservative. With 50+ sources yielding ~48,000 candidates, the pipeline should target 500+ verified proxies for better IP diversity and lower WAF detection rates.

4. **Think mode optimization**: The `glm-zero-preview` model takes ~79s for reasoning. Investigate parallel token fetching from multiple SSE connections (speculative decoding), or reducing the context sent to think models. The current approach waits for a single SSE stream to complete.

5. **Windows compatibility**: `uvloop` (used in async mode) does not work on Windows. Need a platform check or alternative asyncio event loop (e.g., `asyncio.SelectorEventLoop`) when running on Windows.

6. **Parallel account failover**: `_call_with_account_failover()` (glm_client.py line 1265) tries accounts sequentially. Could try 2-3 accounts concurrently and take the first successful result, reducing failover latency.

## AI Agent Prompts

Copy-paste these prompts for subagents to continue specific work:

- **"Benchmark TTFT and TPS for all models"**: `cd /home/uluru/ZCodeProject/glm2api && python -m pytest tests/benchmark_perf.py -v --log-cli-level=INFO 2>&1 | tail -50` then analyze output. Compare glm-4-flash, glm-4.7, glm-5.2, glm-zero-preview for both streaming and non-streaming.

- **"Profile convert_messages CPU usage"**: `cd /home/uluru/ZCodeProject/glm2api && python -m cProfile -s cumulative -m glm2api.__main__ 2>&1 | head -80`. The profile will show whether `convert_messages()` or `sanitize_tool_calls()` dominates per-request CPU. Look for opportunities to compile regex patterns or cache results.

- **"Investigate proxy WAF bypass techniques"**: Read `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/http_client.py`, `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm2api_proxy.py`, and `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm_auth.py`. The WAF (Alibaba Cloud) detects and blocks proxies after ~10-15 requests. Research: rotating User-Agent per request, varying TLS fingerprint between Chrome 120/131/safari15_3, adding random delays between requests from same IP, using residential proxy networks instead of datacenter proxies.

- **"Add new proxy sources from socks5_urls_python.py"**: Read `/home/uluru/ZCodeProject/glm2api/socks5_urls_python.py` and merge new URL sources into `_PUBLIC_SOCKS5_URLS` in `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm2api_proxy.py`. The file has 94 URLs already discovered -- verify each against current sources and add any missing ones. Re-verify all sources for liveness.

- **"Optimize GLM think mode latency"**: The `glm-zero-preview` and `glm-4.1v-thinking-flashx` models use reasoning (thinking) blocks. Investigate: can the SSE stream be parsed to extract reasoning tokens earlier? Can we reduce context sent to think models via aggressive pruning? Is parallel token fetching (opening multiple SSE connections for the same request) feasible for speculation?

## Key Decision Records

1. **Why curl_cffi**: Python's default TLS fingerprint (httpx/requests) is detected and blocked by Alibaba Cloud WAF. `curl_cffi` impersonates Chrome 120's JA3/JA4 fingerprint exactly, bypassing bot detection. Tested with curl_cffi 0.15.0+ through SOCKS5 proxies. Fallback to httpx with forced TLS 1.2 exists when curl_cffi is not installed. (See `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/http_client.py` lines 1-15)

2. **Why orjson is required**: 45x faster JSON serialization than stdlib json on Python 3.14. Used for all SSE framing, API responses, message conversion, and config parsing. The C extension avoids GIL for JSON operations. Declared as hard dependency in `pyproject.toml` line 14 with comment "10x faster JSON (required, not optional)".

3. **Why we keep both sync and async servers**: The sync server (ThreadingHTTPServer) is the production workhorse with proven reliability. The async server (aiohttp) exists for incremental migration -- blocking GLM client calls run via `run_in_executor`, so the event loop stays responsive during transition. Full async migration requires rewriting the entire GLM client stack (glm_auth, glm_client, SSE parser) to use `httpx.AsyncClient`. The `--async` flag in `__main__.py` switches between them.

4. **Why hot pool size is 100**: Empirical measurement showed the top 100 proxies by score cover 99% of `get_best()` selections. Hot pool O(1) selection eliminates the O(n) scan over 6000+ proxies. Refresh interval of 30s keeps scores current without excessive CPU. (See `/home/uluru/ZCodeProject/glm2api/src/glm2api/services/glm2api_proxy.py` line 219)

5. **Why proxy rotation is aggressive (2 successes)**: Public SOCKS5 proxies get WAF-blocked after ~10-15 requests. By rotating after 2 successes (line 128 in `ProxyScore`), the system stays ahead of WAF blocks while maintaining throughput. Premium (authenticated) proxies get 10 successes. Local VPN tunnels get 50 successes.

6. **Why EWMA with power-of-two-choices**: Simple round-robin doesn't account for account latency differences. EWMA tracks sliding average latency per account. Power-of-two-choices sampling (pick 2 random, choose the lower-latency one) provides O(1) selection with good distribution, avoiding both stampede (all hitting the best account) and the complexity of full weighted-round-robin.

## Common Pitfalls

1. **"cannot schedule new futures after interpreter shutdown"**: This error occurs when ThreadPoolExecutors have atexit handlers and the interpreter is shutting down. All shared executors in this project use lazy initialization with shutdown detection (`if _POOL is None or _POOL._shutdown`) and no atexit handlers. See shared executor patterns in `glm_client.py` line 63-69, `glm_auth.py` line 104-111, `glm2api_proxy.py` line 34-41.

2. **Proxies get WAF-blocked after ~10-15 requests**: Alibaba Cloud WAF detects repeated requests from the same IP. The `SmartProxyPool` auto-rotates after 2 successes by default (line 128 in `ProxyScore`), and `report_rate_limited()` at line 1334 applies a 3x consecutive failure penalty and exponential backoff cooldown. If proxies are blocked too quickly, try increasing `_HEADER_POOL_SIZE` in `glm_auth.py` line 98, or switching User-Agent rotation.

3. **Guest tokens expire every ~55min**: Guest access tokens from `chatglm.cn/user-api/guest/access` have a 3600s (1 hour) TTL. The `GLMAccessTokenManager` handles this with sliding refresh (line 801-830) running every 5 minutes, refreshing tokens with <30 min remaining. Access tokens also use randomized 10-30s early expiry buffer (line 1010) to prevent edge-of-expiry race conditions.

4. **Port 8000 already in use**: The default port is 8000. Check with `ss -tlnp | grep 8000` or `fuser 8000/tcp`. Kill the old process with `fuser -k 8000/tcp` or `kill <PID>`. Change port via `PORT=8001 glm2api` in `.env`. The error message will say "端口已被占用: 127.0.0.1:8000" (see `app.py` line 148).

5. **Memory leak from unclosed SSE streams**: Each streaming request opens a file descriptor via `_glm_opener().open()`. The `generate()` function in `stream_chat_completion` has a `finally` block that calls `response.close()` and `self.delete_conversation()`. If the generator is garbage-collected without being fully consumed (e.g., client disconnects), the `__del__` method may not run promptly. The `_close_after_stream()` method in `server.py` line 561 sets `close_connection = True` which triggers proper cleanup in `handle_one_request` override (line 565).

6. **Deadlock in RLock**: `SmartProxyPool` uses `threading.RLock()` (line 210) not `threading.Lock()` because `get_best()`/`get_next()` call `_auto_replenish()` which also takes the lock. A regular Lock would deadlock. Similarly, `GLMAccessTokenManager` uses `threading.RLock()` at line 161.
