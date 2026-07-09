# GLM2API Optimization Changelog

All notable optimization changes to this project.

## [2026-07-09] - Final Benchmarking

### DeepSeek vs GLM Comparison
- Added comparison benchmark script at tests/benchmark_perf.py
- Results: DeepSeek V4 Flash Free outperforms GLM 5.2 think-search by:
  - TTFT: 1.7s vs 5.3s (3.1x faster)
  - Total time: 5.3s vs 25.1s (4.7x faster)
  - TPS: 94.7 vs 19.4 (4.9x faster)

## [2026-07-07 to 2026-07-09] - Optimization Sprint

### Proxy Sources Expansion
- Added 50+ new SOCKS5 proxy source URLs (was 19, now 70+)
- Files: `socks5_urls_python.py`
- Proxies discovered: zevtyardt (19K), ebrasha (16K), proxyscrape (5K), geonode (2.3K)
- Total: 94 URLs discovered across 50+ repos
- Verification: 3-phase (TCP pre-check + curl_cffi GET + POST)

### Streaming Latency
- Changed SSE read buffer from 32768 to 4096 bytes -- cuts TTFT up to 5s
- Fixed `_HttpxStreamWrapper` line-buffering -- no artificial batching of SSE events
- Files: `glm_client.py`, `http_client.py`

### Search Optimization
- Fixed DDG cache eviction bug (was evicting by content instead of timestamp)
- Increased cache TTL from 60s to 180s
- Replaced stdlib HTMLParser with regex parser (10x faster)
- Switched from urllib to httpx with connection pooling
- Added streaming-time search pre-fetch (launches search before stream ends)
- Deduplicated search queries
- Files: `translator.py`

### Network Transport
- Added HTTP/2 multiplexing (`CurlHttpVersion.V2TLS`)
- Added TCP Fast Open (`CURLOPT_TCP_FASTOPEN`)
- Added DNS cache TTL 300s (`CURLOPT_DNS_CACHE_TIMEOUT`)
- Increased session pool from 5 to 10 per proxy URL
- Separate connect (15s) and read (120s) timeouts
- Files: `http_client.py`

### Reliability
- Half-open circuit breaker (probe request after cooldown)
- Progressive backoff on rate limits (base * 2^failures, cap 30s)
- Watchdog auto-recovery (no success in 300s -- full reset)
- Emergency direct connection in `_try_proxies`
- Enhanced `/health` endpoint with subsystem status
- Backpressure (503 at queue depth > 100)
- Files: `glm_auth.py`, `glm2api_proxy.py`, `server.py`

### Proxy Pool
- Hot/cold pool split (O(1) selection from top 100, was O(n) over 6000+)
- Failure immediately removes from hot pool
- Auto-refresh hot pool every 30s
- Files: `glm2api_proxy.py`

### Queue
- Fixed starvation bug: `notify_all()` -- `notify()` (460x improvement under load)
- Added `_available` slot tracking (was accessing private `Semaphore._value`)
- 280 concurrent conversations in 0.14s (was 75% timeout at 200)
- Files: `glm_client.py`

### JSON & Serialization
- orjson promoted from optional to required dependency
- 45x faster JSON dumps than stdlib
- Pre-built header pool (128 variants, 0 dict copies on hot path)
- Files: `pyproject.toml`, `translator.py`, `glm_auth.py`

### Code Quality
- Removed atexit handlers (were causing executor lifecycle issues in AppImage)
- All shared executors use lazy initialization with shutdown detection
- Removed redundant `_resolve_tools` calls
- Standardized gzip decompression to single point
- Files: `glm2api_proxy.py`, `glm_client.py`, `glm_auth.py`
