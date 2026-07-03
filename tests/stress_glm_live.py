"""Mass live test: concurrent GLM 5.2 think-search through proxy.
Measures TPS, latency percentiles, error rates, tool handling.
cavekit: spec-first. ponytail: measure once, fix once.
"""
from __future__ import annotations
import sys, os, time, json, statistics, threading, concurrent.futures
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import httpx

BASE = "http://127.0.0.1:8000/v1"
AUTH = {"Authorization": "Bearer sk-local"}

PASS = 0; FAIL = 0; RESULTS = []
def check(name, ok, detail=""):
    global PASS, FAIL
    if ok: PASS += 1; RESULTS.append(f"  PASS {name}")
    else: FAIL += 1; RESULTS.append(f"  FAIL {name}  {detail}")

# ---- 1. Basic health ----
print("\n=== 1. Health & Models ===")
r = httpx.get(f"{BASE.replace('/v1','')}/health", timeout=5)
check("Health endpoint", r.status_code == 200)
models = httpx.get(f"{BASE}/models", timeout=5).json()
check("78 models exposed", len(models["data"]) == 78)

# ---- 2. Sequential latency benchmark ----
print("\n=== 2. Sequential Latency (GLM-5.2 basic) ===")
lats = []
for i in range(3):
    t0 = time.monotonic()
    r = httpx.post(f"{BASE}/chat/completions",
        json={"model":"glm-5.2","messages":[{"role":"user","content":"Say 3 words"}],"max_tokens":20},
        headers=AUTH, timeout=60)
    elapsed = time.monotonic() - t0
    lats.append(elapsed)
    if r.status_code == 200:
        check(f"Request {i+1} OK", True)
    else:
        check(f"Request {i+1} OK", False, f"HTTP {r.status_code}")

if lats:
    print(f"  Latencies: {[f'{l:.1f}s' for l in lats]}")
    print(f"  Avg: {statistics.mean(lats):.1f}s  Min: {min(lats):.1f}s  Max: {max(lats):.1f}s")

# ---- 3. GLM 5.2 think-search ----
print("\n=== 3. GLM-5.2-Think-Search ===")
for i in range(2):
    t0 = time.monotonic()
    r = httpx.post(f"{BASE}/chat/completions",
        json={"model":"glm-5.2-think-search","messages":[{"role":"user","content":"What is AI? 2 sentences"}],"max_tokens":100},
        headers=AUTH, timeout=60)
    elapsed = time.monotonic() - t0
    if r.status_code == 200:
        data = r.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content","")
        reasoning = msg.get("reasoning_content","")
        tools = msg.get("tool_calls",[])
        check(f"Think-search {i+1} OK (200)", True)
        check(f"  Has content", len(content) > 0, f"{len(content)} chars")
        check(f"  Has reasoning", len(reasoning) > 0, f"{len(reasoning)} chars")
        print(f"  Time: {elapsed:.1f}s  Content: {len(content)}c  Reasoning: {len(reasoning)}c  Tools: {len(tools)}")
    else:
        check(f"Think-search {i+1}", False, f"HTTP {r.status_code} {r.text[:100]}")

# ---- 4. Concurrent throughput ----
print("\n=== 4. Concurrent (5 parallel, glm-5.2 basic) ===")
def single_req(i):
    try:
        t0 = time.monotonic()
        r = httpx.post(f"{BASE}/chat/completions",
            json={"model":"glm-5.2","messages":[{"role":"user","content":"Say 2 words"}],"max_tokens":10},
            headers=AUTH, timeout=60)
        return (r.status_code, time.monotonic() - t0, i)
    except Exception as e:
        return (0, 0, str(e))

t0 = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
    futures = [ex.submit(single_req, i) for i in range(5)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]
elapsed = time.monotonic() - t0

ok_count = sum(1 for s,_,_ in results if s == 200)
total_time = sum(l for _,l,_ in results if l > 0)
avg_lat = total_time / max(ok_count, 1) if ok_count else 0
check(f"5 concurrent: {ok_count}/5 OK", ok_count == 5, f"{ok_count} ok")
if ok_count > 0:
    tps = ok_count / elapsed
    print(f"  Wall time: {elapsed:.1f}s  Avg latency: {avg_lat:.1f}s  TPS: {tps:.2f}")

# ---- 5. Rate limit test ----
print("\n=== 5. Rate Limit Test (quick burst) ===")
burst_results = []
t0 = time.monotonic()
for i in range(3):
    r = httpx.post(f"{BASE}/chat/completions",
        json={"model":"glm-5.2","messages":[{"role":"user","content":"Hi"}],"max_tokens":5},
        headers=AUTH, timeout=60)
    burst_results.append(r.status_code)
burst_ok = sum(1 for s in burst_results if s == 200)
burst_too_many = sum(1 for s in burst_results if s == 429)
check("Burst requests OK", burst_ok >= 2, f"{burst_ok}/3 ok, {burst_too_many} rate-limited")

# ---- 6. Streaming test ----
print("\n=== 6. Streaming glm-5.2-think-search ===")
try:
    t0 = time.monotonic()
    r = httpx.post(f"{BASE}/chat/completions",
        json={"model":"glm-5.2-think-search","messages":[{"role":"user","content":"Count 1-3"}],"max_tokens":50,"stream":True},
        headers=AUTH, timeout=60)
    chunks = 0; reasoning_content = False; tool_calls = False; final = ""
    for line in r.iter_lines():
        if not line: continue
        if line == "data: [DONE]": break
        if line.startswith("data: "):
            chunks += 1
            try:
                d = json.loads(line[6:])
                delta = d.get("choices",[{}])[0].get("delta",{})
                if "reasoning_content" in delta: reasoning_content = True
                if "tool_calls" in delta: tool_calls = True
                if delta.get("content"): final += delta["content"]
            except: pass
    elapsed = time.monotonic() - t0
    check("Streaming works", chunks > 0, f"{chunks} chunks in {elapsed:.1f}s")
    check("Streaming has reasoning content", reasoning_content, "")
except Exception as e:
    check("Streaming", False, str(e)[:100])

# ---- RESULTS ----
print(f"\n{'='*50}")
print(f"MASS TEST RESULTS: {PASS} passed, {FAIL} failed")
if FAIL > 0:
    for r in RESULTS:
        if "FAIL" in r: print(r)
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
