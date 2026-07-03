"""TPS benchmark: GLM 5.2 Search Think (local proxy) vs DeepSeek V4 Flash (OpenCode Zen).
cavekit: spec-first. ponytail: one-shot measurement. superpowers: sub-agents for both.
"""
from __future__ import annotations
import sys, os, time, json, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import httpx

GLM_BASE = "http://127.0.0.1:8000/v1"
GLM_AUTH = {"Authorization": "Bearer sk-local"}
ZEN_BASE = "https://opencode.ai/zen/v1"
ZEN_AUTH = {"Authorization": "Bearer sk-placeholder"}

PROMPT = """Design a globally distributed order processing system handling 50,000 orders/minute across 12 data centers.
Requirements: real-time inventory sync (<100ms), optimal fulfillment routing, fraud detection (<200ms added latency),
99.995% uptime, multi-cloud cost optimization (AWS/GCP/Azure).

Use sub-agents for each sub-problem, then synthesize. Be specific about technologies, trade-offs, costs."""

results = {}

def test_model(name, base_url, headers, model, stream=False, **extra):
    print(f"\n=== {name} ===")
    try:
        payload = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
                   "max_tokens": 4096}
        if stream:
            payload["stream"] = True
        payload.update(extra)
        
        t0 = time.monotonic()
        ttft = None
        total_chars = 0
        reasoning_chars = 0
        
        if stream:
            with httpx.Client(timeout=300) as client:
                with client.stream("POST", f"{base_url}/chat/completions", json=payload, headers=headers) as r:
                    for line in r.iter_lines():
                        if not line: continue
                        if line == "data: [DONE]": break
                        if line.startswith("data: "):
                            try:
                                d = json.loads(line[6:])
                                delta = d.get("choices", [{}])[0].get("delta", {})
                                if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                                    ttft = time.monotonic() - t0
                                if delta.get("content"): total_chars += len(delta["content"])
                                if delta.get("reasoning_content"): reasoning_chars += len(delta["reasoning_content"])
                            except: pass
        else:
            r = httpx.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=300)
            data = r.json()
            msg = data["choices"][0]["message"]
            content = msg.get("content", "")
            total_chars = len(content)
            reasoning_chars = len(msg.get("reasoning_content", ""))
            ttft = time.monotonic() - t0  # non-streaming: TTFB ≈ total
        
        elapsed = time.monotonic() - t0
        tps = total_chars / elapsed if elapsed > 0 else 0
        results[name] = {"time": elapsed, "ttft": ttft or elapsed, "chars": total_chars, 
                         "reasoning": reasoning_chars, "tps": tps}
        print(f"  Time: {elapsed:.1f}s  TTFT: {ttft:.1f}s  Chars: {total_chars}  TPS: {tps:.1f}")
        if reasoning_chars: print(f"  Reasoning: {reasoning_chars} chars")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        results[name] = {"error": str(e)}
        return False

# Run tests
print("="*65)
print("TPS BENCHMARK")
print(f"Prompt: {len(PROMPT)} chars — multi-step architecture analysis")
print(f"Both models can deploy sub-agents via tool calls")
print("="*65)

# 1. GLM 5.2 Think-Search via local proxy (streaming)
test_model("GLM 5.2 Think-Search (local proxy)", GLM_BASE, GLM_AUTH, 
           "glm-5.2-think-search", stream=True, reasoning_effort="high")

# 2. DeepSeek V4 Flash via OpenCode Zen (non-streaming, uses placeholder auth)
test_model("DeepSeek V4 Flash (OpenCode Zen)", ZEN_BASE, ZEN_AUTH,
           "deepseek-v4-flash-free")

# 3. GLM 5.2 directly via OpenCode Zen
test_model("GLM 5.2 (OpenCode Zen)", ZEN_BASE, ZEN_AUTH,
           "glm-5.2")

# Print comparison
print(f"\n{'='*65}")
print(f"{'MODEL':<35} {'Time':<10} {'TTFT':<10} {'Chars':<10} {'TPS':<10}")
print(f"{'-'*65}")
for name, r in sorted(results.items(), key=lambda x: x[1].get("tps", 0), reverse=True):
    if "error" in r:
        print(f"{name:<35} {'ERROR: '+r['error']:<40}")
    else:
        print(f"{name:<35} {r['time']:<8.1f}s {r['ttft']:<8.1f}s {r['chars']:<8d} {r['tps']:<8.1f}")
print(f"{'='*65}")

# Speed comparisons
if all(n in results and "error" not in results[n] for n in ["GLM 5.2 Think-Search (local proxy)", "DeepSeek V4 Flash (OpenCode Zen)"]):
    g = results["GLM 5.2 Think-Search (local proxy)"]
    d = results["DeepSeek V4 Flash (OpenCode Zen)"]
    print(f"\nGLM vs DeepSeek TPS ratio: {g['tps']/d['tps']:.2f}x")
    print(f"GLM vs DeepSeek speed ratio: {d['time']/g['time']:.2f}x (DeepSeek {d['time']/g['time']:.1f}x faster)")
