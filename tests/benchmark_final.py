"""Final TPS benchmark: GLM 5.2 Search Think (local) vs Nemotron 3 Ultra Free (OpenCode Zen).
Both models told: deploy sub-agents via tool calls for sub-problems.
cavekit: spec-first. ponytail: one measurement. superpowers: sub-agents for both.
"""
from __future__ import annotations
import sys, os, time, json, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import httpx

DS_KEY = "sk-PLgvMQLY6taJV1g3Hw13nhPcH43wzIR3UNG3PkLwTxUssw8S8onQvtqb9yKW2R9"

PROMPT = """You can deploy sub-agents via tool calls. Use a sub-agent for each:
1. Inventory sync across 12 DCs (<100ms)
2. Fulfillment routing optimization
3. Fraud detection (<200ms latency)
4. 99.995% uptime architecture
5. Multi-cloud cost optimization

Then synthesize: a global order system handling 50k orders/min across 12 DCs.
Be specific about technologies, trade-offs, and costs."""

def test_model(name, url, headers, model, max_tokens=2048, timeout=180):
    print(f"\n--- {name} ---")
    t0 = time.monotonic()
    ttft = None; out = 0; reas = 0; chunks = 0; tcs = 0
    try:
        with httpx.Client(timeout=timeout) as c:
            with c.stream("POST", f"{url}/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": PROMPT}],
                      "stream": True, "max_tokens": max_tokens},
                headers=headers) as r:
                for line in r.iter_lines():
                    if not line or line == "data: [DONE]": continue
                    if line.startswith("data: "):
                        chunks += 1
                        try:
                            d = json.loads(line[6:])
                            delta = d.get("choices", [{}])[0].get("delta", {})
                            if ttft is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
                                ttft = time.monotonic() - t0
                            if delta.get("content"): out += len(delta["content"])
                            if delta.get("reasoning_content"): reas += len(delta["reasoning_content"])
                            if delta.get("tool_calls"): tcs += 1
                        except: pass
        e = time.monotonic() - t0
        tps = out / e if e > 0 else 0
        gen_tps = out / max(e - (ttft or 0), 0.1) / 4  # chars→tokens rough, during generation only
        print(f"  Total: {e:.1f}s  TTFT: {ttft or 0:.1f}s  Out: {out}c  Reason: {reas}c")
        print(f"  TPS: {tps:.1f} c/s  GenTPS: {gen_tps:.0f} tok/s  Chunks: {chunks}  ToolCallChunks: {tcs}")
        return {"name": name, "time": e, "ttft": ttft or 0, "out": out, "reas": reas, "tps": tps, "gen_tps": gen_tps, "chunks": chunks, "tcs": tcs}
    except Exception as ex:
        print(f"  FAILED: {ex}")
        return {"name": name, "error": str(ex)}

results = []
print("=" * 70)
print("TPS BENCHMARK — Sub-Agent Architecture Design Problem")
print("Both models: can deploy sub-agents via tool calls")
print(f"Prompt: {len(PROMPT)} chars across 5 sub-problems")
print("=" * 70)

# 1. Nemotron 3 Ultra Free via OpenCode Zen (US-based, low latency)
results.append(test_model(
    "Nemotron 3 Ultra Free (OpenCode Zen)",
    "https://opencode.ai/zen/v1",
    {"Authorization": f"Bearer {DS_KEY}"},
    "nemotron-3-ultra-free", max_tokens=2048, timeout=60
))

# 2. GLM 5.2 Think-Search via local proxy (China-based, high latency)
results.append(test_model(
    "GLM 5.2 Think-Search (local proxy)",
    "http://127.0.0.1:8000/v1",
    {"Authorization": "Bearer sk-local"},
    "glm-5.2-think-search", max_tokens=2048, timeout=180
))

# Comparison table
print(f"\n{'='*70}")
print(f"{'Model':40s} {'Time':>8s} {'TTFT':>8s} {'Output':>8s} {'TPS':>8s} {'GenTPS':>8s}")
print(f"{'-'*70}")
for r in results:
    if "error" in r:
        print(f"{r['name']:40s} {'ERROR: '+r['error']:>30s}")
    else:
        print(f"{r['name']:40s} {r['time']:>6.1f}s {r['ttft']:>6.1f}s {r['out']:>6d}c {r['tps']:>6.1f} {r['gen_tps']:>6.0f}t/s")
print(f"{'='*70}")

# Interpretation
if len(results) >= 2 and "error" not in results[0] and "error" not in results[1]:
    n, g = results[0], results[1]
    print(f"\nSpeed comparison:")
    print(f"  Nemotron is {g['time']/n['time']:.1f}x faster in total time")
    print(f"  Nemotron is {g['ttft']/n['ttft']:.1f}x faster in TTFT" if n['ttft'] > 0 else "  TTFT comparison N/A")
    print(f"  GLM produces {g['tps']/n['tps']:.1f}x the TPS of Nemotron" if n['tps'] > 0 else "")
    if g.get('reas', 0) > 0:
        print(f"  GLM has reasoning: {g['reas']} chars of deep reasoning content")
    if g.get('tcs', 0) > 0:
        print(f"  GLM used {g['tcs']} tool call chunks (sub-agent delegation)")

sys.exit(0 if all("error" not in r for r in results) else 1)
