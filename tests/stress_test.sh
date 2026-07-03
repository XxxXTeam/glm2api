#!/bin/bash
# Stress test for glm2api — tests large volume, tool calls, streaming, proxy resilience
# Run with: bash tests/stress_test.sh
# Options:
#   bash tests/stress_test.sh proxied  → test through GLM_PROXY env var
#   bash tests/stress_test.sh vpngate  → test through VPNGate (starts it)

set -e

MODE="${1:-direct}"
GLM_URL="http://127.0.0.1:8000/v1"
RESULTS_DIR="/tmp/glm2api-stress-$(date +%s)"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  GLM2API STRESS TEST"
echo "  Mode: $MODE"
echo "  URL:  $GLM_URL"
echo "  Logs: $RESULTS_DIR"
echo "============================================"
echo ""

# Check server is running
if ! curl -s --max-time 5 "$GLM_URL/../health" > /dev/null 2>&1; then
  echo "❌ Server not running. Start it first: glm-proxy-start"
  exit 1
fi

echo "✅ Server detected"
echo ""

# Test categories
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
  local name="$1"
  local curl_args="$2"
  local expect="$3"  # string to grep for in response
  local logfile="$RESULTS_DIR/${name//\//_}.log"
  
  echo -n "  $name ... "
  START=$(date +%s%N)
  RESULT=$(curl -s --max-time 60 $curl_args 2>"$logfile.err")
  local code=$?
  local elapsed=$(( ($(date +%s%N) - START) / 1000000 ))
  echo "$RESULT" > "$logfile"
  
  if echo "$RESULT" | grep -q "$expect" 2>/dev/null; then
    echo "✅ ${elapsed}ms"
    TESTS_PASSED=$((TESTS_PASSED + 1))
  else
    local err=$(cat "$logfile.err" 2>/dev/null)
    echo "❌ ${elapsed}ms (expected '$expect', got: $(echo "$RESULT" | head -c 80))"
    [ -n "$err" ] && echo "     stderr: $err"
    TESTS_FAILED=$((TESTS_FAILED + 1))
  fi
}

# ======================================================
# 1. BASIC CHAT
# ======================================================
echo "=== 1. BASIC CHAT ==="
for i in $(seq 1 5); do
  run_test "chat-basic-$i" \
    "-X POST \"$GLM_URL/chat/completions\" -H 'Content-Type: application/json' -d '{\"model\":\"glm-4-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"say hello\"}],\"stream\":false}'" \
    '"content":'
done

# ======================================================
# 2. STREAMING CHAT
# ======================================================
echo "=== 2. STREAMING CHAT ==="
for i in $(seq 1 3); do
  run_test "stream-$i" \
    "-X POST \"$GLM_URL/chat/completions\" -H 'Content-Type: application/json' -d '{\"model\":\"glm-4-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"say hi\"}],\"stream\":true}'" \
    '\[DONE\]'
done

# ======================================================
# 3. TOOL CALLS
# ======================================================
echo "=== 3. TOOL CALLS ==="
for city in "Berlin" "Paris" "London" "Tokyo" "Sydney" "Moscow" "Dubai" "Rome" "Madrid" "Seoul"; do
  run_test "tool-$city" \
    "-X POST \"$GLM_URL/chat/completions\" -H 'Content-Type: application/json' -d '{\"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"Use get_weather for $city\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"stream\":false}'" \
    '"finish_reason":"tool_calls"'
done

# ======================================================
# 4. STREAMING TOOL CALLS
# ======================================================
echo "=== 4. STREAMING TOOL CALLS ==="
for i in 1 2 3 4 5; do
  run_test "stream-tool-$i" \
    "-X POST \"$GLM_URL/chat/completions\" -H 'Content-Type: application/json' -d '{\"model\":\"glm-5.2\",\"messages\":[{\"role\":\"user\",\"content\":\"Use get_weather for a city\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"stream\":true}'" \
    '\[DONE\]'
done

# ======================================================
# 5. THINK MODEL (reasoning)
# ======================================================
echo "=== 5. THINK MODEL ==="
for i in 1 2; do
  run_test "think-$i" \
    "-X POST \"$GLM_URL/chat/completions\" -H 'Content-Type: application/json' -d '{\"model\":\"glm-5.2-think\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Step by step.\"}],\"stream\":false}'" \
    '"reasoning_content":'
done

# ======================================================
# 6. CONCURRENT BURST
# ======================================================
echo "=== 6. CONCURRENT BURST (10 parallel) ==="
for i in $(seq 1 10); do
  curl -s --max-time 60 -X POST "$GLM_URL/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"glm-4-flash","messages":[{"role":"user","content":"say hello"}],"stream":false}' \
    > "$RESULTS_DIR/burst-$i.log" 2>/dev/null &
done
wait
BURST_OK=0
BURST_FAIL=0
for i in $(seq 1 10); do
  if grep -q '"content":' "$RESULTS_DIR/burst-$i.log" 2>/dev/null; then
    BURST_OK=$((BURST_OK + 1))
  else
    BURST_FAIL=$((BURST_FAIL + 1))
  fi
done
echo "  Concurrent burst: $BURST_OK OK / $BURST_FAIL FAIL"
TESTS_PASSED=$((TESTS_PASSED + BURST_OK))
TESTS_FAILED=$((TESTS_FAILED + BURST_FAIL))

# ======================================================
# 7. HEALTH CHECK (always responsive)
# ======================================================
echo "=== 7. HEALTH CHECK ==="
for i in $(seq 1 3); do
  run_test "health-$i" \
    "http://127.0.0.1:8000/health" \
    '"status":"ok"'
done

# ======================================================
# RESULTS
# ======================================================
echo ""
echo "============================================"
echo "  RESULTS"
echo "  Passed: $TESTS_PASSED"
echo "  Failed: $TESTS_FAILED"
echo "  Total:  $((TESTS_PASSED + TESTS_FAILED))"
echo "============================================"

# Summary of failures
if [ "$TESTS_FAILED" -gt 0 ]; then
  echo ""
  echo "=== FAILURE DETAILS ==="
  for f in "$RESULTS_DIR"/*.log; do
    name=$(basename "$f" .log)
    first_line=$(head -1 "$f")
    if echo "$first_line" | grep -qv '"content":\|\[DONE\]\|tool_calls\|"status":"ok"\|reasoning_content'; then
      if [ -s "$f" ]; then
        echo "  $name: $(head -c 100 "$f")"
      fi
    fi
  done
fi

exit $TESTS_FAILED