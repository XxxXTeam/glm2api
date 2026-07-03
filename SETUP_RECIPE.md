# glm2api — Linux Setup Recipe

Full production setup for glm2api with ZCode/t3 integration, sub-agent optimizations,
60-account concurrent guest mode for **glm-5.2-think-search**, adapted for **Linux (Arch-based)**.

## Table of Contents
- [What's Here](#whats-here)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running](#running)
- [Systemd Service (Auto-Start)](#systemd-service-auto-start)
- [ZCode Integration](#zcode-integration)
- [Verification](#verification)
- [File Manifest](#file-manifest)
- [Troubleshooting](#troubleshooting)

## What's Here

| File | Purpose |
|------|---------|
| `.env` | Live production config — 60 accounts, guest mode, `sk-local` auth |
| `glm2api.service` | Systemd unit for auto-start on boot |
| `scripts/manage.sh` | CLI helper: `start`, `install-service`, `status`, `logs`, `test` |
| `SETUP_RECIPE.md` | This file |

Test status: **58 passed** (all tests green).

## Quick Start

```bash
# 1. Enter directory (already cloned)
cd /home/uluru/ZCodeProject/glm2api

# 2. Activate venv & verify
source .venv/bin/activate
python -m pytest tests/ -q

# 3. Start proxy (foreground, Ctrl+C to stop)
python -m glm2api

# 4. Test it
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

## Configuration

Current `.env` production settings:

| Setting | Value | Why |
|---------|-------|-----|
| `GLM_USE_GUEST_REFRESH_TOKEN` | `true` | Guest mode — no login needed |
| `GLM_MAX_CONCURRENCY` | `60` | 60 concurrent guest accounts |
| `SERVER_API_KEYS` | `sk-local` | Auth key for ZCode |
| `REQUEST_TIMEOUT_SECONDS` | `1800` | 30min — search-think can be slow |
| `GLM_DELETE_CONVERSATION` | `false` | Keep convs for debugging |
| `GLM_GUEST_COOLDOWN_SECONDS` | `3` | Prevents rapid retry loops |
| `GLM_GUEST_MAX_RETRIES` | `20` | Retry guest token 20× |
| `GLM_BUSY_MAX_RETRIES` | `30` | Retry upstream busy 30× |
| `MAX_OUTPUT_TOKENS` | `128000` | Maximum output |
| `CORS_ALLOW_ORIGIN` | `http://127.0.0.1:8000` | Locked for local use |

See `.env` for full reference.

## Running

```bash
# Foreground (Ctrl+C to stop)
python -m glm2api

# Via management script
./scripts/manage.sh start
```

Expected output:
```
初始化应用 并发=60 账号数=60 暴露模型=78
启动服务 host=127.0.0.1 port=8000 prefix=/v1 accounts=60 models=glm-5.2,glm-5.2-think,...
```

## Systemd Service (Auto-Start)

### Install

```bash
sudo ./scripts/manage.sh install-service
```

This copies `glm2api.service` to `/etc/systemd/system/`, enables it, and starts it.

### Manage

```bash
./scripts/manage.sh status    # Check service
./scripts/manage.sh logs      # Follow logs
./scripts/manage.sh test      # Test health/models
```

### Service file details (`glm2api.service`)

- Runs as user `uluru`
- Uses venv Python at `.venv/bin/python`
- Restarts on failure (10s delay)
- Hardened with `NoNewPrivileges=true`, `PrivateTmp=true`

## ZCode Integration

ZCode is already configured for glm2api. The provider `glm-local` is set up in
`~/.zcode/cli/config.json`:

```json
{
  "model": "glm-local/glm-5.2",
  "provider": {
    "glm-local": {
      "name": "GLM Local (Free Guest)",
      "kind": "openai-compatible",
      "enabled": true,
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKeyRequired": false,
        "apiKey": "dummy"
      },
      "models": {
        "glm-5.2": { "name": "GLM-5.2" },
        "glm-5.2-think": { "name": "GLM-5.2 Thinking" },
        "glm-5.2-search": { "name": "GLM-5.2 Search" },
        "glm-5.2-think-search": { "name": "GLM-5.2 Think+Search" }
      }
    }
  }
}
```

To use: start the proxy, then in ZCode, select `glm-local/glm-5.2-think-search`.

## Verification

### Tests
```bash
python -m pytest tests/ -v
# → 58 passed
```

### API endpoints
```bash
# Health
curl http://127.0.0.1:8000/health

# Models list
curl http://127.0.0.1:8000/v1/models

# Chat (requires upstream access)
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-local" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Hello"}]}'

# Streaming
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-local" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Search-think
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-local" \
  -H "Content-Type: application/json" \
  -d '{"model":"glm-5.2-think-search","messages":[{"role":"user","content":"Latest news?"}]}'

# Anthropic adapter (used by ZCode)
curl -X POST http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-local" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"glm-5.2","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}'
```

## File Manifest

| File | What | Tracked? |
|------|------|----------|
| `.env` | Production config (60 accounts, guest mode, sk-local auth) | No (gitignored) |
| `.env.example` | Config template | Yes |
| `glm2api.service` | Systemd unit for auto-start | Yes |
| `scripts/manage.sh` | CLI management | Yes |
| `SETUP_RECIPE.md` | This file | Yes |

### Code changes applied

| File | Change |
|------|--------|
| `.env` | Production settings: 60 accounts, 30min timeout, guest mode, sk-local auth |
| `tests/test_translator.py` | Fix 3 assertions: `content is None` → `content == ""` (ponytail change) |
| `tests/test_translator.py` | Fix streaming reasoning test: tool calls now detected during consume_event (was a tuple unpacking bug) |
| `tests/test_protocol_adapters.py` | Fix `_DummyConfig` missing `glm_use_guest_refresh_token` attribute |
| `tests/test_protocol_adapters.py` | Fix keepalive test: use read loop instead of blocking `response.read()` |
| `src/glm2api/server.py` | Fix SSE connections: `Connection: keep-alive` → `Connection: close` so HTTP clients can detect end-of-stream |
| `src/glm2api/services/translator.py` | Fix tuple unpacking bug: `parse_tool_calls_from_text` returns `(text, tool_calls)`, was swapped |

## Troubleshooting

### Server won't start
| Symptom | Cause | Fix |
|---------|-------|-----|
| Port in use | Another instance | `kill $(lsof -ti:8000)` |
| No module glm2api | Not installed | `source .venv/bin/activate && pip install -e .` |
| ConfigError | Bad .env | Compare with `.env.example` |

### No upstream access
| Symptom | Cause | Fix |
|---------|-------|-----|
| 502 Upstream errors | chatglm.cn blocked | Enable proxy: uncomment `GLM_PROXY_URL` in .env |
| All accounts fail | CDN ban | Use proxy or reduce `GLM_MAX_CONCURRENCY` |
| 429 Rate limited | Token endpoint | Lower `GLM_MAX_CONCURRENCY` to 10-20 |
| Slow startup | Guest fetch retries | Check network to chatglm.cn |

### ZCode issues
| Symptom | Cause | Fix |
|---------|-------|-----|
| Connection refused | Proxy not running | `./scripts/manage.sh start` |
| Model not found | Name mismatch | Check `glm-local` config model ids match `/v1/models` |
| Slow responses | Upstream latency | Use `glm-5-turbo-think-search` instead of `glm-5.2-think-search` |
