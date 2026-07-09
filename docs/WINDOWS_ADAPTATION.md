# GLM2API Windows Adaptation Guide

Run GLM2API natively on Windows 10/11 with Python 3.14+. No WSL, no Docker, no emulation layer.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Python Version](#python-version)
- [Install Dependencies](#install-dependencies)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [Testing](#testing)
- [Known Issues & Workarounds](#known-issues--workarounds)
  - [uvloop (Not Available on Windows)](#1-uvloop-not-available-on-windows)
  - [Signal Handlers](#2-signal-handlers)
  - [aiohost Event Loop Policy](#3-aiohttp-event-loop-policy)
  - [Unicode on Windows Console](#4-unicode-on-windows-console)
  - [curl_cffi Build Tools](#5-curl_cffi-build-tools)
  - [Port Error Codes](#6-port-error-codes)
- [Dependency Reference](#dependency-reference)
- [Troubleshooting](#troubleshooting)

---

## Requirements

| Item | Requirement |
|------|-------------|
| **OS** | Windows 10 (build 19041+) or Windows 11 |
| **Python** | 3.14 or later (install from [python.org](https://www.python.org/downloads/)) |
| **Architecture** | x64 (recommended) or ARM64 via x64 emulation |
| **Visual C++ Redistributable** | [VC++ Redistributable 14.40+](https://aka.ms/vs/17/release/vc_redist.x64.exe) (required by `curl_cffi` wheels) |
| **Disk space** | ~200 MB |
| **RAM** | 256 MB minimum, 1 GB recommended |

### Optional Build Tools

If `pip install curl-cffi --only-binary` fails (very rare), install Visual C++ Build Tools:

- Download from: <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
- During setup, select **Desktop development with C++**
- Or install the smaller [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) standalone

Most users will **not** need this -- prebuilt wheels exist for Windows x64.

---

## Installation

### 1. Install Python 3.14+

1. Go to <https://www.python.org/downloads/>
2. Download **Python 3.14.x** for Windows (64-bit)
3. Run the installer
4. **IMPORTANT**: Check **"Add Python to PATH"** at the bottom of the installer
5. Click **Install Now**

Verify:

```powershell
python --version
# → Python 3.14.x
```

### 2. Enable Unicode support (strongly recommended)

Before any Python commands, set this environment variable so JSON/Chinese text displays correctly in the console:

```powershell
# PowerShell (run once per session, or add to profile)
$env:PYTHONUTF8 = "1"
```

Or set it permanently:

```powershell
# As Administrator
[System.Environment]::SetEnvironmentVariable("PYTHONUTF8", "1", "User")
```

Then **restart PowerShell** for the change to take effect.

### 3. Create a virtual environment

```powershell
cd C:\path\to\glm2api
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install Visual C++ Redistributable

Download and install from:
<https://aka.ms/vs/17/release/vc_redist.x64.exe>

This is required for `curl_cffi`'s compiled C extension (libcurl). Python 3.14 ships with its own CRT, but system-wide VC++ Redist is still needed for native extension wheels.

---

## Install Dependencies

### One-command install

```powershell
# Activate venv first
.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install with pre-built binaries (recommended)
pip install --only-binary curl-cffi httpx[http2,socks] aiohttp orjson

# If --only-binary fails, try without it:
pip install curl-cffi httpx[http2,socks] aiohttp orjson
```

### What each package does

| Package | Windows Status | Notes |
|---------|---------------|-------|
| `curl-cffi` | Works -- native wheel for x64 | Chrome TLS fingerprinting. Requires VC++ Redist. |
| `httpx[http2,socks]` | Works natively | HTTP/2 + SOCKS5 proxy support. Pure Python + h2. |
| `aiohttp` | Works natively | Async HTTP server. C extension compiles on MSVC. |
| `orjson` | Works natively | Fast JSON. Prebuilt wheel for Windows x64. |
| `uvloop` | **SKIP -- not on Windows** | See [uvloop workaround](#1-uvloop-not-available-on-windows). |

Do **not** install `uvloop` -- it is a Linux-only library (uses `epoll` / `libuv`).

### Windows event loop policy note

If you use `aiohttp` (the `--async` mode), Python's default `ProactorEventLoop` on Windows sometimes has issues with subprocesses and signal handling. The application already configures `WindowsSelectorEventLoopPolicy` via the wrapper suggested in this guide, but you can also set it globally:

```powershell
set PYTHONASYNCIOEVENTLOOP=SelectorEventLoop
```

(Not usually needed -- the app handles this internally.)

---

## Configuration

### .env file

Same as Linux, no changes needed. Example:

```ini
HOST=127.0.0.1
PORT=8000
API_PREFIX=/v1
LOG_LEVEL=INFO
GLM_TOKEN_FILE=token.txt
GLM_USE_GUEST_REFRESH_TOKEN=true
GLM_MAX_CONCURRENCY=10
```

The codebase uses `pathlib.Path` for all file paths (not `os.path`), so Windows paths like `token.txt` or `log/glm2api_debug.log` work correctly.

### Token file (`token.txt`)

Place it in the project root directory alongside `.env`. Same format as Linux:

```
eyJhbGciOiJSUzI1NiIs...
eyJhbGciOiJSUzI1NiIs...
```

### Log directory

The `log/` directory is created automatically on first run -- no manual setup needed.

---

## Running the Server

### Basic startup

```powershell
cd C:\path\to\glm2api
.venv\Scripts\Activate.ps1

# Sync server (default, uses threading)
python -m glm2api

# Async server (uses aiohttp)
python -m glm2api --async
```

### What to expect

On first run, you should see output similar to:

```
> INFO  │ glm2api.app  │ 初始化应用 并发=10 账号数=10 暴露模型=36
> INFO  │ glm2api.app  │ 启动服务 host=127.0.0.1 port=8000 prefix=/v1 accounts=10 ...
```

If you see an error like `ModuleNotFoundError: No module named 'uvloop'`, see the [uvloop workaround](#1-uvloop-not-available-on-windows) below.

### Running in the background (PowerShell)

```powershell
# Start server in background (process will survive shell close)
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m glm2api" -WorkingDirectory "C:\path\to\glm2api"

# Or use a job
$job = Start-Job -ScriptBlock {
    Set-Location "C:\path\to\glm2api"
    .\.venv\Scripts\Activate.ps1
    python -m glm2api
}
```

### Running as a Windows Service

For production use, install via [NSSM (Non-Sucking Service Manager)](https://nssm.cc/):

```powershell
# Download nssm.exe and place it in PATH
nssm install glm2api "C:\path\to\glm2api\.venv\Scripts\python.exe" "-m glm2api"
nssm set glm2api AppDirectory "C:\path\to\glm2api"
nssm set glm2api AppEnvironmentExtra "PYTHONUTF8=1"
nssm start glm2api
```

---

## Testing

### Health check

Open a new PowerShell window:

```powershell
curl.exe http://127.0.0.1:8000/health
# → {"status":"ok","timestamp":..., ...}
```

### Models list

```powershell
curl.exe http://127.0.0.1:8000/v1/models
# → {"object":"list","data":[{"id":"cogView-4-250304",...}]}
```

### Chat completion

```powershell
curl.exe -X POST http://127.0.0.1:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"glm-4-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'
```

### Install and run tests

```powershell
cd C:\path\to\glm2api
.venv\Scripts\Activate.ps1
pip install pytest
python -m pytest tests/ -v
```

(Some tests may require upstream connectivity to chatglm.cn.)

---

## Known Issues & Workarounds

### 1. uvloop (Not Available on Windows)

**Severity**: BLOCKER -- the server will crash on startup without a fix.

**Symptom**:

```
ModuleNotFoundError: No module named 'uvloop'
```

**Root cause**: `src/glm2api/__main__.py` imports `uvloop` unconditionally at the top of the file (line 4). `uvloop` is a Linux-only library that uses `epoll`/`libuv` -- it does not exist on Windows and cannot be installed.

**Fix**: Edit `src/glm2api/__main__.py` to make the uvloop import conditional.

Change `__main__.py` line 1-6:

```python
from __future__ import annotations

import traceback

from .app import StartupError, create_application, create_async_application
from .config import ConfigError
from .logging_utils import get_logger, setup_logging
```

And also handle `app.py` lines 62-63 where `uvloop.install()` is called in `run()`:

```python
def run(self) -> None:
    if self.async_server:
        import asyncio
        import uvloop
        uvloop.install()
        asyncio.run(self._run_async())
    else:
        self._run_sync()
```

**Recommended patch**:

**File: `src/glm2api/__main__.py`**

Replace the unconditional import:

```python
import uvloop
```

With a conditional import:

```python
try:
    import uvloop
except ImportError:
    uvloop = None  # Windows: uvloop not available
```

**File: `src/glm2api/app.py`** (lines 61-64)

Replace:

```python
    def run(self) -> None:
        if self.async_server:
            import asyncio
            import uvloop
            uvloop.install()
            asyncio.run(self._run_async())
```

With:

```python
    def run(self) -> None:
        if self.async_server:
            import asyncio
            try:
                import uvloop
                uvloop.install()
            except ImportError:
                # Windows: uvloop not available, use default asyncio
                import sys
                if sys.platform == "win32":
                    asyncio.set_event_loop_policy(
                        asyncio.WindowsSelectorEventLoopPolicy()
                    )
            asyncio.run(self._run_async())
```

**Note on performance**: Without `uvloop`, the `asyncio` event loop runs on the standard `SelectorEventLoop` (Windows) or `ProactorEventLoop` (Windows default with Python 3.8+). For a proxy server -- which is I/O-bound with moderate concurrency -- this is perfectly adequate. The `uvloop` removal will not be a bottleneck.

### 2. Signal Handlers

**Severity**: Minor -- server may fail to handle graceful shutdown.

**Symptom**:

```
ValueError: signal only works in main thread of the main interpreter
```
or
```
NotImplementedError: SIGTERM not supported on Windows
```

**Root cause**: On Windows:
- `signal.SIGTERM` is not supported -- `signal.signal()` raises `ValueError` or `NotImplementedError`
- Some signal operations may raise `OSError` instead of `ValueError`

The code in `app.py` already has partial protection:

```python
def _install_signal_handlers(self) -> None:
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, self._handle_signal)
        except (ValueError, AttributeError):
            continue
```

But `server.py` (the sync server) has **no** protection:

```python
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
```

**Recommended patch for `server.py`** (lines 46-51):

Replace:

```python
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
```

With:

```python
        try:
            signal.signal(signal.SIGTERM, _signal_handler)
        except (ValueError, NotImplementedError, OSError):
            pass  # Windows: SIGTERM not supported
        try:
            signal.signal(signal.SIGINT, _signal_handler)
        except (ValueError, NotImplementedError, OSError):
            pass
```

Additionally, update `app.py` `_install_signal_handlers` to catch `OSError` and `NotImplementedError`:

```python
def _install_signal_handlers(self) -> None:
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, self._handle_signal)
        except (ValueError, AttributeError, NotImplementedError, OSError):
            continue
```

### 3. aiohttp Event Loop Policy

**Severity**: Minor -- only affects `--async` mode.

**Symptom**:

```
RuntimeError: Event loop is closed
```
or subprocess-related errors when using `--async` mode.

**Root cause**: Windows uses `ProactorEventLoop` by default (Python 3.8+). `aiohttp` works with both `ProactorEventLoop` and `SelectorEventLoop`, but the `SelectorEventLoop` is more compatible with certain subprocess and signal operations.

**Fix**: The recommended Windows uvloop patch (see section 1) already sets `WindowsSelectorEventLoopPolicy()` when uvloop is not available. If you're not using the uvloop patch, set it manually before startup:

```powershell
set PYTHONASYNCIOEVENTLOOP=SelectorEventLoop
python -m glm2api --async
```

Or in code, add at the entry point:

```python
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

### 4. Unicode on Windows Console

**Severity**: Cosmetic -- log output will have garbled characters.

**Symptom**: Logs show `?` or boxes instead of Chinese characters and special symbols.

**Root cause**: Windows Console uses legacy code pages (e.g., CP936 for Chinese, CP437 for English) by default. Python's `print()` and logging output gets encoded/corrupted.

**Codebase already handles this**: `src/glm2api/logging_utils.py` detects Windows (`sys.platform.startswith("win")`) and:

1. Uses ASCII-safe log icons (`>`, `!`, `x`, `X`, `*`) instead of Unicode symbols (`●`, `▲`, `■`, `◈`, `◆`).
2. Reconfigures stdout/stderr with UTF-8 encoding:

```python
if _IS_WINDOWS:
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
```

**User-side mitigation**: Set the `PYTHONUTF8` environment variable to `1` (see [installation step 2](#2-enable-unicode-support-strongly-recommended)). This tells Python to use UTF-8 for all console I/O, which also ensures Chinese characters in API responses are logged correctly.

### 5. curl_cffi Build Tools

**Severity**: Medium -- may block installation.

**Symptom**:

```
error: Microsoft Visual C++ 14.0 or greater is required.
```

or:

```
ERROR: Failed building wheel for curl-cffi
```

**Root cause**: `curl_cffi` ships prebuilt wheels for most platforms, including Windows x64 for Python 3.14. If `pip` cannot find a matching wheel (e.g., ARM64 Windows, or an older pip version), it falls back to building from source, which requires the full Visual C++ Build Tools.

**Fix precedence**:

1. **Try prebuilt wheels first** (works >99% of cases):

```powershell
pip install --only-binary curl-cffi curl-cffi
```

2. **If that fails**, upgrade pip and retry:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install --only-binary curl-cffi curl-cffi
```

3. **If still failing**, install Visual C++ Build Tools:

   - Download from: <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
   - Run the installer
   - Select **Desktop development with C++**
   - Under **Individual components**, ensure **Windows 10/11 SDK** is selected
   - Install (may take 5-15 minutes)
   - Restart PowerShell and retry:

```powershell
pip install curl-cffi
```

4. **Last resort**: If you are on a restricted machine without admin rights and cannot install build tools, install `httpx` only (without `curl-cffi`). The application will fall back to `httpx` for all HTTP requests (documented in `http_client.py`):

```powershell
pip install httpx[http2,socks] aiohttp orjson
```

Then run the server. TLS fingerprinting will be less effective (may trigger WAF blocking), but basic chat functionality will work.

### 6. Port Error Codes

**Severity**: Informational -- already handled.

The codebase in `app.py` `_wrap_server_error` already maps Windows socket error codes:

```python
if exc.errno in {errno.EADDRINUSE, 10048}:  # 10048 = WSAEADDRINUSE
    return StartupError(f"端口已被占用: {self.config.host}:{self.config.port}")
if exc.errno in {errno.EACCES, 10013}:       # 10013 = WSAEACCES
    return StartupError(f"没有权限监听地址: {self.config.host}:{self.config.port}")
if exc.errno in {errno.EADDRNOTAVAIL, 10049}: # 10049 = WSAEADDRNOTAVAIL
    return StartupError(f"监听地址不可用: {self.config.host}")
```

No changes needed -- Windows socket error codes are already handled.

### 7. Threading

**Severity**: None -- works identically.

The application uses `threading.Thread`, `queue.Queue`, `threading.Lock`, and `ThreadingHTTPServer` -- all of which work identically on Windows.

### 8. SOCKS5 Proxies

**Severity**: None -- same configuration.

SOCKS5 proxy configuration via `.env`:

```ini
GLM_PROXY_URL=socks5://127.0.0.1:1080
```

Or with authentication:

```ini
GLM_PROXY_URL=socks5://user:pass@127.0.0.1:1080
```

`httpx[socks]` uses the `socksio` package which works natively on Windows. No changes needed.

---

## Dependency Reference

| Dependency | Python Package | Windows Support | Notes |
|-----------|---------------|-----------------|-------|
| Chrome TLS fingerprint | `curl-cffi` | Native wheel (x64) | See section 5 for build issues |
| HTTP/2 client | `httpx[http2,socks]` | Native | Pure Python + h2 |
| Async HTTP server | `aiohttp` | Native (C extension) | See section 3 for event loop |
| Fast JSON | `orjson` | Native wheel (x64) | No special steps |
| Async event loop | `uvloop` | **NOT AVAILABLE** | Skip install, see section 1 |
| SOCKS5 support | `socksio` (via httpx) | Native | Pure Python |

### pyproject.toml dependency changes (recommended)

The current `pyproject.toml` has `uvloop` as a hard dependency:

```toml
dependencies = [
    "httpx[http2,socks]>=0.28.1",
    "orjson>=3.10.0",
    "aiohttp>=3.11.0",
    "uvloop>=0.21.0",
]
```

Consider making `uvloop` optional to allow `pip install` to succeed on Windows:

```toml
dependencies = [
    "httpx[http2,socks]>=0.28.1",
    "orjson>=3.10.0",
    "aiohttp>=3.11.0",
]
[project.optional-dependencies]
uvloop = ["uvloop>=0.21.0"]
```

This lets users on Linux install with `pip install glm2api[uvloop]` for optimal performance, while Windows users run `pip install glm2api` without issues.

---

## Troubleshooting

### Server won't start

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `No module named 'uvloop'` | uvloop imported on Windows | Apply patch from section 1 |
| `No module named 'glm2api'` | Package not installed | `pip install -e .` or check `PYTHONPATH` |
| `Port 8000 already in use` | Another process on the port | `netstat -ano \| findstr :8000`, then `taskkill /PID <pid>` |
| `ValueError: signal only works in main thread` | Signal in wrong thread | Happens with some IDE debuggers; run from plain PowerShell |
| `ImportError: DLL load failed` | VC++ Redist missing | Install VC++ Redist 14.40+ from section 1 |
| `OSError: [WinError 10049]` | Invalid bind address | Check `HOST` in `.env` -- use `127.0.0.1` not `0.0.0.0` on some configurations |
| `OSError: [WinError 10013]` | Permission denied (port) | Port < 1024 requires admin; use port >= 1024 |

### Upstream connectivity

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `502 Upstream errors` | chatglm.cn blocked (China CDN) | Enable proxy: set `GLM_PROXY_URL` in `.env` |
| All accounts fail | WAF blocking | curl_cffi should bypass; try proxy or reduce `GLM_MAX_CONCURRENCY` |
| `connect ETIMEDOUT` | Network proxy issue | Verify proxy is running: `curl.exe --proxy socks5://127.0.0.1:1080 https://chatglm.cn/` |

### Windows Firewall

If clients on other machines cannot connect:

```powershell
# As Administrator
New-NetFirewallRule -DisplayName "glm2api" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

### WSL2 / Docker conflict

If you have WSL2 or Docker Desktop, they may reserve port 8000 for WSL2 NAT. Either:

1. Change `PORT` in `.env` to something else (e.g., `8001`)
2. Or release the WSL2 port: `netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=127.0.0.1`

---

## Summary of Required Code Changes for Windows

These are the minimum changes needed to run on Windows without patching:

| File | Line(s) | Change |
|------|---------|--------|
| `src/glm2api/__main__.py` | 4 | Make `import uvloop` conditional with `try/except ImportError` |
| `src/glm2api/app.py` | 62-63 | Wrap `uvloop.install()` in `try/except ImportError` |
| `src/glm2api/app.py` | 139 | Add `OSError, NotImplementedError` to signal handler except clause |
| `src/glm2api/server.py` | 46-51 | Wrap `signal.signal()` calls in `try/except` |
| `pyproject.toml` | 15 | Remove `uvloop>=0.21.0` from hard dependencies (or move to optional) |

After these changes, `pip install -e .` will succeed on Windows, and the server will start with full functionality.

---

## Quick Start (Windows)

```powershell
# 1. Install Python 3.14 from python.org
# 2. Install VC++ Redist from aka.ms/vs/17/release/vc_redist.x64.exe
# 3. Open PowerShell as current user

cd C:\path\to\glm2api
python -m venv .venv
.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"

# 4. Apply patches from this guide (see section above)
# 5. Install deps
pip install --only-binary curl-cffi httpx[http2,socks] aiohttp orjson

# 6. Configure
copy .env.example .env
# Edit .env with your settings

# 7. Run
python -m glm2api

# 8. Test (in another terminal)
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/v1/models
```

That is it. No WSL, no Docker, no Linux emulation.
