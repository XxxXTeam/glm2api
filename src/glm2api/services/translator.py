from __future__ import annotations

import orjson
import httpx
import logging
import re
import time
from bisect import insort
from dataclasses import dataclass, field
from html.parser import HTMLParser
from logging import Logger
import threading
from threading import Lock
from urllib.parse import parse_qs, unquote, urlparse

from ..config import AppConfig
from ..logging_utils import debug_dump
from ..model_variants import model_requests_search, model_requests_thinking, split_model_features
from ..utils.tool_parser import StreamingToolParser, parse_tool_calls_from_text

# Ponytail: pre-compiled patterns — faster, avoids re.compile cache lookups
_THINK_OPEN_RE = re.compile(r'<think>')
_THINK_CLOSE_RE = re.compile(r'</think>')
_THINK_BLOCK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
from ..utils.tool_protocol import (
    SERVER_SIDE_TOOL_NAMES,
    build_tool_call_instructions as _protocol_build_tool_call_instructions,
    filter_tools,
    safe_json_dumps,
    serialize_tool_call_block as _protocol_serialize_tool_call_block,
    serialize_tool_result_block as _protocol_serialize_tool_result_block,
    serialize_tool_call,
    tools_to_prompt as _protocol_tools_to_prompt,
)


ASSISTANT_ID_PATTERN = re.compile(r"^[a-z0-9]{24,}$")

# CherryStudio MCP fetch tool names — passthrough with URL auto-fill from context.
CHERRY_FETCH_TOOL_NAMES = {
    "mcp__CherryFetch__fetchHtml",
    "mcp__CherryFetch__fetchMarkdown",
    "mcp__CherryFetch__fetchTxt",
    "mcp__CherryFetch__fetchJson",
}
URL_PATTERN = re.compile(r"https?://[^\s<>()\"']+")
POWERSHELL_CMDLET_PATTERN = re.compile(r"^[A-Z][A-Za-z]+-[A-Z][A-Za-z]+$")
POWERSHELL_ALIASES = {"cat", "cd", "copy", "del", "dir", "echo", "erase", "ls", "md", "move", "pwd", "rd", "ren", "rm", "sc", "type"}

_SEARCH_RESULT_COUNT = 6  # ponytail: top N is enough for LLM context, fewer = faster parse

# Thread-safe search cache for parallel pre-fetch (ponytail: zero-wait DDG)
_SEARCH_CACHE: dict[str, tuple[float, str]] = {}
_SEARCH_CACHE_LOCK = Lock()
_SEARCH_CACHE_TTL = 180.0  # ponytail: 3min — better conversation reuse than 60s
_SEARCH_CACHE_MAX = 200

# Shared httpx client for connection reuse across searches (HTTP/2, keep-alive)
_HTTPX_CLIENT: httpx.Client | None = None
_HTTPX_CLIENT_LOCK = Lock()

def _json_dumps(obj: object) -> str:
    """Ultra-fast JSON serialization with orjson (C extension)."""
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS).decode("utf-8")

def _json_loads(s: str | bytes) -> object:
    """Ultra-fast JSON parsing with orjson (C extension)."""
    return orjson.loads(s)

def _cache_search_query(query: str) -> None:
    """Pre-fetch search results in a background thread and cache them.
    Uses _run_web_search (which has all imports resolved at call time)."""
    t = threading.Thread(target=_do_cache_search, args=(query,), daemon=True)
    t.start()

def _evict_search_cache_if_needed() -> None:
    """Evict oldest 25% of entries when cache exceeds max."""
    if len(_SEARCH_CACHE) > _SEARCH_CACHE_MAX:
        # Sort by timestamp (kv[1][0]), keep newest entries
        oldest = sorted(_SEARCH_CACHE.items(), key=lambda kv: kv[1][0])[:50]
        for k, _ in oldest:
            del _SEARCH_CACHE[k]

def _do_cache_search(query: str) -> None:
    """Worker: run the search and cache results."""
    try:
        result = _run_web_search(query)
        if result and not result.startswith("Search error") and not result.startswith("No search"):
            with _SEARCH_CACHE_LOCK:
                _SEARCH_CACHE[query.lower().strip()] = (time.monotonic(), result)
                _evict_search_cache_if_needed()
    except Exception:
        pass

def _get_cached_search(query: str) -> str | None:
    """Get cached search results if fresh. Updates access time for LRU eviction."""
    with _SEARCH_CACHE_LOCK:
        entry = _SEARCH_CACHE.get(query.lower().strip())
        if entry and (time.monotonic() - entry[0]) < _SEARCH_CACHE_TTL:
            # ponytail: touch timestamp on access — LRU survives eviction
            _SEARCH_CACHE[query.lower().strip()] = (time.monotonic(), entry[1])
            return entry[1]
    return None

def _wait_for_search_result(query: str, timeout: float = 5.0) -> str | None:
    """Get cached search results, waiting up to `timeout` seconds if still being fetched."""
    # First check cache immediately
    cached = _get_cached_search(query)
    if cached:
        return cached
    
    # If not cached yet, a background fetch may be in progress - wait a bit
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.1)
        cached = _get_cached_search(query)
        if cached:
            return cached
    return None


def _ddg_unwrap_url(href: str) -> str:
    """DuckDuckGo Lite wraps real result URLs as
    //duckduckgo.com/l/?uddg=ENCODED&rut=... — unwrap the ``uddg`` param."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    try:
        qs = parse_qs(urlparse(href).query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    except Exception:
        pass
    return href


# ponytail: fast regex parser for DDG Lite HTML (~10x faster than stdlib HTMLParser)
_DDG_LINK_RE = re.compile(r'<a[^>]*class="result-link"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL)
_DDG_SNIPPET_RE = re.compile(r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>', re.DOTALL)

def _parse_ddg_lite(html: str) -> list[tuple[str, str, str]]:
    """Parse DuckDuckGo Lite HTML — fast regex path, fallback stdlib HTMLParser."""
    try:
        links = _DDG_LINK_RE.findall(html)
        snippets = _DDG_SNIPPET_RE.findall(html)
        results: list[tuple[str, str, str]] = []
        for i, (href, title) in enumerate(links):
            snippet = snippets[i].strip() if i < len(snippets) else ""
            results.append((title.strip(), _ddg_unwrap_url(href), snippet))
        if results:
            return results
    except Exception:
        pass
    # Fallback: stdlib HTMLParser
    parser = _DDGLiteParser()
    parser.feed(html)
    parser.close()
    return parser.results


class _DDGLiteParser(HTMLParser):
    """Stdlib fallback parser for DuckDuckGo Lite (used only if bs4 missing)."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[tuple[str, str, str]] = []
        self._in_link = False
        self._in_snippet = False
        self._title = ""
        self._href = ""
        self._snippet = ""
        self._pending: tuple[str, str] | None = None

    def _commit(self) -> None:
        if self._pending is not None:
            title, url = self._pending
            self.results.append((title, url, " ".join(self._snippet.split())))
            self._pending = None
        self._snippet = ""

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == "a" and ("result-link" in d.get("class", "") or "result-a" in d.get("class", "")):
            self._commit()
            self._in_link = True
            self._href = d.get("href", "")
            self._title = ""
        elif tag == "td" and "result-snippet" in d.get("class", ""):
            self._in_snippet = True
            self._snippet = ""

    def handle_endtag(self, tag):
        if tag == "a" and self._in_link:
            self._in_link = False
            self._pending = (self._title.strip(), _ddg_unwrap_url(self._href))
        elif tag == "td" and self._in_snippet:
            self._in_snippet = False

    def handle_data(self, data):
        if self._in_link:
            self._title += data
        if self._in_snippet:
            self._snippet += data

    def close(self):  # type: ignore[override]
        super().close()
        self._commit()


def _get_httpx_client() -> httpx.Client:
    """Get or create shared httpx client with connection pooling and HTTP/2."""
    global _HTTPX_CLIENT
    if _HTTPX_CLIENT is None:
        with _HTTPX_CLIENT_LOCK:
            if _HTTPX_CLIENT is None:
                _HTTPX_CLIENT = httpx.Client(http2=True, timeout=5.0, follow_redirects=True)
    return _HTTPX_CLIENT


def _run_web_search(query: str) -> str:
    """Search via DuckDuckGo Lite using httpx (connection-pooled, HTTP/2)."""
    try:
        client = _get_httpx_client()
        resp = client.post(
            "https://lite.duckduckgo.com/lite",
            data={"q": query, "kl": "us-en"},
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
        )
        if resp.status_code == 200:
            results = _parse_ddg_lite(resp.text)
            if results:
                lines = [f'Web search results for "{query}":']
                for i, (title, link, snippet) in enumerate(results[:_SEARCH_RESULT_COUNT], 1):
                    lines.append(f"{i}. {title}\n   URL: {link}\n   {snippet}")
                return "\n".join(lines)
    except Exception as exc:
        return f"Search error: {exc}"

    return f"No search results found for: {query}"



def _run_and_cache_search(query: str) -> str:
    """Run web search and atomically cache the result for future reuse."""
    result = _run_web_search(query)
    if result and not result.startswith("Search error") and not result.startswith("No search"):
        with _SEARCH_CACHE_LOCK:
            _SEARCH_CACHE[query.lower().strip()] = (time.monotonic(), result)
            _evict_search_cache_if_needed()
    return result


def _execute_retrieve_tool_calls(tool_calls: list[dict[str, object]]) -> tuple[str, list[dict[str, object]]]:
    """Intercept retrieve/search tool calls, execute via DuckDuckGo concurrently, return content + remaining."""
    if not tool_calls:
        return "", tool_calls

    search_tasks: list[tuple[str, str]] = []  # (query, tool_call_id)
    remaining: list[dict[str, object]] = []

    for tc in tool_calls:
        fn = tc.get("function", {})
        tool_name = str(fn.get("name", "")).strip().lower()
        tool_id = str(tc.get("id", ""))
        if tool_name not in {"retrieve", "search", "web_search"}:
            remaining.append(tc)
            continue
        arguments = fn.get("arguments", "{}")
        try:
            args = _json_loads(arguments) if isinstance(arguments, str) else arguments
        except (orjson.JSONDecodeError, Exception):
            remaining.append(tc)
            continue
        if isinstance(args, dict):
            for val in args.values():
                if isinstance(val, str):
                    search_tasks.append((val.strip(), tool_id))
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            for v in item.values():
                                if isinstance(v, str):
                                    search_tasks.append((v.strip(), tool_id))
                        elif isinstance(item, str):
                            search_tasks.append((item.strip(), tool_id))

    if not search_tasks:
        return "", tool_calls

    # ponytail: deduplicate — same query searched once, not N times
    seen: set[str] = set()
    deduped: list[str] = []
    for q, _ in search_tasks:
        key = q.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(q)

    # ponytail: run searches concurrently, checking cache first
    import concurrent.futures as _cf
    search_results: list[str] = []

    with _cf.ThreadPoolExecutor(max_workers=10) as pool:
        fut_map: dict[_cf.Future[str], str] = {}
        for q in deduped:
            cached = _get_cached_search(q)
            if cached is not None:
                search_results.append(f"Search query: {q}\n\nResults:\n{cached}")
            else:
                # Start background prefetch AND pool task — fastest-wins
                _cache_search_query(q)
                fut = pool.submit(_run_and_cache_search, q)
                fut_map[fut] = q
        for f in _cf.as_completed(fut_map):
            q = fut_map[f]
            try:
                result = f.result()
                search_results.append(f"Search query: {q}\n\nResults:\n{result}")
            except Exception as exc:
                search_results.append(f"Search query: {q}\n\nResults:\nSearch error: {exc}")

    if search_results:
        content = "\n\n---\n\n".join(search_results)
        return content, remaining

    return "", tool_calls



def extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _json_dumps(content)
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text_parts.append(str(item.get("text", "")))
        elif item_type == "image_url":
            url = item.get("image_url", {}).get("url", "")
            text_parts.append(f"[image:{url}]")
        elif item_type == "file":
            url = item.get("file_url", {}).get("url", "")
            text_parts.append(f"[file:{url}]")
    return "\n".join(part for part in text_parts if part)


def extract_first_url(text: str) -> str | None:
    match = URL_PATTERN.search(text)
    if not match:
        return None
    return match.group(0).rstrip(".,;:!?)}+")


def extract_recent_user_url(messages: list[dict[str, object]]) -> str | None:
    for message in reversed(messages):
        if str(message.get("role", "")).strip() != "user":
            continue
        text = extract_text_content(message.get("content"))
        url = extract_first_url(text)
        if url:
            return url
    return None


def sanitize_tool_call_payload(
    tool_name: str,
    arguments: object,
    fallback_url: str | None = None,
) -> dict[str, object] | None:
    parsed_arguments = arguments
    if isinstance(arguments, str):
        try:
            parsed_arguments = _json_loads(arguments)
        except (orjson.JSONDecodeError, Exception):
            return None

    if parsed_arguments is None:
        parsed_arguments = {}
    if not isinstance(parsed_arguments, dict):
        return None

    cleaned = {str(key): value for key, value in parsed_arguments.items()}
    if cleaned == {"param_name": "url"} and fallback_url:
        cleaned = {"url": fallback_url}
    elif cleaned == {"param_name": "url"}:
        cleaned = {}
    if "param_name" in cleaned and "param_value" not in cleaned and len(cleaned) == 1:
        cleaned = {}

    if tool_name == "shell":
        command = cleaned.get("command")
        if isinstance(command, str):
            stripped_command = command.strip()
            if stripped_command.startswith("["):
                try:
                    parsed_command = _json_loads(stripped_command)
                except (orjson.JSONDecodeError, Exception):
                    parsed_command = None
                if isinstance(parsed_command, list):
                    cleaned["command"] = [str(part) for part in parsed_command]
            elif stripped_command.startswith('"'):
                try:
                    parsed_command = _json_loads(f"[{stripped_command}]")
                except (orjson.JSONDecodeError, Exception):
                    parsed_command = None
                if isinstance(parsed_command, list):
                    cleaned["command"] = [str(part) for part in parsed_command]
            else:
                cleaned["command"] = ["powershell.exe", "-Command", stripped_command]
        elif isinstance(command, list) and command:
            command_parts = [str(part) for part in command]
            command_name = command_parts[0].strip()
            lower_name = command_name.lower()
            is_shell_host = lower_name in {"powershell", "powershell.exe", "pwsh", "pwsh.exe", "cmd", "cmd.exe"}
            is_powershell_command = bool(POWERSHELL_CMDLET_PATTERN.fullmatch(command_name)) or lower_name in POWERSHELL_ALIASES
            if is_powershell_command and not is_shell_host:
                cleaned["command"] = ["powershell.exe", "-Command", " ".join(command_parts)]


    # CherryStudio MCP fetch: auto-fill URL from recent user message if missing
    if tool_name in CHERRY_FETCH_TOOL_NAMES:
        url_value = cleaned.get("url")
        if not isinstance(url_value, str) or not url_value.strip():
            if fallback_url:
                cleaned["url"] = fallback_url
            else:
                return None

    return cleaned


def sanitize_tool_calls(
    tool_calls: list[dict[str, object]],
    fallback_url: str | None = None,
) -> list[dict[str, object]]:
    sanitized: list[dict[str, object]] = []
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function", {})
        if not isinstance(function, dict):
            continue
        tool_name = str(function.get("name", "")).strip()
        if not tool_name:
            continue
        original_arguments = function.get("arguments", "{}")
        original_value: object = original_arguments
        if isinstance(original_arguments, str):
            try:
                original_value = _json_loads(original_arguments)
            except (orjson.JSONDecodeError, Exception):
                original_value = original_arguments
        cleaned_arguments = sanitize_tool_call_payload(
            tool_name=tool_name,
            arguments=original_arguments,
            fallback_url=fallback_url,
        )
        if cleaned_arguments is None:
            continue
        repaired = not isinstance(original_value, dict) or safe_json_dumps(cleaned_arguments) != safe_json_dumps(original_value)
        sanitized.append(
            {
                "id": str(tool_call.get("id", "")) or f"call_repaired_{index}",
                "type": "function",
                "index": index,
                "_repaired": repaired,
                "function": {
                    "name": tool_name,
                    "arguments": safe_json_dumps(cleaned_arguments),
                },
            }
        )
    return sanitized


def parse_tool_choice_policy(tool_choice: object, available_tool_names: set[str] | None = None) -> dict[str, object]:
    available = available_tool_names or set()
    if tool_choice is None:
        return {"mode": "auto", "tool_name": None}
    if isinstance(tool_choice, str):
        normalized = tool_choice.strip().lower()
        if normalized in {"auto", "none", "required"}:
            return {"mode": normalized, "tool_name": None}
        return {"mode": "auto", "tool_name": None}
    if not isinstance(tool_choice, dict):
        return {"mode": "auto", "tool_name": None}

    choice_type = str(tool_choice.get("type", "")).strip().lower()
    if choice_type == "function":
        function = tool_choice.get("function", {})
        if isinstance(function, dict):
            tool_name = str(function.get("name", "")).strip()
            if tool_name and (not available or tool_name in available):
                return {"mode": "specific", "tool_name": tool_name}
        return {"mode": "auto", "tool_name": None}

    if choice_type in {"auto", "none", "required"}:
        return {"mode": choice_type, "tool_name": None}
    return {"mode": "auto", "tool_name": None}


build_tool_call_instructions = _protocol_build_tool_call_instructions
serialize_tool_call_block = _protocol_serialize_tool_call_block
serialize_tool_result_block = _protocol_serialize_tool_result_block
tools_to_prompt = _protocol_tools_to_prompt


# ponytail: context management — sliding window + token truncation
def apply_context_strategy(
    messages: list[dict[str, object]],
    strategy: str = "sliding",
    limit: int = 30,
) -> list[dict[str, object]]:
    """Apply sliding-window or token-limit strategy to trim context.
    chars/3 ≈ tokens for mixed CN/EN content."""
    if not messages or limit <= 0:
        return messages
    system_msg = None
    rest = list(messages)
    if rest and rest[0].get("role") in ("system", "developer"):
        system_msg = rest.pop(0)
    if strategy == "sliding" and len(rest) > limit:
        rest = rest[-limit:]
    if strategy == "token":
        total = sum(len(str(m.get("content", ""))) // 3 for m in rest)
        while total > limit and len(rest) > 1:
            removed = rest.pop(0)
            total -= len(str(removed.get("content", ""))) // 3
    result = [system_msg] if system_msg else []
    result.extend(rest)
    return result


def convert_messages(
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
    blocked_tool_names: set[str] | None = None,
    tool_choice: object | None = None,
    server_side_tool_names: set[str] | None = None,
    # ponytail: context strategy params
    context_strategy: str = "sliding",
    context_limit: int = 30,
    # ponytail: client detection for bracket protocol
    client: str = "default",
    # ponytail: pre-extracted URL from caller to avoid double scan
    latest_user_url: str | None = None,
) -> list[dict[str, object]]:
    # ponytail: trim context before processing
    messages = apply_context_strategy(messages, context_strategy, context_limit)
    tools = filter_tools(tools, blocked_tool_names or set())
    available_tool_names = {
        str(tool.get("function", {}).get("name", "")).strip()
        for tool in (tools or [])
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    available_tool_names.discard("")
    server_side_tool_names = server_side_tool_names or SERVER_SIDE_TOOL_NAMES
    tool_choice_policy = parse_tool_choice_policy(tool_choice, available_tool_names)
    processed: list[dict[str, str]] = []
    if latest_user_url is None:
        latest_user_url = extract_recent_user_url(messages)
    valid_tool_call_ids: set[str] = set()
    repaired_tool_call_ids: set[str] = set()
    # ponytail: local content cache to avoid re-iterating list content
    _content_cache: dict[int, str] = {}
    def _extract(content: object) -> str:
        if isinstance(content, str):
            return content
        key = id(content)
        cached = _content_cache.get(key)
        if cached is not None:
            return cached
        result = extract_text_content(content)
        _content_cache[key] = result
        return result
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if role == "user":
            current_text = _extract(content)
            current_url = extract_first_url(current_text)
            if current_url:
                latest_user_url = current_url
        if role == "assistant" and message.get("tool_calls"):
            tool_blocks: list[str] = []
            raw_tool_calls = message.get("tool_calls", []) # pyright: ignore[reportGeneralTypeIssues]
            sanitized_tool_calls = sanitize_tool_calls(
                raw_tool_calls if isinstance(raw_tool_calls, list) else [],
                fallback_url=latest_user_url,
            )
            for tool_call in sanitized_tool_calls:
                function = tool_call.get("function", {})
                tool_name = str(function.get("name", "unknown"))
                if available_tool_names and tool_name not in available_tool_names:
                    continue
                tool_blocks.append(
                    serialize_tool_call(
                        name=tool_name,
                        arguments=function.get("arguments", "{}"),
                        client=client,
                    )
                )
                tool_call_id = str(tool_call.get("id", "")).strip()
                if tool_call_id and not tool_call_id.startswith("call_repaired_"):
                    valid_tool_call_ids.add(tool_call_id)
                    if bool(tool_call.get("_repaired")):
                        repaired_tool_call_ids.add(tool_call_id)
            assistant_text = _extract(content).strip() if content else ""
            block = "\n".join(tool_blocks)
            if not assistant_text and not block:
                continue
            content = f"{assistant_text}\n{block}".strip() if assistant_text and block else (assistant_text or block)
        elif role == "tool":
            tool_call_id = str(message.get("tool_call_id", "")).strip()
            if tool_call_id and valid_tool_call_ids and tool_call_id not in valid_tool_call_ids:
                continue
            if tool_call_id and tool_call_id in repaired_tool_call_ids:
                continue
            role = "user"
            tool_name = str(message.get("name", "")).strip() or "unknown_tool"
            tool_result_text = _extract(content)
            content = serialize_tool_result_block(
                tool_call_id=tool_call_id or message.get("tool_call_id", "unknown"),
                tool_name=tool_name,
                content=tool_result_text,
            )
        elif role == "assistant" and not content:
            continue

        text = _extract(content) if content else ""
        if text:
            processed.append({"role": role, "content": text})

    transcript_parts: list[str] = []

    if tools and tool_choice_policy.get("mode") != "none":
        transcript_parts.append(
            tools_to_prompt(
                tools,
                blocked_tool_names=blocked_tool_names,
                tool_choice_policy=tool_choice_policy,
                server_side_tool_names=server_side_tool_names,
            )
        )
        transcript_parts.append("# CONVERSATION")

    for item in processed:
        title = (
            item["role"]
            .replace("system", "System")
            .replace("assistant", "Assistant")
            .replace("user", "User")
            .replace("developer", "Developer")
        )
        line = f"{title}: {item['content']}".strip()
        if line:
            transcript_parts.append(line)

    prompt = "\n\n".join(transcript_parts).strip()
    return [{"role": "user", "content": [{"type": "text", "text": prompt + "\n\nAssistant: "}]}]


def resolve_upstream_model(requested_model: str, config: AppConfig) -> tuple[str, str]:
    base_model, _ = split_model_features(requested_model)
    upstream_model = config.model_aliases.get(base_model, base_model)
    assistant_id = upstream_model if ASSISTANT_ID_PATTERN.fullmatch(upstream_model) else config.glm_assistant_id
    return upstream_model, assistant_id


def resolve_chat_mode(model: str, reasoning_effort: object, deep_research: object) -> str:
    lower_model = (model or "").lower()
    if deep_research or "deepresearch" in lower_model or "deep-research" in lower_model:
        return "deep_research"
    if reasoning_effort or model_requests_thinking(model) or "think" in lower_model or "zero" in lower_model:
        return "zero"
    return ""


def resolve_networking(model: str, web_search: object) -> bool:
    return bool(web_search) or model_requests_search(model)


@dataclass
class GLMEventAccumulator:
    # Pre-serialized SSE frame prefix for speed
    _SSE_PREFIX = b'data: '
    _SSE_SUFFIX = b'\n\n'

    model: str
    allowed_tool_names: set[str] | None = None
    fallback_tool_url: str | None = None
    debug_enabled: bool = False
    logger: Logger | None = None
    conversation_id: str = ""
    created: int = field(default_factory=lambda: int(time.time()))
    parts_by_logic_id: dict[str, dict[str, object]] = field(default_factory=dict)
    ordered_logic_ids: list[str] = field(default_factory=list)
    last_full_text: str = ""
    last_full_reasoning: str = ""
    _part_text_sent: dict[str, int] = field(default_factory=dict)
    _part_reasoning_sent: dict[str, int] = field(default_factory=dict)
    _known_logic_ids_for_text: set[str] = field(default_factory=set)
    _known_logic_ids_for_reasoning: set[str] = field(default_factory=set)
    tool_parser: StreamingToolParser = field(default_factory=StreamingToolParser)
    emitted_role: bool = False
    _render_cache_dirty: bool = True
    _cached_full_text: str = ""
    _cached_full_reasoning: str = ""
    _cached_part_texts: dict[str, str] = field(default_factory=dict)
    _cached_part_reasonings: dict[str, str] = field(default_factory=dict)
    _server_side_tool_calls: list[dict[str, object]] = field(default_factory=list)
    _server_side_tool_call_ids: set[str] = field(default_factory=set)
    _deferred_visible_text_parts: list[str] = field(default_factory=list)
    _on_search_callback: object | None = None
    _on_tool_callback: object | None = None
    # Track tool calls emitted during streaming (so finalize doesn't re-emit)
    _tool_calls_emitted_count: int = 0
    _emitted_tool_call_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.tool_parser.allowed_tool_names = self.allowed_tool_names

    def _prefetch_search_from_tool_call(self, tool_call: dict[str, object]) -> None:
        """If the tool call is a search/retrieve, start background cache pre-fetch."""
        fn = tool_call.get("function", {})
        tool_name = str(fn.get("name", "")).strip().lower()
        if tool_name not in {"retrieve", "search", "web_search"}:
            return
        arguments = fn.get("arguments", "{}")
        try:
            args = _json_loads(arguments) if isinstance(arguments, str) else arguments
        except Exception:
            return
        if not isinstance(args, dict):
            return
        for val in args.values():
            if isinstance(val, str) and val.strip():
                _cache_search_query(val.strip())
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, str) and v.strip():
                                _cache_search_query(v.strip())
                    elif isinstance(item, str) and item.strip():
                        _cache_search_query(item.strip())

    def consume_event(self, payload: dict[str, object]) -> tuple[list[bytes], str | None]:
        debug_dump(self.logger or logging.getLogger("glm2api.null"), self.debug_enabled, "GLM SSE 解析事件", payload)
        if not self.conversation_id and payload.get("conversation_id"):
            self.conversation_id = str(payload["conversation_id"])

        for part in payload.get("parts", []) if isinstance(payload.get("parts"), list) else []: # pyright: ignore[reportGeneralTypeIssues]
            if isinstance(part, dict) and part.get("logic_id"):
                logic_id = str(part["logic_id"])
                if logic_id not in self.parts_by_logic_id:
                    insort(self.ordered_logic_ids, logic_id)
                self.parts_by_logic_id[logic_id] = part
                self._render_cache_dirty = True
            # Extract server-side native tool_calls from content items
            if isinstance(part, dict) and isinstance(part.get("content"), list):
                for content in part["content"]:
                    if isinstance(content, dict) and content.get("type") == "tool_calls":
                        tool_calls_data = content.get("tool_calls")
                        if isinstance(tool_calls_data, dict):
                            tool_name = str(tool_calls_data.get("name", "")).strip()
                            tool_id = str(tool_calls_data.get("id", "")).strip()
                            arguments = tool_calls_data.get("arguments", "{}")
                            if self.allowed_tool_names is not None and tool_name not in self.allowed_tool_names:
                                continue
                            if tool_name and tool_id and tool_id not in self._server_side_tool_call_ids:
                                self._server_side_tool_call_ids.add(tool_id)
                                self._server_side_tool_calls.append(
                                    {
                                        "id": tool_id,
                                        "type": "function",
                                        "index": len(self._server_side_tool_calls),
                                        "function": {
                                            "name": tool_name,
                                            "arguments": str(arguments) if isinstance(arguments, str) else safe_json_dumps(arguments),
                                        },
                                    }
                                )

        text_delta, reasoning_delta = self._compute_deltas()
        self.last_full_text = self._cached_full_text
        self.last_full_reasoning = self._cached_full_reasoning

        chunks: list[bytes] = []
        if reasoning_delta:
            chunks.append(
                self._chunk_json(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": reasoning_delta},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
            )

        # Ponytail: check reasoning content for tool calls during streaming.
        # The think-search model sometimes emits DSML XML inside reasoning instead of text.
        # Extract early so the client sees tool calls in real-time, not just at finalize.
        _, reasoning_tool_calls = parse_tool_calls_from_text(
            self._cached_full_reasoning.strip(),
            allowed_tool_names=self.allowed_tool_names,
        )
        if reasoning_tool_calls:
            # Deduplicate against already-emitted tool calls
            new_calls = [tc for tc in reasoning_tool_calls if tc.get("id", "") not in self._emitted_tool_call_ids]
            if new_calls:
                if not self.emitted_role:
                    chunks.append(
                        self._chunk_json({
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                        })
                    )
                    self.emitted_role = True
                for tc in new_calls:
                    tc["index"] = len(self.tool_parser.tool_calls)
                    self.tool_parser.tool_calls.append(tc)
                    self._emitted_tool_call_ids.add(tc["id"])
                    chunks.append(
                        self._chunk_json({
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": tc["index"],
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": tc["function"],
                                    }]
                                },
                                "finish_reason": None,
                            }]
                        })
                    )
                # ponytail: pre-fetch search queries in background during streaming
                for tc in new_calls:
                    self._prefetch_search_from_tool_call(tc)

        visible_text_delta = self.tool_parser.consume(text_delta)
        if visible_text_delta:
            if self.allowed_tool_names is not None:
                self._deferred_visible_text_parts.append(visible_text_delta)
            else:
                delta_payload: dict[str, object] = {"content": visible_text_delta}
                if not self.emitted_role:
                    delta_payload = {"role": "assistant", "content": visible_text_delta}
                    self.emitted_role = True
                chunks.append(
                    self._chunk_json(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta_payload,
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                )

        # Emit newly detected tool calls from early invoke extraction
        new_tool_calls = self.tool_parser.tool_calls[self._tool_calls_emitted_count:]
        if new_tool_calls:
            if not self.emitted_role:
                chunks.append(
                    self._chunk_json(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant"},
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                )
                self.emitted_role = True
            for tc in new_tool_calls:
                chunks.append(
                    self._chunk_json(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": tc["index"],
                                                "id": tc["id"],
                                                "type": "function",
                                                "function": tc["function"],
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                )
                self._emitted_tool_call_ids.add(tc["id"])
            self._tool_calls_emitted_count = len(self.tool_parser.tool_calls)
            # ponytail: pre-fetch search queries in background during streaming
            for tc in new_tool_calls:
                self._prefetch_search_from_tool_call(tc)
        debug_dump(self.logger or logging.getLogger("glm2api.null"), self.debug_enabled, "GLM SSE 生成增量块", chunks)
        return chunks, str(payload.get("status")) if payload.get("status") is not None else None

    def finalize(self, status: str | None, last_error: dict[str, object] | None = None) -> list[bytes]:
        tail_text, xml_tool_calls = self.tool_parser.flush()
        xml_tool_calls = sanitize_tool_calls(xml_tool_calls, fallback_url=self.fallback_tool_url)
        if not xml_tool_calls:
            xml_tool_calls = self._extract_reasoning_tool_calls()

        # Merge server-side and XML tool calls, re-indexing.
        # Only include tool calls not already emitted during streaming.
        all_tool_calls: list[dict[str, object]] = list(self._server_side_tool_calls)
        for tc in xml_tool_calls:
            if tc.get("id", "") in self._emitted_tool_call_ids:
                continue  # Already emitted during streaming
            tc_copy = dict(tc)
            tc_copy["index"] = len(all_tool_calls)
            all_tool_calls.append(tc_copy)

        # Ponytail: intercept retrieve/search tool calls with DuckDuckGo fallback
        search_content, remaining_tool_calls = _execute_retrieve_tool_calls(all_tool_calls)
        if search_content:
            if self._on_search_callback:
                self._on_search_callback()
            all_tool_calls = remaining_tool_calls
            if self.allowed_tool_names is not None:
                self._deferred_visible_text_parts.append("\n\n" + search_content)
            else:
                if tail_text:
                    tail_text += "\n\n" + search_content
                else:
                    tail_text = search_content
            if self.logger:
                self.logger.info("已拦截检索工具调用，通过 DuckDuckGo 执行搜索 text_len=%s", len(search_content))

        if self.logger:
            self.logger.info(
                "响应收尾 status=%s text_len=%s reasoning_len=%s tool_calls=%s server_tools=%s",
                status,
                len(self._cached_full_text),
                len(self._cached_full_reasoning),
                len(xml_tool_calls),
                len(self._server_side_tool_calls),
            )

        chunks: list[bytes] = []
        had_tool_calls = bool(all_tool_calls) or bool(self._emitted_tool_call_ids)
        final_text = "".join(self._deferred_visible_text_parts) + tail_text
        self._deferred_visible_text_parts: list[str] = []
        if not final_text and not had_tool_calls and self.allowed_tool_names is not None:
                _, attempted_tool_calls = parse_tool_calls_from_text(
                    self._cached_full_text.strip(),
                    allowed_tool_names=None,
                )
                unavailable_names = sorted(
                    {
                        str(tool_call.get("function", {}).get("name", "")).strip()
                        for tool_call in attempted_tool_calls
                        if isinstance(tool_call.get("function"), dict)
                        and str(tool_call.get("function", {}).get("name", "")).strip()
                        not in self.allowed_tool_names
                    }
                )
                if unavailable_names:
                    allowed_names = ", ".join(sorted(self.allowed_tool_names)) or "(none)"
                    final_text = (
                        "模型尝试调用未声明工具 "
                        + ", ".join(f"`{name}`" for name in unavailable_names)
                        + f"，已阻止。本轮只允许这些工具：{allowed_names}。"
                    )
        if final_text and not had_tool_calls:
                delta_payload: dict[str, object] = {"content": final_text}
                if not self.emitted_role:
                    delta_payload = {"role": "assistant", "content": final_text}
                    self.emitted_role = True
                chunks.append(
                    self._chunk_json(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": delta_payload,
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                )

        if status == "intervene" and last_error and last_error.get("intervene_text"):
            chunks.append(
                self._chunk_json(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "\n\n" + str(last_error["intervene_text"])},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
            )

        if had_tool_calls:
                if not self.emitted_role:
                    chunks.append(
                        self._chunk_json(
                            {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"role": "assistant"},
                                        "finish_reason": None,
                                    }
                                ]
                            }
                        )
                    )
                    self.emitted_role = True
                for tool_call in all_tool_calls:
                    chunks.append(
                        self._chunk_json(
                            {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "tool_calls": [
                                                {
                                                    "index": tool_call["index"],
                                                    "id": tool_call["id"],
                                                    "type": "function",
                                                    "function": tool_call["function"],
                                                }
                                            ]
                                        },
                                        "finish_reason": None,
                                    }
                                ]
                            }
                        )
                    )

        finish_reason = "tool_calls" if had_tool_calls else "stop"
        chunks.append(
            self._chunk_json(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )
        )
        chunks.append(b"data: [DONE]\n\n")
        debug_dump(self.logger or logging.getLogger("glm2api.null"), self.debug_enabled, "GLM SSE finalize 输出", chunks)
        return chunks

    def build_response(self) -> dict[str, object]:
        full_text, full_reasoning = self._render_full_output()
        if not full_text and self.last_full_text:
            full_text = self.last_full_text
        if not full_reasoning and self.last_full_reasoning:
            full_reasoning = self.last_full_reasoning
        clean_content, xml_tool_calls = parse_tool_calls_from_text(
            full_text.strip(),
            allowed_tool_names=self.allowed_tool_names,
        )
        xml_tool_calls = sanitize_tool_calls(xml_tool_calls, fallback_url=self.fallback_tool_url)
        if not xml_tool_calls:
            xml_tool_calls = self._extract_reasoning_tool_calls(full_reasoning)

        # Merge server-side and XML tool calls, re-indexing
        all_tool_calls: list[dict[str, object]] = list(self._server_side_tool_calls)
        for tc in xml_tool_calls:
            tc_copy = dict(tc)
            tc_copy["index"] = len(all_tool_calls)
            all_tool_calls.append(tc_copy)

        # Ponytail: intercept retrieve/search tool calls with DuckDuckGo fallback.
        # Run BEFORE computing final_content so search results flow into `content`.
        search_content, remaining_tool_calls = _execute_retrieve_tool_calls(all_tool_calls)
        if search_content:
            if self._on_search_callback:
                self._on_search_callback()
            all_tool_calls = remaining_tool_calls
            clean_content = (clean_content + "\n\n" + search_content) if clean_content else search_content
            if self.logger:
                self.logger.info("非流式：已拦截检索工具调用，通过 DuckDuckGo 执行搜索 text_len=%s", len(search_content))

        final_content = clean_content.strip()
        # Ponytail: upstream may return finish=stop with zero text parts
        # (e.g. upstream timeout mid-generation, or model refuses the prompt).
        # Return empty string instead of None so clients don't crash.
        if not final_content and not all_tool_calls:
            if self.logger:
                self.logger.warning("上游返回空内容（finish=stop 但无文本），返回空白响应 model=%s", self.model)
        message: dict[str, object] = {
            "role": "assistant",
            "content": "" if all_tool_calls or not final_content else final_content,
            "reasoning_content": full_reasoning or None,
        }
        if all_tool_calls:
            message["tool_calls"] = [
                {"id": item["id"], "type": "function", "function": item["function"]}
                for item in all_tool_calls
            ]
        response = {
            "id": self.conversation_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if all_tool_calls else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        if self.logger:
            self.logger.info(
                "非流式响应构建完成 model=%s text_len=%s reasoning_len=%s tool_calls=%s",
                self.model,
                len(final_content),
                len(full_reasoning),
                len(all_tool_calls),
            )
        debug_dump(self.logger or logging.getLogger("glm2api.null"), self.debug_enabled, "GLM 非流式最终响应", response)
        return response

    def _extract_reasoning_tool_calls(self, reasoning_text: str | None = None) -> list[dict[str, object]]:
        source = (reasoning_text if reasoning_text is not None else self.last_full_reasoning) or self._cached_full_reasoning
        if not source:
            return []
        _, tool_calls = parse_tool_calls_from_text(
            source.strip(),
            allowed_tool_names=self.allowed_tool_names,
        )
        return sanitize_tool_calls(tool_calls, fallback_url=self.fallback_tool_url)

    def _compute_deltas(self) -> tuple[str, str]:
        self._render_full_output()
        text_delta_parts: list[str] = []
        reasoning_delta_parts: list[str] = []

        for logic_id in self.ordered_logic_ids:
            rendered_text = self._cached_part_texts.get(logic_id, "")
            rendered_reasoning = self._cached_part_reasonings.get(logic_id, "")

            if rendered_text:
                prev_len = self._part_text_sent.get(logic_id, 0)
                is_new = logic_id not in self._known_logic_ids_for_text
                if is_new:
                    self._known_logic_ids_for_text.add(logic_id)
                    if text_delta_parts or self._part_text_sent:
                        text_delta_parts.append("\n\n")
                    text_delta_parts.append(rendered_text)
                elif len(rendered_text) > prev_len:
                    text_delta_parts.append(rendered_text[prev_len:])
                self._part_text_sent[logic_id] = len(rendered_text)

            if rendered_reasoning:
                prev_len = self._part_reasoning_sent.get(logic_id, 0)
                is_new = logic_id not in self._known_logic_ids_for_reasoning
                if is_new:
                    self._known_logic_ids_for_reasoning.add(logic_id)
                    if reasoning_delta_parts or self._part_reasoning_sent:
                        reasoning_delta_parts.append("\n\n")
                    reasoning_delta_parts.append(rendered_reasoning)
                elif len(rendered_reasoning) > prev_len:
                    reasoning_delta_parts.append(rendered_reasoning[prev_len:])
                self._part_reasoning_sent[logic_id] = len(rendered_reasoning)

        return "".join(text_delta_parts), "".join(reasoning_delta_parts)

    def _render_full_output(self) -> tuple[str, str]:
        if not self._render_cache_dirty:
            return self._cached_full_text, self._cached_full_reasoning

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        self._cached_part_texts.clear()
        self._cached_part_reasonings.clear()
        for logic_id in self.ordered_logic_ids:
            part = self.parts_by_logic_id.get(logic_id)
            if not isinstance(part, dict):
                continue
            content_items = part.get("content", [])
            if not isinstance(content_items, list):
                continue

            part_text: list[str] = []
            part_reasoning: list[str] = []
            for content in content_items:
                if not isinstance(content, dict):
                    continue
                item_type = content.get("type")
                if item_type == "text":
                    part_text.append(str(content.get("text", "")))
                elif item_type == "think":
                    part_reasoning.append(str(content.get("think", "")))
                elif item_type == "code":
                    part_text.append(f"```python\n{content.get('code', '')}\n```")
                elif item_type == "execution_output":
                    part_text.append(str(content.get("content", "")))
                elif item_type == "image":
                    images = content.get("image", [])
                    if isinstance(images, list):
                        for image in images:
                            if isinstance(image, dict) and image.get("image_url"):
                                part_text.append(f"![image]({image['image_url']})")

            rendered_text = "\n".join(filter(None, part_text)).strip()
            rendered_reasoning = "\n".join(filter(None, part_reasoning)).strip()
            if rendered_text:
                text_parts.append(rendered_text)
                self._cached_part_texts[logic_id] = rendered_text
            if rendered_reasoning:
                reasoning_parts.append(rendered_reasoning)
                self._cached_part_reasonings[logic_id] = rendered_reasoning

        self._cached_full_text = "\n\n".join(text_parts)
        self._cached_full_reasoning = "\n\n".join(reasoning_parts)
        self._render_cache_dirty = False
        return self._cached_full_text, self._cached_full_reasoning

    # Pre-built SSE frame prefix/suffix as bytes (Ponytail: avoid repeat encode())
    _SSE_PREFIX = b"data: "
    _SSE_SUFFIX = b"\n\n"

    def _chunk_json(self, patch: dict[str, object]) -> bytes:
        """Build SSE frame as bytes — orjson for 10x faster serialization."""
        payload = {
            "id": self.conversation_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
        }
        payload.update(patch)
        try:
            raw = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
        except Exception:
            raw = safe_json_dumps(payload).encode("utf-8")
        return self._SSE_PREFIX + raw + self._SSE_SUFFIX

