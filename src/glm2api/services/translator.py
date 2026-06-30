from __future__ import annotations

import json
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
from ..utils.tool_protocol import (
    BLOCKED_NATIVE_TOOL_NAMES,
    CANONICAL_TOOL_CALL_EXAMPLE,
    SERVER_SIDE_TOOL_NAMES,
    build_tool_call_instructions as _protocol_build_tool_call_instructions,
    filter_tools,
    normalize_tool_name,
    safe_json_dumps,
    serialize_tool_call_block as _protocol_serialize_tool_call_block,
    serialize_tool_result_block as _protocol_serialize_tool_result_block,
    tools_to_prompt as _protocol_tools_to_prompt,
)


ASSISTANT_ID_PATTERN = re.compile(r"^[a-z0-9]{24,}$")
URL_PATTERN = re.compile(r"https?://[^\s<>()\"']+")
POWERSHELL_CMDLET_PATTERN = re.compile(r"^[A-Z][A-Za-z]+-[A-Z][A-Za-z]+$")
POWERSHELL_ALIASES = {"cat", "cd", "copy", "del", "dir", "echo", "erase", "ls", "md", "move", "pwd", "rd", "ren", "rm", "sc", "type"}

# Ponytail: free DuckDuckGo search as fallback for guest mode search
_SEARCH_TOOL_NAMES = {"retrieve", "search", "web_search"}
_SEARCH_RESULT_COUNT = 8

# Thread-safe search cache for parallel pre-fetch (ponytail: zero-wait DDG)
_SEARCH_CACHE: dict[str, tuple[float, str]] = {}
_SEARCH_CACHE_LOCK = Lock()
_SEARCH_CACHE_TTL = 60.0  # seconds to keep cached results

def _cache_search_query(query: str) -> None:
    """Pre-fetch search results in a background thread and cache them.
    Uses _run_web_search (which has all imports resolved at call time)."""
    t = threading.Thread(target=_do_cache_search, args=(query,), daemon=True)
    t.start()

def _do_cache_search(query: str) -> None:
    """Worker: run the search and cache results."""
    try:
        result = _run_web_search(query)
        if result and not result.startswith("Search error") and not result.startswith("No search"):
            with _SEARCH_CACHE_LOCK:
                _SEARCH_CACHE[query.lower().strip()] = (time.time(), result)
    except Exception:
        pass

def _get_cached_search(query: str) -> str | None:
    """Get cached search results if they exist and are fresh."""
    with _SEARCH_CACHE_LOCK:
        entry = _SEARCH_CACHE.get(query.lower().strip())
        if entry and (time.time() - entry[0]) < _SEARCH_CACHE_TTL:
            return entry[1]
    return None

def _evict_stale_cache() -> None:
    """Remove expired cache entries."""
    now = time.time()
    with _SEARCH_CACHE_LOCK:
        expired = [k for k, v in _SEARCH_CACHE.items() if (now - v[0]) >= _SEARCH_CACHE_TTL]
        for k in expired:
            del _SEARCH_CACHE[k]


def _wait_for_search_result(query: str, timeout: float = 5.0) -> str | None:
    """Get cached search results, waiting up to `timeout` seconds if still being fetched."""
    # First check cache immediately
    cached = _get_cached_search(query)
    if cached:
        return cached
    
    # If not cached yet, a background fetch may be in progress - wait a bit
    deadline = time.time() + timeout
    while time.time() < deadline:
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


def _parse_ddg_lite(html: str) -> list[tuple[str, str, str]]:
    """Parse DuckDuckGo Lite HTML into (title, url, snippet) tuples.

    Uses BeautifulSoup with the lxml backend (tolerant of malformed markup).
    Results anchors carry class ``result-link``; snippets live in
    ``td.result-snippet``. Both lists are in document order and line up 1:1.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _parse_ddg_lite_fallback(html)

    soup = BeautifulSoup(html, "lxml")
    links = soup.select("a.result-link") or soup.select("a.result-a")
    snippets = [td.get_text(" ", strip=True) for td in soup.select("td.result-snippet")]
    results: list[tuple[str, str, str]] = []
    for idx, link in enumerate(links):
        title = link.get_text(" ", strip=True)
        if not title:
            continue
        url = _ddg_unwrap_url(link.get("href", ""))
        snippet = snippets[idx] if idx < len(snippets) else ""
        results.append((title, url, snippet))
    return results


def _parse_ddg_lite_fallback(html: str) -> list[tuple[str, str, str]]:
    """Minimal stdlib fallback if BeautifulSoup/lxml are unavailable."""
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


def _run_web_search(query: str) -> str:
    """Search via DuckDuckGo Lite using curl_cffi (Chrome TLS/JA3 fingerprint).

    No API key required. The ``impersonate="chrome"`` flag reproduces Chrome's
    TLS handshake + HTTP/2 fingerprint, which bypasses DDG bot detection that
    plain requests/httpx cannot pass.
    """
    try:
        from curl_cffi import requests as cffi_requests
    except ImportError as exc:
        return f"Search unavailable: curl_cffi not installed ({exc})."

    try:
        resp = cffi_requests.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query, "kl": "us-en"},
            impersonate="chrome",
            timeout=15,
        )
        if resp.status_code != 200:
            return f"Search error: HTTP {resp.status_code}"
        results = _parse_ddg_lite(resp.text)
    except Exception as exc:
        return f"Search error: {exc}"

    if not results:
        return f"No search results found for: {query}"

    lines = [f'Web search results for "{query}":']
    for i, (title, link, snippet) in enumerate(results[:_SEARCH_RESULT_COUNT], 1):
        lines.append(f"{i}. {title}\n   URL: {link}\n   {snippet}")
    return "\n".join(lines)


def _execute_retrieve_tool_calls(tool_calls: list[dict[str, object]]) -> tuple[str, list[dict[str, object]]]:
    """Intercept retrieve/search tool calls, execute via DuckDuckGo, return content + remaining tool calls."""
    if not tool_calls:
        return "", tool_calls

    search_results: list[str] = []
    remaining: list[dict[str, object]] = []

    for tc in tool_calls:
        fn = tc.get("function", {})
        tool_name = str(fn.get("name", "")).strip().lower()
        if tool_name not in _SEARCH_TOOL_NAMES:
            remaining.append(tc)
            continue

        arguments = fn.get("arguments", "{}")
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            remaining.append(tc)
            continue

        # Extract queries from various argument formats
        queries: list[str] = []
        if isinstance(args, dict):
            for val in args.values():
                if isinstance(val, str):
                    queries.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            for v in item.values():
                                if isinstance(v, str):
                                    queries.append(v)
                        elif isinstance(item, str):
                            queries.append(item)

        for q in queries:
            if q.strip():
                result = _run_web_search(q.strip())
                search_results.append(f"Search query: {q}\n\nResults:\n{result}")

    if search_results:
        content = "\n\n---\n\n".join(search_results)
        return content, remaining

    return "", tool_calls



def extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
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
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
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
                    parsed_command = json.loads(stripped_command)
                except json.JSONDecodeError:
                    parsed_command = None
                if isinstance(parsed_command, list):
                    cleaned["command"] = [str(part) for part in parsed_command]
            elif stripped_command.startswith('"'):
                try:
                    parsed_command = json.loads(f"[{stripped_command}]")
                except json.JSONDecodeError:
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
                original_value = json.loads(original_arguments)
            except json.JSONDecodeError:
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


def _legacy_build_tool_call_instructions(
    tool_names: list[str],
    server_side_tool_names: set[str] | None = None,
    tool_choice_policy: dict[str, object] | None = None,
) -> str:
    server_side_tool_names = server_side_tool_names or set()
    xml_tools = [name for name in tool_names if name not in server_side_tool_names]
    server_tools = [name for name in tool_names if name in server_side_tool_names]

    available_xml_names = ", ".join(f"`{name}`" for name in xml_tools) or "`(none)`"
    available_server_names = ", ".join(f"`{name}`" for name in server_tools) or "`(none)`"

    policy = tool_choice_policy or {"mode": "auto", "tool_name": None}
    mode = str(policy.get("mode", "auto"))
    specific_name = str(policy.get("tool_name", "") or "")
    lines = [
        "# TOOL USE PROTOCOL",
        "The following tool schemas are the only executable tool definitions for this turn.",
        "Ignore any tool names that are not listed below, even if they appear in prior context or model memory.",
        "You are connected through an OpenAI-compatible proxy. You do not have hidden browser, web, or URL-opening tools.",
        "Never call native tools such as `open_url`, `web.search`, `web.run`, `browser.open`, `browse`, `open_link`, `search`, or `find`.",
        "Do not output hidden reasoning, chain-of-thought, or labels such as `Thinking:`.",
        "Do not narrate tool selection, failed tool attempts, retries, fallback plans, or tool status banners.",
    ]

    if server_tools:
        lines.extend(
            [
                "",
                f"Server-side native tools (executed by backend automatically): {available_server_names}.",
                "When you need to call a server-side native tool, output a single structured JSON block with type 'tool_calls' in the assistant content.",
                'Format: {"type":"tool_calls","tool_calls":{"id":"call_<random_hex>","name":"TOOL_NAME","arguments":"<JSON_STRING>"}}',
                "The arguments field must be a JSON string (not a raw object). The server will intercept this block, execute the tool, and inject the result back into the stream as a tool message.",
                "Do not wrap server-side tool calls in XML. Do not mix prose and the tool_calls JSON block in the same response.",
            ]
        )

    if xml_tools:
        lines.extend(
            [
                "",
                f"XML-based tools (parsed by this server): {available_xml_names}.",
                "Only these XML-based tools are available. Use their exact names and exact parameter fields from the schemas.",
                "If an XML-based tool is needed, output executable XML only. Do not add prose, apologies, analysis, or progress text in the same assistant answer.",
                "Use the private ml-prefixed canonical format below exactly.",
                CANONICAL_TOOL_CALL_EXAMPLE,
                "The server will parse this XML intermediate language back into standard OpenAI tool_calls.",
                "Parameter rules:",
                "- The root executable block must be <ml_tool_calls> and each call must be a <ml_tool_call> child.",
                "- Each <ml_tool_call> must contain exactly one <ml_tool_name> and one <ml_parameters> block.",
                "- Use the real parameter names as XML tags inside <ml_parameters>; never use a literal <param_name> placeholder tag.",
                "- Encode arguments as nested XML tags inside <ml_parameters>.",
                "- Use repeated <item> tags to represent arrays.",
            ]
        )

    lines.extend(
        [
            "",
            "Rules:",
            "- Do not invent tool names outside the declared list.",
            "- If a URL, browsing, or search action is needed, use only an explicitly listed client tool. If none is listed, explain that no such tool is available. Never use bare tool names `search` or `find` unless they are explicitly listed above.",
            "- If you decide to call a tool, call the selected tool directly; do not say you will try, switch, retry, or use a correct tool.",
            "- Never output tool-call display text such as `⚙ tool_name [...]`; output only the executable XML block.",
            "- After receiving a tool result, answer the user directly from the result and do not repeat the earlier tool-call decision process.",
            "- For XML-based tools, do not emit OpenAI JSON tool_calls arrays, function_call objects, or any non-XML tool syntax.",
            "- For XML-based tools, do not use <tool_calls>, <tool_call>, <tool_name>, <parameters>, <function_call>, <tool_use>, <invoke>, or any legacy wrapper.",
            "- Do not place raw JSON directly inside <ml_parameters>.",
            "- Do not mix normal explanation text with executable tool XML.",
            "- Prefer <![CDATA[...]]> for arbitrary strings.",
            "- Put multiple XML calls inside one <ml_tool_calls> root when you truly need multiple calls in one turn.",
            "- After a <ml_tool_result ...> block, continue from that result and call another tool only when necessary.",
        ]
    )
    if mode == "none":
        lines.extend(
            [
                "Tool choice policy: none.",
                "Do not emit any executable tool markup. Answer with normal text only.",
            ]
        )
    elif mode == "required":
        lines.extend(
            [
                "Tool choice policy: required.",
                "You must call at least one tool before giving a final answer.",
            ]
        )
    elif mode == "specific" and specific_name:
        lines.extend(
            [
                "Tool choice policy: specific function.",
                f"You must call exactly `{specific_name}` before giving a final answer.",
                f"Do not call any tool other than `{specific_name}`.",
            ]
        )
    return "\n".join(lines)


def _legacy_tools_to_prompt(
    tools: list[dict[str, object]],
    blocked_tool_names: set[str] | None = None,
    tool_choice_policy: dict[str, object] | None = None,
    server_side_tool_names: set[str] | None = None,
) -> str:
    tool_names: list[str] = []
    tool_schemas: list[str] = []
    for tool in tools:
        fn = tool.get("function", {})
        name = str(fn.get("name", "unknown")) # type: ignore
        description = str(fn.get("description", "") or "") # type: ignore
        parameters = fn.get("parameters", {}) # type: ignore
        tool_names.append(name)
        tool_schemas.append(
            "\n".join(
                [
                    f"Tool: {name}",
                    f"Description: {description}",
                    f"Parameters: {safe_json_dumps(parameters) if isinstance(parameters, dict) else '{}'}",
                ]
            )
        )

    parts = [
        "# TOOL SCHEMAS",
        "Treat the following schema list as the authoritative tool contract for this request.",
        "",
        "\n\n".join(tool_schemas),
        "",
        build_tool_call_instructions(
            tool_names,
            server_side_tool_names=server_side_tool_names,
            tool_choice_policy=tool_choice_policy,
        ),
    ]
    return "\n".join(part for part in parts if part is not None).strip()


build_tool_call_instructions = _protocol_build_tool_call_instructions
serialize_tool_call_block = _protocol_serialize_tool_call_block
serialize_tool_result_block = _protocol_serialize_tool_result_block
tools_to_prompt = _protocol_tools_to_prompt


def convert_messages(
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
    blocked_tool_names: set[str] | None = None,
    tool_choice: object | None = None,
    server_side_tool_names: set[str] | None = None,
) -> list[dict[str, object]]:
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
    latest_user_url: str | None = extract_recent_user_url(messages)
    valid_tool_call_ids: set[str] = set()
    repaired_tool_call_ids: set[str] = set()
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if role == "user":
            current_text = extract_text_content(content)
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
                    serialize_tool_call_block(
                        name=tool_name,
                        arguments=function.get("arguments", "{}"),
                    )
                )
                tool_call_id = str(tool_call.get("id", "")).strip()
                if tool_call_id and not tool_call_id.startswith("call_repaired_"):
                    valid_tool_call_ids.add(tool_call_id)
                    if bool(tool_call.get("_repaired")):
                        repaired_tool_call_ids.add(tool_call_id)
            assistant_text = extract_text_content(content).strip() if content else ""
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
            tool_result_text = extract_text_content(content)
            content = serialize_tool_result_block(
                tool_call_id=tool_call_id or message.get("tool_call_id", "unknown"),
                tool_name=tool_name,
                content=tool_result_text,
            )
        elif role == "assistant" and not content:
            continue

        text = extract_text_content(content) if content else ""
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
        transcript_parts.append(f"{title}: {item['content']}".strip())

    prompt = "\n\n".join(part for part in transcript_parts if part).strip()
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
    _known_logic_ids_for_text: list[str] = field(default_factory=list)
    _known_logic_ids_for_reasoning: list[str] = field(default_factory=list)
    tool_parser: StreamingToolParser = field(default_factory=StreamingToolParser)
    emitted_role: bool = False
    _render_cache_dirty: bool = True
    _cached_full_text: str = ""
    _cached_full_reasoning: str = ""
    _cached_part_texts: dict[str, str] = field(default_factory=dict)
    _cached_part_reasonings: dict[str, str] = field(default_factory=dict)
    _server_side_tool_calls: list[dict[str, object]] = field(default_factory=list)
    _server_side_tool_call_ids: set[str] = field(default_factory=set)
    _deferred_visible_text: str = ""

    def __post_init__(self) -> None:
        self.tool_parser.allowed_tool_names = self.allowed_tool_names

    def consume_event(self, payload: dict[str, object]) -> tuple[list[str], str | None]:
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

        chunks: list[str] = []
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

        visible_text_delta = self.tool_parser.consume(text_delta)
        if visible_text_delta:
            if self.allowed_tool_names is not None:
                self._deferred_visible_text += visible_text_delta
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
        debug_dump(self.logger or logging.getLogger("glm2api.null"), self.debug_enabled, "GLM SSE 生成增量块", chunks)
        return chunks, str(payload.get("status")) if payload.get("status") is not None else None

    def finalize(self, status: str | None, last_error: dict[str, object] | None = None) -> list[str]:
        tail_text, xml_tool_calls = self.tool_parser.flush()
        xml_tool_calls = sanitize_tool_calls(xml_tool_calls, fallback_url=self.fallback_tool_url)
        if not xml_tool_calls:
            xml_tool_calls = self._extract_reasoning_tool_calls()

        # Merge server-side and XML tool calls, re-indexing
        all_tool_calls: list[dict[str, object]] = list(self._server_side_tool_calls)
        for tc in xml_tool_calls:
            tc_copy = dict(tc)
            tc_copy["index"] = len(all_tool_calls)
            all_tool_calls.append(tc_copy)

        # Ponytail: intercept retrieve/search tool calls with DuckDuckGo fallback
        search_content, remaining_tool_calls = _execute_retrieve_tool_calls(all_tool_calls)
        if search_content:
            all_tool_calls = remaining_tool_calls
            if self.allowed_tool_names is not None:
                self._deferred_visible_text += "\n\n" + search_content
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

        chunks: list[str] = []
        final_text = self._deferred_visible_text + tail_text
        self._deferred_visible_text = ""
        if not final_text and not all_tool_calls and self.allowed_tool_names is not None:
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
        if final_text and not all_tool_calls:
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

        if all_tool_calls:
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

        finish_reason = "tool_calls" if all_tool_calls else "stop"
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
        chunks.append("data: [DONE]\n\n")
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
            all_tool_calls = remaining_tool_calls
            clean_content = (clean_content + "\n\n" + search_content) if clean_content else search_content
            if self.logger:
                self.logger.info("非流式：已拦截检索工具调用，通过 DuckDuckGo 执行搜索 text_len=%s", len(search_content))

        final_content = clean_content.strip()
        message: dict[str, object] = {
            "role": "assistant",
            "content": None if all_tool_calls or not final_content else final_content,
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
                    self._known_logic_ids_for_text.append(logic_id)
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
                    self._known_logic_ids_for_reasoning.append(logic_id)
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

    def _chunk_json(self, patch: dict[str, object]) -> str:
        payload = {
            "id": self.conversation_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
        }
        payload.update(patch)
        return "data: " + safe_json_dumps(payload) + "\n\n"
