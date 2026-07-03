from __future__ import annotations

import base64
import codecs
import gzip
import select
import http.client
import json
import mimetypes
import os as _os_module
import re
import socket
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.generator import _make_boundary # type: ignore
from io import BufferedReader, BytesIO
from logging import Logger
from typing import Callable

from ..config import AppConfig
from ..logging_utils import debug_dump
from .glm_auth import GLMAccessTokenManager, build_sign
STREAM_READ_TIMEOUT = int(_os_module.environ.get('GLM_STREAM_READ_TIMEOUT_SECONDS', '60'))


from .translator import (
    _cache_search_query,
    _wait_for_search_result,
    BLOCKED_NATIVE_TOOL_NAMES,
    GLMEventAccumulator,
    SERVER_SIDE_TOOL_NAMES,
    convert_messages,
    extract_recent_user_url,
    extract_text_content,
    filter_tools,
    resolve_chat_mode,
    resolve_networking,
    resolve_upstream_model,
)


FILE_UPLOAD_URL_SUFFIX = "/backend-api/assistant/file_upload"
FILE_SIZE_LIMIT = 100 * 1024 * 1024
IMAGE_SIZE_TO_ASPECT_RATIO = {
    "1024x1024": "1:1",
    "1024x1536": "2:3",
    "1536x1024": "3:2",
    "1024x1792": "9:16",
    "1792x1024": "16:9",
}


def _get_glm_opener():
    """Compatibility shim — delegates to http_client.do_request.
    
    Existing code calls `_get_glm_opener().open(request, timeout=...)` which returns
    a urllib addinfourl. We intercept that and route through curl_cffi.
    """
    from .http_client import do_request
    from .glm2api_proxy import get_pool

    class _OpenerShim:
        def open(self, request, timeout=None):
            pool = get_pool()
            proxy_url = pool.get_next() if pool._proxies else None
            headers = dict(getattr(request, 'header_items', lambda: [])())
            method = request.get_method() if hasattr(request, 'get_method') else "POST"
            url = request.full_url if hasattr(request, 'full_url') else str(request)
            data = request.data if hasattr(request, 'data') and request.data else None
            try:
                # stream=True for all requests — StreamingResponseWrapper handles
                # both chunked reading (SSE) and full reads (JSON responses)
                result = do_request(method, url, headers, data, proxy_url, timeout or 300, stream=True)
                if proxy_url:
                    pool.report_success(proxy_url, 0)
                return result
            except Exception as exc:
                if proxy_url:
                    pool.report_failure(proxy_url)
                raise urllib.error.URLError(str(exc)) from exc

        def close(self):
            from .http_client import close_all
            close_all()

    return _OpenerShim()



class UpstreamAPIError(RuntimeError):
    def __init__(self, status_code: int, message: str, payload: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


class QueueTimeoutError(RuntimeError):
    pass


@dataclass(slots=True)
class QueueLease:
    ticket: int
    release_callback: Callable[[int], None]
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.release_callback(self.ticket)


class ConcurrentRequestQueue:
    """Per-account semaphore queue. Each account gets 5 concurrent slots (upstream per-IP limit).
    30 accounts × 5 = 150 concurrent max, instead of global FIFO-5 bottleneck."""
    
    PER_ACCOUNT_LIMIT = 5  # upstream rate-limit per IP

    def __init__(self, logger: Logger, wait_timeout: int, max_concurrency: int) -> None:
        self.logger = logger
        self.wait_timeout = wait_timeout
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._sems: list[threading.Semaphore] = []

    def _ensure_accounts(self, count: int) -> None:
        with self._lock:
            while len(self._sems) < count:
                self._sems.append(threading.Semaphore(self.PER_ACCOUNT_LIMIT))

    def acquire(self, request_name: str, account_pool_size: int = 0) -> QueueLease:
        self._ensure_accounts(max(account_pool_size, 3))
        start = time.monotonic()

        while True:
            remaining = self.wait_timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise QueueTimeoutError(f"GLM 队列等待超时，请稍后重试。")

            with self._lock:
                # Pick account with most free slots
                best_idx = max(range(len(self._sems)), key=lambda i: self._sems[i]._value)
                sem = self._sems[best_idx]
                acquired = sem.acquire(blocking=False)

            if acquired:
                self.logger.info("请求获得 GLM 执行槽位 account=%s/%s request=%s", best_idx, len(self._sems), request_name)
                return QueueLease(ticket=best_idx, release_callback=lambda t: self._do_release(t))

            # All accounts busy — wait on condition
            with self._cond:
                self._cond.wait(timeout=min(remaining, 2.0))

    def _do_release(self, account_idx: int) -> None:
        with self._lock:
            if account_idx < len(self._sems):
                self._sems[account_idx].release()
        with self._cond:
            self._cond.notify_all()


class GLMWebClient:
    def __init__(self, config: AppConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self.auth = GLMAccessTokenManager(config=config, logger=logger)
        self.request_queue = ConcurrentRequestQueue(
            logger=logger,
            wait_timeout=config.glm_queue_wait_timeout,
            max_concurrency=config.glm_max_concurrency,
        )
        # Track which account index was used for the last request
        self._last_account_index: int = 0

    def _record_usage(self, account_index: int, response_data: dict[str, object] | None = None) -> None:
        """
        Record tool call / search usage for an account and pre-rotate if near quota.

        Counts tool_calls in the response and records searches.
        If the account is approaching its ~8 tool call / ~5 search limit,
        proactively pre-warms the next guest account.
        """
        if not self.config.glm_use_guest_refresh_token:
            return
        if response_data:
            for choice in response_data.get("choices", []):
                if isinstance(choice, dict):
                    msg = choice.get("message", {})
                    if isinstance(msg, dict):
                        tcs = msg.get("tool_calls", [])
                        if isinstance(tcs, list):
                            for tc in tcs:
                                if isinstance(tc, dict):
                                    fn = tc.get("function", {})
                                    name = fn.get("name", "") if isinstance(fn, dict) else ""
                                    if isinstance(name, str):
                                        self.auth.record_tool_call(account_index)
                                        if name.lower() in ("retrieve", "search", "web_search"):
                                            self.auth.record_search(account_index)
        # If near limit, pre-warm next slot
        if self.auth.is_near_quota_limit(account_index):
            self.logger.info(
                "账号 index=%s 接近配额限制 (tool=%s,search=%s)，提前准备新账号",
                account_index,
                self.auth.get_usage(account_index)[0],
                self.auth.get_usage(account_index)[1],
            )
            # Try to pre-warm the next lazy slot
            prewarmed = self.auth.prewarm_next_guest_slot()
            if prewarmed is not None:
                self.logger.info("已提前预热备用游客账号 index=%s", prewarmed)
            elif self.auth.spare_guest_slot_available():
                self.logger.info("有未预热槽位，将在下次请求时预热")
            else:
                # All slots active - spawn an extra one
                try:
                    fresh = self.auth.spawn_fresh_guest_account()
                    self.logger.info("提前创建新游客账号 index=%s 为配额轮换准备", fresh)
                except Exception as exc:
                    self.logger.warning("预创建新游客账号失败: %s", exc)

    def _resolve_tools(self, openai_payload: dict[str, object]) -> tuple[list[dict[str, object]] | None, set[str] | None]:
        raw_tools = list(openai_payload.get("tools", [])) if isinstance(openai_payload.get("tools"), list) else None # type: ignore
        blocked_tool_names = {
            name.strip()
            for name in self.config.blocked_tool_names
            if name.strip()
        } | BLOCKED_NATIVE_TOOL_NAMES
        filtered_tools = filter_tools(raw_tools, blocked_tool_names)
        if raw_tools and len(raw_tools) != len(filtered_tools or []):
            blocked_names: list[str] = []
            for tool in raw_tools:
                fn = tool.get("function", {})
                tool_name = str(fn.get("name", "")).strip()
                if tool_name in blocked_tool_names:
                    blocked_names.append(tool_name)
            if blocked_names:
                self.logger.info("已过滤不受支持的工具: %s", ", ".join(blocked_names))
        return filtered_tools, {tool["function"]["name"] for tool in filtered_tools} if filtered_tools else None # type: ignore[index]

    def chat_completion(self, payload: dict[str, object]) -> tuple[dict[str, object], str | None]:
        _, allowed_tool_names = self._resolve_tools(payload)
        lease = self.request_queue.acquire(f"chat:{payload.get('model', 'unknown')}", account_pool_size=self.auth.get_account_count())
        try:
            response, assistant_id = self._open_chat_stream(payload, preferred_account_index=self._get_preferred_account_index(lease.ticket), lease=lease)
        except Exception:
            lease.release()
            raise
        account_index = self._last_account_index
        accumulator = GLMEventAccumulator(
            model=str(payload["model"]),
            allowed_tool_names=allowed_tool_names,
            fallback_tool_url=extract_recent_user_url(list(payload.get("messages", []))), # type: ignore[arg-type]
            debug_enabled=self.config.debug_dump_all,
            logger=self.logger,
        )
        # Wire usage callbacks for search/tool tracking
        _chat_account_index = self._last_account_index
        auth_mgr2 = self.auth
        def _on_search_chat():
            auth_mgr2.record_search(_chat_account_index)
            if auth_mgr2.is_near_quota_limit(_chat_account_index):
                self.logger.info("账号 index=%s 接近配额限制 (search)，提前准备新账号", _chat_account_index)
                auth_mgr2.prewarm_next_guest_slot()
        def _on_tool_chat():
            auth_mgr2.record_tool_call(_chat_account_index)
            if auth_mgr2.is_near_quota_limit(_chat_account_index):
                self.logger.info("账号 index=%s 接近配额限制 (tool)，提前准备新账号", _chat_account_index)
                auth_mgr2.prewarm_next_guest_slot()
        accumulator._on_search_callback = _on_search_chat
        accumulator._on_tool_callback = _on_tool_chat
        try:
            for event in self._iter_sse_events(response):
                if not event:
                    continue
                status = event.get("status")
                last_error = event.get("last_error")
                if status != "intervene" or not isinstance(last_error, dict):
                    self._raise_for_event_error(event, stream=False)
                accumulator.consume_event(event)
                if status in {"finish", "intervene"}:
                    result = accumulator.build_response()
                    self._record_usage(account_index, result)
                    # Ponytail: upstream may return finish=stop with zero text
                    # (e.g. upstream timeout or model refusal). Retry once with
                    # a different account to avoid delivering empty responses.
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if not content and not result.get("choices", [{}])[0].get("message", {}).get("tool_calls"):
                        self.logger.warning("上游返回空内容，使用新账号重试 account_index=%s", account_index)
                        self.auth.advance_account(account_index, "empty_content")
                        raise UpstreamAPIError(
                            status_code=502,
                            message="GLM 上游返回空内容，正在使用新账号重试",
                        )
                    return result, accumulator.conversation_id
        finally:
            try:
                response.close() # type: ignore
            except Exception:
                pass
            try:
                self.delete_conversation(accumulator.conversation_id, assistant_id=assistant_id)
            except Exception:
                pass
            lease.release()
        return accumulator.build_response(), accumulator.conversation_id

    def generate_images(self, payload: dict[str, object]) -> dict[str, object]:
        lease = self.request_queue.acquire(f"image:{payload.get('model', self.config.glm_image_model_name)}", account_pool_size=self.auth.get_account_count())
        try:
            response, assistant_id = self._open_image_stream(payload, preferred_account_index=self._get_preferred_account_index(lease.ticket))
        except Exception:
            lease.release()
            raise

        accumulator = GLMEventAccumulator(
            model=str(payload.get("model", self.config.glm_image_model_name)),
            debug_enabled=self.config.debug_dump_all,
            logger=self.logger,
        )
        try:
            for event in self._iter_sse_events(response):
                if not event:
                    continue
                status = event.get("status")
                accumulator.consume_event(event)
                if status == "finish":
                    return self._build_images_response(payload, event, accumulator)

            return self._build_images_response(payload, {}, accumulator)
        finally:
            try:
                response.close() # type: ignore
            except Exception:
                pass
            try:
                self.delete_conversation(accumulator.conversation_id, assistant_id=assistant_id)
            except Exception:
                pass
            lease.release()

    def stream_chat_completion(self, payload: dict[str, object]):
        _, allowed_tool_names = self._resolve_tools(payload)
        lease = self.request_queue.acquire(f"stream:{payload.get('model', 'unknown')}", account_pool_size=self.auth.get_account_count())
        try:
            response, assistant_id = self._open_chat_stream(payload, preferred_account_index=self._get_preferred_account_index(lease.ticket))
        except Exception:
            lease.release()
            raise

        accumulator = GLMEventAccumulator(
            model=str(payload["model"]),
            allowed_tool_names=allowed_tool_names,
            fallback_tool_url=extract_recent_user_url(list(payload.get("messages", []))), # type: ignore[arg-type]
            debug_enabled=self.config.debug_dump_all,
            logger=self.logger,
        )
        # Wire usage callbacks: track searches/tool calls via accumulator
        _st_account_index = self._last_account_index
        auth_mgr = self.auth
        def _on_search():
            auth_mgr.record_search(_st_account_index)
            if auth_mgr.is_near_quota_limit(_st_account_index):
                self.logger.info("账号 index=%s 接近配额限制 (search)，提前准备新账号", _st_account_index)
                auth_mgr.prewarm_next_guest_slot()
        def _on_tool():
            auth_mgr.record_tool_call(_st_account_index)
            if auth_mgr.is_near_quota_limit(_st_account_index):
                self.logger.info("账号 index=%s 接近配额限制 (tool)，提前准备新账号", _st_account_index)
                auth_mgr.prewarm_next_guest_slot()
        accumulator._on_search_callback = _on_search
        accumulator._on_tool_callback = _on_tool

        def generate():
            try:
                for event in self._iter_sse_events(response):
                    if not event:
                        continue
                    status = event.get("status")
                    last_error = event.get("last_error")
                    if status != "intervene" or not isinstance(last_error, dict):
                        self._raise_for_event_error(event, stream=True)
                    chunks, status2 = accumulator.consume_event(event)
                    actual_status = status2 or status
                    for chunk in chunks:
                        yield chunk

                    if actual_status in {"finish", "intervene"}:
                        # On intervene (captcha/plan issue), invalidate this account's
                        # cached token so the next request fetches fresh guest credentials.
                        if actual_status == "intervene" and self.config.glm_use_guest_refresh_token:
                            self.auth.record_silence(lease.ticket % self.auth.get_account_count())
                            self.logger.warning("上游干预/账户失效，标记账号冷却并尝试全量刷新")
                            if self.auth.is_starved():
                                self.auth.force_refresh_all_guest_tokens()
                        for chunk in accumulator.finalize(
                            status=actual_status,
                            last_error=last_error if isinstance(last_error, dict) else None,
                        ):
                            yield chunk
                        return

                for chunk in accumulator.finalize(status="stop"):
                    yield chunk
            finally:
                try:
                    response.close() # type: ignore
                except Exception:
                    pass
                try:
                    self.delete_conversation(accumulator.conversation_id, assistant_id=assistant_id)
                except Exception:
                    pass
                lease.release()

        return generate()

    def _raise_for_event_error(self, event: dict[str, object], stream: bool) -> None:
        status = str(event.get("status", "")).strip().lower()
        last_error = event.get("last_error")
        event_error = self._extract_event_error(event)
        if status != "error" and not event_error and not isinstance(last_error, dict):
            return

        error_payload: dict[str, object] = {}
        if isinstance(last_error, dict):
            # intervene events are not real errors — skip raising
            if last_error.get("intervene_text"):
                return
            error_payload.update(last_error)
        if isinstance(event_error, dict):
            error_payload.update(event_error)
        if not error_payload and status != "error":
            return

        error_code = error_payload.get("error_code", error_payload.get("code"))
        error_message = str(
            error_payload.get("err_msg")
            or error_payload.get("message")
            or ("GLM stream request error" if stream else "GLM request error")
        ).strip()
        detail = f"code={error_code} " if error_code is not None else ""
        raise UpstreamAPIError(
            status_code=502,
            message=f"GLM 上游返回错误 | {detail}{error_message}".strip(),
            payload=error_payload or event,
        )

    def _extract_event_error(self, event: dict[str, object]) -> dict[str, object] | None:
        parts = event.get("parts")
        if not isinstance(parts, list):
            return None
        for part in parts:
            if not isinstance(part, dict):
                continue
            error = part.get("error")
            if isinstance(error, dict) and error:
                return error
            part_status = str(part.get("status", "")).strip().lower()
            if part_status == "error":
                return {"message": "GLM part status error"}
        return None

    def delete_conversation(self, conversation_id: str, assistant_id: str | None = None) -> None:
        if not self.config.glm_delete_conversation:
            return
        if not conversation_id:
            self.logger.debug("跳过删除 GLM 会话：未获取到 conversation_id assistant_id=%s", assistant_id or self.config.glm_assistant_id)
            return

        actual_assistant_id = assistant_id or self.config.glm_assistant_id
        body = json.dumps(
            {
                "assistant_id": actual_assistant_id,
                "conversation_id": conversation_id,
            }
        ).encode("utf-8")
        # ponytail: fire-and-forget — don't block the request on cleanup
        _body = body
        _logger = self.logger
        _auth = self.auth
        _config = self.config
        def _do_delete():
            try:
                timestamp, nonce, sign = build_sign()
                request = urllib.request.Request(
                    _config.delete_conversation_url,
                    method="POST",
                    data=_body,
                    headers={
                        **_auth.get_browser_headers(),
                        "Authorization": f"Bearer {_auth.get_access_token()}",
                        "Referer": "https://chatglm.cn/main/alltoolsdetail",
                        "X-Device-Id": uuid.uuid4().hex,
                        "X-Nonce": nonce,
                        "X-Request-Id": uuid.uuid4().hex,
                        "X-Sign": sign,
                        "X-Timestamp": timestamp,
                    },
                )
                with _get_glm_opener().open(request, timeout=min(_config.request_timeout, 120)) as response:
                    _auth.read_json_response(response)
                _logger.debug("已删除 GLM 会话 conversation_id=%s", conversation_id)
            except Exception:
                pass  # best-effort cleanup, never retry
        threading.Thread(target=_do_delete, daemon=True).start()

    def _open_chat_stream(self, openai_payload: dict[str, object], preferred_account_index: int | None = None, lease: QueueLease | None = None):
        requested_model = str(openai_payload.get("model", "glm-4"))
        upstream_model, assistant_id = resolve_upstream_model(requested_model, self.config)
        filtered_tools, _ = self._resolve_tools(openai_payload)
        converted_messages = convert_messages(
            messages=list(openai_payload.get("messages", [])), # type: ignore
            tools=filtered_tools,
            blocked_tool_names={name.strip() for name in self.config.blocked_tool_names if name.strip()},
            tool_choice=openai_payload.get("tool_choice"),
            server_side_tool_names=SERVER_SIDE_TOOL_NAMES,
        )
        if self.config.glm_use_guest_refresh_token:
            converted_messages = [
                {"role": "system", "content": [{"type": "text", "text": "Respond in English."}]},
                *converted_messages,
            ]
        debug_dump(self.logger, self.config.debug_dump_all, "OpenAI 原始 chat 请求 payload", openai_payload)
        debug_dump(self.logger, self.config.debug_dump_all, "转换后的 GLM messages", converted_messages)
        refs = self._upload_referenced_files(list(openai_payload.get("messages", []))) # type: ignore
        if refs:
            converted_messages[0]["content"] = refs + list(converted_messages[0]["content"]) # type: ignore
            debug_dump(self.logger, self.config.debug_dump_all, "附加上传引用后的 GLM messages", converted_messages)

        chat_mode = resolve_chat_mode(
            model=requested_model,
            reasoning_effort=openai_payload.get("reasoning_effort"),
            deep_research=openai_payload.get("deep_research"),
        )
        is_networking = resolve_networking(
            model=requested_model,
            web_search=openai_payload.get("web_search"),
        )



        # NATIVE UPSTREAM SEARCH FIRST: GLM's built-in search (is_networking=True).
        # DDG background prefetch as fallback when upstream fails / accounts exhaust.
        _prefetch_query = None
        _prefetch_search_results = None
        _should_search = (
            is_networking or "search" in requested_model.lower()
        )
        if _should_search and self.config.glm_use_guest_refresh_token:
            _prefetch_account_idx = preferred_account_index if preferred_account_index is not None else self.auth.get_next_account_index()
            self.auth.record_search(_prefetch_account_idx)
            if self.auth.is_near_quota_limit(_prefetch_account_idx):
                self.logger.info("搜索请求: 账号 %s 接近配额限制，提前准备新账号", _prefetch_account_idx)
                self.auth.prewarm_next_guest_slot()
        # Background DDG prefetch for fallback (cached for _execute_retrieve_tool_calls)
        if _should_search:
            for msg in openai_payload.get("messages", []):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_text = extract_text_content(msg.get("content"))
                    if user_text and user_text.strip():
                        _prefetch_query = user_text.strip()
                        _cache_search_query(_prefetch_query)
                        break


        request_body = json.dumps(
            {
                "assistant_id": assistant_id,
                "conversation_id": "",
                "project_id": "",
                "chat_type": "user_chat",
                "messages": converted_messages,
                "meta_data": {
                    "channel": "",
                    "chat_mode": chat_mode,
                    "draft_id": "",
                    "if_plus_model": True,
                    "input_question_type": "xxxx",
                    "is_networking": is_networking,
                    "is_test": False,
                    "platform": "pc",
                    "quote_log_id": "",
                    "cogview": {"rm_label_watermark": False},
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        self.logger.info(
            "转发请求 model=%s upstream=%s stream=%s",
            requested_model,
            upstream_model,
            openai_payload.get("stream"),
        )
        debug_dump(self.logger, self.config.debug_dump_all, "转发到 GLM 的 chat 原始请求体", request_body)

        def send_request(account_index: int, access_token: str, lease: QueueLease | None = None):
            for attempt in range(self.config.glm_busy_max_retries + 1):
                try:
                    timestamp, nonce, sign = build_sign()
                    request = urllib.request.Request(
                        self.config.chat_stream_url,
                        data=request_body,
                        method="POST",
                        headers={
                            **self.auth.get_browser_headers(),
                            "Authorization": f"Bearer {access_token}",
                            "X-Device-Id": uuid.uuid4().hex,
                            "X-Nonce": nonce,
                            "X-Request-Id": uuid.uuid4().hex,
                            "X-Sign": sign,
                            "X-Timestamp": timestamp,
                        },
                    )
                    debug_dump(
                        self.logger,
                        self.config.debug_dump_all,
                        f"转发到 GLM 的 chat 请求头 account={account_index} attempt={attempt + 1}",
                        dict(request.header_items()),
                    )
                    return self._prepare_chat_response(
                        # ponytail: if hangs persist, reduce self.config.request_timeout below 90
                        _get_glm_opener().open(request, timeout=self.config.request_timeout)
                    )
                except urllib.error.HTTPError as exc:
                    error_payload = self._read_error_payload(exc)
                    # ponytail: handle rate limit (429) by rotating to a new proxy immediately
                    if exc.code == 429 or (error_payload and "too many" in str(error_payload).lower()):
                        self.logger.warning(
                            "GLM 速率限制 (%s), 立即黑名单当前代理并轮换 attempt=%s/%s account=%s",
                            exc.code, attempt + 1, self.config.glm_busy_max_retries, account_index
                        )
                        # Immediately blacklist current proxy so next call picks a different one
                        from .glm2api_proxy import get_pool
                        pool = get_pool()
                        if pool._current:
                            pool.report_rate_limited(pool._current)
                        wait_seconds = self.config.glm_busy_retry_interval * (attempt + 1)
                    elif self._should_retry_busy_error(exc.code, error_payload) and attempt < self.config.glm_busy_max_retries:
                        wait_seconds = self.config.glm_busy_retry_interval * (attempt + 1)  # ponytail: exponential backoff 2,4,6,8,10s
                        self.logger.warning(
                            "GLM 正在处理其他对话，等待重试 attempt=%s/%s wait=%.1fs account=%s",
                            attempt + 1,
                            self.config.glm_busy_max_retries,
                            wait_seconds,
                            account_index,
                        )
                        # Keep the lease during sleep — the slot is still 'ours', just waiting for upstream
                        time.sleep(wait_seconds)
                        continue

                    message = self._build_error_message(exc.code, error_payload)
                    raise UpstreamAPIError(status_code=exc.code, message=message, payload=error_payload) from exc

            raise UpstreamAPIError(status_code=429, message="GLM 长时间忙碌，请稍后重试。")

        response = self._call_with_account_failover(
            f"chat:{requested_model}",
            send_request,
            preferred_account_index=preferred_account_index,
            lease=lease,
        )
        return response, assistant_id

    def _open_image_stream(self, payload: dict[str, object], preferred_account_index: int | None = None):
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise UpstreamAPIError(status_code=400, message="图片生成请求缺少 prompt")

        size = str(payload.get("size", "1024x1024")).strip().lower()
        aspect_ratio = self._resolve_aspect_ratio(size)
        user_model = str(payload.get("model", self.config.glm_image_model_name)).strip() or self.config.glm_image_model_name
        request_body = json.dumps(
            {
                "assistant_id": self.config.glm_image_assistant_id,
                "conversation_id": "",
                "project_id": "",
                "chat_type": "user_chat",
                "meta_data": {
                    "cogview": {
                        "aspect_ratio": aspect_ratio,
                        "style": self._resolve_image_style(payload),
                        "scene": self._resolve_image_scene(payload),
                        "chat_model": "",
                        "rm_label_watermark": False,
                    },
                    "is_test": False,
                    "input_question_type": "xxxx",
                    "channel": "",
                    "draft_id": "",
                    "chat_mode": "",
                    "is_networking": False,
                    "quote_log_id": "",
                    "platform": "pc",
                },
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        self.logger.info(
            "转发绘图请求 model=%s assistant_id=%s size=%s n=%s",
            user_model,
            self.config.glm_image_assistant_id,
            size,
            payload.get("n", 1),
        )
        debug_dump(self.logger, self.config.debug_dump_all, "OpenAI 原始 image 请求 payload", payload)
        debug_dump(self.logger, self.config.debug_dump_all, "转发到 GLM 的 image 原始请求体", request_body)

        def send_request(account_index: int, access_token: str):
            timestamp, nonce, sign = build_sign()
            request = urllib.request.Request(
                self.config.chat_stream_url,
                data=request_body,
                method="POST",
                headers={
                    **self.auth.get_browser_headers(),
                    "Authorization": f"Bearer {access_token}",
                    "X-Device-Id": uuid.uuid4().hex,
                    "X-Nonce": nonce,
                    "X-Request-Id": uuid.uuid4().hex,
                    "X-Sign": sign,
                    "X-Timestamp": timestamp,
                },
            )
            debug_dump(
                self.logger,
                self.config.debug_dump_all,
                f"转发到 GLM 的 image 请求头 account={account_index}",
                dict(request.header_items()),
            )
            try:
                return self._prepare_chat_response(_get_glm_opener().open(request, timeout=self.config.request_timeout))
            except urllib.error.HTTPError as exc:
                error_payload = self._read_error_payload(exc)
                message = self._build_error_message(exc.code, error_payload)
                raise UpstreamAPIError(status_code=exc.code, message=message, payload=error_payload) from exc

        response = self._call_with_account_failover(
            f"image:{user_model}",
            send_request,
            preferred_account_index=preferred_account_index,
        )
        return response, self.config.glm_image_assistant_id

    def _prepare_chat_response(self, response):
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/json" in content_type:
            payload = self.auth.read_json_response(response)
            debug_dump(self.logger, self.config.debug_dump_all, "GLM 非流式原始 JSON 响应", payload)
            status = payload.get("status")
            message = str(payload.get("message", "")).strip()
            # GLM returns status=0, message="ok" for success
            if status is not None and status != 0:
                raise UpstreamAPIError(
                    status_code=502,
                    message=self._build_error_message(200, payload),
                    payload=payload,
                )

            response_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            return BufferedReader(BytesIO(response_body))

        return self._wrap_stream_response(response)

    def _build_images_response(
        self,
        request_payload: dict[str, object],
        final_event: dict[str, object],
        accumulator: GLMEventAccumulator,
    ) -> dict[str, object]:
        requested_count = self._coerce_positive_int(request_payload.get("n"), default=1, maximum=10)
        response_format = str(request_payload.get("response_format", "url")).strip().lower()
        created = int(time.time())

        data: list[dict[str, object]] = []
        ordered_parts = list(accumulator.parts_by_logic_id.values())
        ordered_parts.sort(key=lambda item: str(item.get("logic_id", "")))

        for part in ordered_parts:
            if len(data) >= requested_count:
                break
            if not isinstance(part, dict):
                continue
            part_status = str(part.get("status", ""))
            if part_status != "finish":
                continue
            content_items = part.get("content", [])
            if not isinstance(content_items, list):
                continue
            for content in content_items:
                if len(data) >= requested_count:
                    break
                if not isinstance(content, dict) or content.get("type") != "image":
                    continue
                images = content.get("image", [])
                if not isinstance(images, list):
                    continue
                revised_prompt = str(content.get("code", "")).strip() or None
                for image in images:
                    if len(data) >= requested_count:
                        break
                    if not isinstance(image, dict):
                        continue
                    image_url = str(image.get("image_url", "")).strip()
                    if not image_url:
                        continue
                    item: dict[str, object] = {}
                    if response_format == "b64_json":
                        item["b64_json"] = self._download_image_as_base64(image_url)
                    else:
                        item["url"] = image_url
                    if revised_prompt:
                        item["revised_prompt"] = revised_prompt
                    data.append(item)

        if not data:
            raise UpstreamAPIError(
                status_code=502,
                message="GLM 绘图请求已完成，但未返回可用图片结果。",
                payload=final_event,
            )

        self.logger.info("绘图完成 返回图片数=%s", len(data))
        return {
            "created": created,
            "data": data,
        }

    def _resolve_aspect_ratio(self, size: str) -> str:
        normalized = size.strip().lower()
        if normalized in IMAGE_SIZE_TO_ASPECT_RATIO:
            return IMAGE_SIZE_TO_ASPECT_RATIO[normalized]
        if re.fullmatch(r"\d+x\d+", normalized):
            width_str, height_str = normalized.split("x", 1)
            width = max(int(width_str), 1)
            height = max(int(height_str), 1)
            return f"{width}:{height}"
        return "1:1"

    def _resolve_image_style(self, payload: dict[str, object]) -> str:
        style = str(payload.get("style", "none")).strip().lower()
        return style if style else "none"

    def _resolve_image_scene(self, payload: dict[str, object]) -> str:
        scene = str(payload.get("scene", "none")).strip().lower()
        return scene if scene else "none"

    def _coerce_positive_int(self, value: object, default: int, maximum: int) -> int:
        try:
            parsed = int(value) if value is not None else default # type: ignore
        except (TypeError, ValueError):
            parsed = default
        return max(1, min(parsed, maximum))

    def _download_image_as_base64(self, image_url: str) -> str:
        try:
            with _get_glm_opener().open(image_url, timeout=self.config.request_timeout) as response:
                image_bytes = response.read()
            return base64.b64encode(image_bytes).decode("ascii")
        except Exception as exc:
            raise UpstreamAPIError(status_code=502, message=f"下载图片失败: {image_url} error={exc}") from exc

    def _iter_sse_events(self, response, stream_timeout: int = 0):
        # Set socket read timeout on the response to detect upstream silence
        if stream_timeout <= 0:
            stream_timeout = STREAM_READ_TIMEOUT
        try:
            sock = None
            # Try all plausible socket access paths for urllib responses
            for attr_path in [('fp', 'raw', '_sock'), ('fp', 'buf', 'raw', '_sock'), ('fp', '_sock')]:
                obj = response
                try:
                    for a in attr_path:
                        obj = getattr(obj, a)
                    sock = obj
                    break
                except AttributeError:
                    continue
            if sock is not None:
                sock.settimeout(stream_timeout)
                self.logger.debug("设置 SSE 读取超时=%ss (sock=%s)", stream_timeout, type(sock).__name__)
        except Exception:
            pass

        pending = b""
        _READ_SIZE = 16384  # Ponytail: 16KB buffer — better throughput, still low latency
        last_data_time = time.monotonic()

        def emit_block(payload: str):
            debug_dump(self.logger, self.config.debug_dump_all, "GLM SSE payload", payload)
            if payload == "[DONE]":
                return "[DONE]"
            try:
                parsed = json.loads(payload)
                return parsed
            except json.JSONDecodeError:
                self.logger.debug("忽略无法解析的 SSE 片段: %s", payload[:200])
                return None

        while True:
            stop_after_chunk = False
            try:
                try:
                    fileno = None
                    for f in [response, getattr(response, "fp", None), getattr(getattr(response, "fp", None), "raw", None)]:
                        if hasattr(f, "fileno"):
                            fileno = f.fileno()
                            break
                    if fileno is not None:
                        r, _, _ = select.select([fileno], [], [], stream_timeout)
                        if not r:
                            raise UpstreamAPIError(
                                status_code=504,
                                message=f"上游无响应 (SSE 读取超时 {stream_timeout}s)"
                            )
                except (OSError, AttributeError, ValueError):
                    pass
                raw_chunk = response.read(_READ_SIZE)
                if raw_chunk:
                    last_data_time = time.monotonic()
            except socket.timeout:
                elapsed = time.monotonic() - last_data_time
                self.logger.warning("上游 SSE 读取超时 (%.0fs 无数据)", elapsed)
                raise UpstreamAPIError(
                    status_code=504,
                    message=f"上游无响应 (SSE 读取超时 {stream_timeout}s)"
                )
            except http.client.IncompleteRead as exc:
                raw_chunk = exc.partial or b""
                stop_after_chunk = True
                self.logger.warning("上游 SSE 连接提前断开，按已接收内容收尾 bytes=%s", len(raw_chunk))
            if not raw_chunk:
                break

            pending += raw_chunk

            while b"\n\n" in pending:
                block_bytes, pending = pending.split(b"\n\n", 1)
                block = block_bytes.decode("utf-8", errors="ignore").replace("\r\n", "\n")
                for line in block.split("\n"):
                    if line.startswith("data:"):
                        payload = line[5:].strip()
                        event = emit_block(payload)
                        if event == "[DONE]":
                            return
                        if event is not None:
                            yield event

            if stop_after_chunk:
                break

        if pending.strip():
            remaining = pending.decode("utf-8", errors="ignore")
            for line in remaining.split("\n"):
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    event = emit_block(payload)
                    if event not in (None, "[DONE]"):
                        yield event

    def _upload_referenced_files(self, messages: list[dict[str, object]]) -> list[dict[str, object]]:
        refs: list[dict[str, object]] = []
        upload_tasks: list[tuple[str, bool]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "image_url":
                    url = item.get("image_url", {}).get("url")
                    if isinstance(url, str) and url:
                        upload_tasks.append((url, True))
                elif item_type == "file":
                    url = item.get("file_url", {}).get("url")
                    if isinstance(url, str) and url:
                        upload_tasks.append((url, False))
        
        # ponytail: parallelize file uploads for faster processing
        if upload_tasks:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=min(len(upload_tasks), 10)) as executor:
                futures = {
                    executor.submit(self._upload_file_reference, url, is_image): url
                    for url, is_image in upload_tasks
                }
                for future in as_completed(futures):
                    try:
                        ref = future.result()
                        if ref:
                            refs.append(ref)
                    except Exception as exc:
                        self.logger.warning("上传附件失败 url=%s error=%s", futures[future], exc)
        
        if refs:
            self.logger.info("上传附件完成 成功数=%s", len(refs))
        return refs

    def _upload_file_reference(self, file_url: str, is_image: bool) -> dict[str, object] | None:
        try:
            filename, mime_type, payload = self._fetch_file_payload(file_url)
            boundary = _make_boundary()
            body = self._build_multipart(boundary, filename, mime_type, payload)
            upload_url = f"{self.config.glm_base_url}{FILE_UPLOAD_URL_SUFFIX}"
            debug_dump(
                self.logger,
                self.config.debug_dump_all,
                f"准备上传附件 url={file_url} filename={filename} mime={mime_type}",
                {"filename": filename, "mime_type": mime_type, "bytes": len(payload)},
            )

            def send_request(account_index: int, access_token: str):
                timestamp, nonce, sign = build_sign()
                request = urllib.request.Request(
                    upload_url,
                    method="POST",
                    data=body,
                    headers={
                        **self.auth.get_browser_headers(),
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                        "Referer": "https://chatglm.cn/",
                        "X-Device-Id": uuid.uuid4().hex,
                        "X-Nonce": nonce,
                        "X-Request-Id": uuid.uuid4().hex,
                        "X-Sign": sign,
                        "X-Timestamp": timestamp,
                    },
                )
                debug_dump(
                    self.logger,
                    self.config.debug_dump_all,
                    f"转发到 GLM 的 file_upload 请求头 account={account_index}",
                    dict(request.header_items()),
                )
                debug_dump(
                    self.logger,
                    self.config.debug_dump_all,
                    f"转发到 GLM 的 file_upload 原始请求体 account={account_index}",
                    body,
                )
                return _get_glm_opener().open(request, timeout=self.config.request_timeout)

            with self._call_with_account_failover("file_upload", send_request) as response: # type: ignore
                result = self.auth.read_json_response(response).get("result", {})
            debug_dump(self.logger, self.config.debug_dump_all, "GLM 文件上传响应 result", result)
            source_id = result.get("source_id") # type: ignore
            file_result_url = result.get("file_url", file_url) # type: ignore
            if not source_id:
                return None
            if is_image:
                return {"type": "image_url", "image_url": {"url": file_result_url or source_id}}
            return {"type": "file", "file": [{"source_id": source_id, "file_url": file_result_url}]}
        except Exception as exc:
            self.logger.warning("上传附件失败 url=%s error=%s", file_url, exc)
            return None

    def _fetch_file_payload(self, file_url: str) -> tuple[str, str, bytes]:
        if file_url.startswith("data:"):
            header, encoded = file_url.split(",", 1)
            mime_type = header.split(";")[0][5:] or "application/octet-stream"
            extension = mimetypes.guess_extension(mime_type) or ".bin"
            payload = base64.b64decode(encoded)
            return f"upload-{uuid.uuid4().hex}{extension}", mime_type, payload

        parsed = urllib.parse.urlparse(file_url)
        filename = parsed.path.rsplit("/", 1)[-1] or f"upload-{uuid.uuid4().hex}.bin"
        with _get_glm_opener().open(file_url, timeout=self.config.request_timeout) as response:
            payload = response.read(FILE_SIZE_LIMIT + 1)
            if len(payload) > FILE_SIZE_LIMIT:
                raise ValueError("文件超过 100MB，拒绝上传。")
            mime_type = response.headers.get_content_type()
        mime_type = mime_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return filename, mime_type, payload

    def _build_multipart(self, boundary: str, filename: str, mime_type: str, payload: bytes) -> bytes:
        start = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8")
        end = f"\r\n--{boundary}--\r\n".encode("utf-8")
        return start + payload + end

    def _wrap_stream_response(self, response):
        content_encoding = response.headers.get("Content-Encoding", "").lower()
        if content_encoding == "gzip":
            return BufferedReader(gzip.GzipFile(fileobj=response))
        return response

    def _read_error_payload(self, error: urllib.error.HTTPError) -> dict[str, object]:
        try:
            raw_body = error.read()
            content_encoding = error.headers.get("Content-Encoding", "").lower()

            if content_encoding == "gzip":
                raw_body = gzip.decompress(raw_body)

            text = raw_body.decode("utf-8", errors="ignore")
        except Exception as exc:
            return {"message": f"读取上游错误响应失败: {exc}"}
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        return {"message": text}

    def _should_retry_busy_error(self, status_code: int, payload: dict[str, object]) -> bool:
        if status_code != 429:
            return False
        message = str(payload.get("message", ""))
        inner_status = payload.get("status")
        return inner_status == 10061 or "请等待其他对话生成完毕" in message

    def _build_error_message(self, status_code: int, payload: dict[str, object]) -> str:
        message = str(payload.get("message", "")).strip()
        inner_status = payload.get("status")
        rid = payload.get("rid")
        parts = [f"GLM 请求失败 HTTP {status_code}"]
        if inner_status is not None:
            parts.append(f"status={inner_status}")
        if message:
            parts.append(message)
        if rid:
            parts.append(f"rid={rid}")
        return " | ".join(parts)

    def _get_preferred_account_index(self, ticket: int) -> int | None:
        account_count = self.auth.get_account_count()
        if account_count <= 0:
            return None
        return ticket % account_count

    def _call_with_account_failover(
        self,
        request_name: str,
        operation: Callable[[int, str], object],
        preferred_account_index: int | None = None,
        lease: QueueLease | None = None,
    ):
        account_count = self.auth.get_account_count()
        if account_count <= 0:
            raise RuntimeError("没有可用的 GLM 账号或游客 token 配置")
        start_index = preferred_account_index % account_count if preferred_account_index is not None else self.auth.get_next_account_index()
        last_exc: Exception | None = None

        for offset in range(account_count):
            account_index = (start_index + offset) % account_count
            # ponytail: skip accounts without cached tokens on first pass to avoid slow token refreshes
            if not self.auth.has_cached_token(account_index):
                continue
            guest_retry_limit = self.config.glm_guest_max_retries if self.auth.is_guest_account(account_index) else 0
            for attempt in range(guest_retry_limit + 1):
                try:
                    access_token = self.auth.get_access_token_for_account(account_index)
                    self._last_account_index = account_index
                    import time as _tm
                    _op_start = _tm.time()
                    result = operation(account_index, access_token, lease)
                    _op_elapsed = (_tm.time() - _op_start) * 1000
                    self.auth.record_latency(account_index, _op_elapsed)
                    self.auth.clear_account_failures(account_index)
                    self.auth.record_success()
                    return result
                except Exception as exc:
                    last_exc = exc
                    err_str = str(exc).lower()
                    # ponytail: on SSL errors, rotate proxy immediately
                    if "ssl" in err_str or "certificate" in err_str or "eof" in err_str:
                        from .glm2api_proxy import get_pool as _get_pool
                        _pool = _get_pool()
                        if _pool._current:
                            _pool.report_rate_limited(_pool._current)
                            self.logger.warning("SSL错误, 立即黑名单当前代理 account=%s error=%s", account_index, str(exc)[:80])
                    is_quota_limited = any(kw in err_str for kw in ("多次体验", "请登录", "登录后继续", "rate limit", "too many requests", "频繁"))
                    if is_quota_limited and hasattr(self.auth, "mark_rate_limited"):
                        self.auth.mark_rate_limited(account_index)
                        break
                    should_switch = self.auth.should_switch_account(exc)
                    if should_switch:
                        self.auth.invalidate_account(account_index)
                        # Circuit breaker: track failures, auto-blacklist after threshold
                        self.auth.track_account_failure(account_index)
                    if should_switch and attempt < guest_retry_limit:
                        self.logger.warning(
                            "游客账号请求失败，重新获取游客 ck 重试 attempt=%s/%s request=%s account=%s error=%s",
                            attempt + 1,
                            guest_retry_limit,
                            request_name,
                            account_index,
                            exc,
                        )
                        continue
                    if not should_switch or account_count == 1:
                        raise
                    self.auth.advance_account(account_index, f"{request_name}: {exc}")
                    break

        # Ponytail: guest pool exhausted -- spawn a fresh account on the fly
        # instead of failing.  Each fresh spawn gets ~5 more searches.
        if self.config.glm_use_guest_refresh_token and last_exc is not None:
            try:
                fresh_idx = self.auth.spawn_fresh_guest_account()
                self.logger.info("游客池耗尽，动态创建新账号 index=%s request=%s", fresh_idx, request_name)
                self.auth.reset_account_cycle()
                access_token = self.auth.get_access_token_for_account(fresh_idx)
                return operation(fresh_idx, access_token)
            except Exception as spawn_exc:
                self.logger.warning("动态创建新游客账号也失败 request=%s error=%s", request_name, spawn_exc)
                raise last_exc from spawn_exc

        if hasattr(self.auth, "force_refresh_all_guest_tokens"):
            self.logger.warning("所有账号耗尽，强制刷新游客 token request=%s", request_name)
            self.auth.force_refresh_all_guest_tokens()
            try:
                return operation(0, self.auth.get_access_token_for_account(0))
            except Exception as exc:
                last_exc = exc
        self.auth.reset_account_cycle()
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"账号轮换失败：{request_name}")
