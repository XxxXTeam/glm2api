from __future__ import annotations

import base64
import gzip
import json
import mimetypes
import threading
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.generator import _make_boundary
from io import BufferedReader, BytesIO
from logging import Logger
from typing import Callable

from ..config import AppConfig
from .glm_auth import GLMAccessTokenManager, build_sign
from .translator import GLMEventAccumulator, convert_messages, resolve_chat_mode, resolve_upstream_model


FILE_UPLOAD_URL_SUFFIX = "/backend-api/assistant/file_upload"
FILE_SIZE_LIMIT = 100 * 1024 * 1024


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


class SerialRequestQueue:
    def __init__(self, logger: Logger, wait_timeout: int) -> None:
        self.logger = logger
        self.wait_timeout = wait_timeout
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving_ticket = 0

    def acquire(self, request_name: str) -> QueueLease:
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            queue_ahead = ticket - self._serving_ticket
            start = time.monotonic()

            if queue_ahead > 0:
                self.logger.info("请求进入 GLM 队列 ticket=%s ahead=%s request=%s", ticket, queue_ahead, request_name)

            while ticket != self._serving_ticket:
                remaining = self.wait_timeout - (time.monotonic() - start)
                if remaining <= 0:
                    raise QueueTimeoutError(
                        f"GLM 队列等待超时，前方仍有 {ticket - self._serving_ticket} 个请求，请稍后重试。"
                    )
                self._condition.wait(timeout=remaining)

            self.logger.info("请求获得 GLM 执行槽位 ticket=%s request=%s", ticket, request_name)
            return QueueLease(ticket=ticket, release_callback=self._release)

    def _release(self, ticket: int) -> None:
        with self._condition:
            if ticket == self._serving_ticket:
                self._serving_ticket += 1
                self.logger.info("请求离开 GLM 执行槽位 ticket=%s", ticket)
                self._condition.notify_all()


class GLMWebClient:
    def __init__(self, config: AppConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self.auth = GLMAccessTokenManager(config=config, logger=logger)
        self.request_queue = SerialRequestQueue(logger=logger, wait_timeout=config.glm_queue_wait_timeout)

    def chat_completion(self, payload: dict[str, object]) -> tuple[dict[str, object], str | None]:
        lease = self.request_queue.acquire(f"chat:{payload.get('model', 'unknown')}")
        try:
            response = self._open_chat_stream(payload)
        except Exception:
            lease.release()
            raise
        accumulator = GLMEventAccumulator(model=str(payload["model"]))
        try:
            for event in self._iter_sse_events(response):
                if not event:
                    continue
                status = event.get("status")
                accumulator.consume_event(event)
                if status in {"finish", "intervene"}:
                    return accumulator.build_response(), accumulator.conversation_id
        finally:
            response.close()
            lease.release()
        return accumulator.build_response(), accumulator.conversation_id

    def stream_chat_completion(self, payload: dict[str, object]):
        lease = self.request_queue.acquire(f"stream:{payload.get('model', 'unknown')}")
        try:
            response = self._open_chat_stream(payload)
        except Exception:
            lease.release()
            raise

        accumulator = GLMEventAccumulator(model=str(payload["model"]))

        def generate():
            try:
                for event in self._iter_sse_events(response):
                    if not event:
                        continue
                    chunks, status = accumulator.consume_event(event)
                    for chunk in chunks:
                        yield chunk.encode("utf-8")

                    if status in {"finish", "intervene"}:
                        for chunk in accumulator.finalize(
                            status=status,
                            last_error=event.get("last_error") if isinstance(event.get("last_error"), dict) else None,
                        ):
                            yield chunk.encode("utf-8")
                        return

                for chunk in accumulator.finalize(status="stop"):
                    yield chunk.encode("utf-8")
            finally:
                response.close()
                self.delete_conversation(accumulator.conversation_id)
                lease.release()

        return generate()

    def delete_conversation(self, conversation_id: str) -> None:
        if not conversation_id or not self.config.glm_delete_conversation:
            return

        timestamp, nonce, sign = build_sign()
        access_token = self.auth.get_access_token()
        body = json.dumps(
            {
                "assistant_id": self.config.glm_assistant_id,
                "conversation_id": conversation_id,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.config.delete_conversation_url,
            method="POST",
            data=body,
            headers={
                **self.auth.get_browser_headers(),
                "Authorization": f"Bearer {access_token}",
                "Referer": "https://chatglm.cn/main/alltoolsdetail",
                "X-Device-Id": uuid.uuid4().hex,
                "X-Nonce": nonce,
                "X-Request-Id": uuid.uuid4().hex,
                "X-Sign": sign,
                "X-Timestamp": timestamp,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.request_timeout):
                self.logger.debug("已删除 GLM 会话 conversation_id=%s", conversation_id)
        except Exception as exc:
            self.logger.warning("删除 GLM 会话失败 conversation_id=%s error=%s", conversation_id, exc)

    def _open_chat_stream(self, openai_payload: dict[str, object]):
        requested_model = str(openai_payload.get("model", "glm-4"))
        upstream_model, assistant_id = resolve_upstream_model(requested_model, self.config)
        converted_messages = convert_messages(
            messages=list(openai_payload.get("messages", [])),
            tools=list(openai_payload.get("tools", [])) if isinstance(openai_payload.get("tools"), list) else None,
        )
        refs = self._upload_referenced_files(list(openai_payload.get("messages", [])))
        if refs:
            converted_messages[0]["content"] = refs + list(converted_messages[0]["content"])

        chat_mode = resolve_chat_mode(
            model=upstream_model,
            reasoning_effort=openai_payload.get("reasoning_effort"),
            deep_research=openai_payload.get("deep_research"),
        )

        request_body = json.dumps(
            {
                "assistant_id": assistant_id,
                "conversation_id": "",
                "project_id": "",
                "chat_type": "user_chat",
                "messages": converted_messages,
                "meta_data": {
                    "channel": "",
                    "chat_mode": chat_mode or None,
                    "draft_id": "",
                    "if_plus_model": True,
                    "input_question_type": "openai_compatible",
                    "is_networking": bool(openai_payload.get("web_search")),
                    "is_test": False,
                    "platform": "pc",
                    "quote_log_id": "",
                    "cogview": {"rm_label_watermark": False},
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        timestamp, nonce, sign = build_sign()
        access_token = self.auth.get_access_token()
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

        self.logger.info(
            "转发请求 model=%s upstream=%s stream=%s",
            requested_model,
            upstream_model,
            openai_payload.get("stream"),
        )

        for attempt in range(self.config.glm_busy_max_retries + 1):
            try:
                response = urllib.request.urlopen(request, timeout=self.config.request_timeout)
                return self._prepare_chat_response(response)
            except urllib.error.HTTPError as exc:
                error_payload = self._read_error_payload(exc)
                if self._should_retry_busy_error(exc.code, error_payload) and attempt < self.config.glm_busy_max_retries:
                    wait_seconds = self.config.glm_busy_retry_interval
                    self.logger.warning(
                        "GLM 正在处理其他对话，等待重试 attempt=%s/%s wait=%.1fs",
                        attempt + 1,
                        self.config.glm_busy_max_retries,
                        wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue

                message = self._build_error_message(exc.code, error_payload)
                raise UpstreamAPIError(status_code=exc.code, message=message, payload=error_payload) from exc

        raise UpstreamAPIError(status_code=429, message="GLM 长时间忙碌，请稍后重试。")

    def _prepare_chat_response(self, response):
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/json" in content_type:
            payload = self.auth.read_json_response(response)
            status = payload.get("status")
            message = str(payload.get("message", "")).strip()
            if status not in (0, None) or message:
                raise UpstreamAPIError(
                    status_code=502,
                    message=self._build_error_message(200, payload),
                    payload=payload,
                )

            response_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            return BufferedReader(BytesIO(response_body))

        return self._wrap_stream_response(response)

    def _iter_sse_events(self, response):
        pending = ""

        def emit_block(block: str):
            lines = [line for line in block.split("\n") if line.startswith("data:")]
            if not lines:
                return None
            payload = "\n".join(line[5:].strip() for line in lines)
            if payload == "[DONE]":
                return "[DONE]"
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                self.logger.debug("忽略无法解析的 SSE 片段: %s", payload)
                return None

        while True:
            raw_chunk = response.read(4096)
            if not raw_chunk:
                break

            pending += raw_chunk.decode("utf-8", errors="ignore").replace("\r\n", "\n")

            while "\n\n" in pending:
                block, pending = pending.split("\n\n", 1)
                event = emit_block(block.strip())
                if event == "[DONE]":
                    return
                if event is not None:
                    yield event

        if pending.strip():
            event = emit_block(pending.strip())
            if event not in (None, "[DONE]"):
                yield event

    def _upload_referenced_files(self, messages: list[dict[str, object]]) -> list[dict[str, object]]:
        refs: list[dict[str, object]] = []
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
                        ref = self._upload_file_reference(url, is_image=True)
                        if ref:
                            refs.append(ref)
                elif item_type == "file":
                    url = item.get("file_url", {}).get("url")
                    if isinstance(url, str) and url:
                        ref = self._upload_file_reference(url, is_image=False)
                        if ref:
                            refs.append(ref)
        return refs

    def _upload_file_reference(self, file_url: str, is_image: bool) -> dict[str, object] | None:
        try:
            filename, mime_type, payload = self._fetch_file_payload(file_url)
            boundary = _make_boundary()
            body = self._build_multipart(boundary, filename, mime_type, payload)
            timestamp, nonce, sign = build_sign()
            access_token = self.auth.get_access_token()
            upload_url = f"{self.config.glm_base_url}{FILE_UPLOAD_URL_SUFFIX}"
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
            with urllib.request.urlopen(request, timeout=self.config.request_timeout) as response:
                result = self.auth.read_json_response(response).get("result", {})
            source_id = result.get("source_id")
            file_result_url = result.get("file_url", file_url)
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
        with urllib.request.urlopen(file_url, timeout=self.config.request_timeout) as response:
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
        raw_body = error.read()
        content_encoding = error.headers.get("Content-Encoding", "").lower()

        if content_encoding == "gzip":
            raw_body = gzip.decompress(raw_body)

        text = raw_body.decode("utf-8", errors="ignore")
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
