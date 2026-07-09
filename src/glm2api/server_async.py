"""Async HTTP server for glm2api using aiohttp.

Provides the same API endpoints as GLM2APIServer (server.py) but using
aiohttp for async I/O.  Blocking GLM client calls (chat_completion,
stream_chat_completion, generate_images) are forwarded to a thread-pool
executor so the event loop stays responsive during the incremental migration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import traceback
from typing import Any

from aiohttp import web

from .config import AppConfig
from .logging_utils import debug_dump
from .services.anthropic_adapter import (
    AnthropicStreamAccumulator,
    anthropic_to_openai,
    openai_to_anthropic_response,
)
from .services.glm_client import GLMWebClient, QueueTimeoutError, UpstreamAPIError
from .services.responses_adapter import (
    ResponsesStreamAccumulator,
    openai_to_responses,
    responses_to_openai,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLIENT_DISCONNECTED = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)
_RESPONSES_STREAM_HEARTBEAT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Exception for structured body-parse errors
# ---------------------------------------------------------------------------

class _BodyParseError(ValueError):
    """Raised by _parse_body to carry both a message and an error type."""

    def __init__(self, message: str, error_type: str = "invalid_request") -> None:
        super().__init__(message, error_type)


# ---------------------------------------------------------------------------
# Async server class
# ---------------------------------------------------------------------------

class AsyncGLM2APIServer:
    """aiohttp-based async HTTP server for glm2api.

    Provides the same endpoints as :class:`GLM2APIServer`:

      GET  /health
      GET  {api_prefix}/models
      GET  /proxy/status
      POST {api_prefix}/chat/completions   (streaming + non-streaming)
      POST {api_prefix}/images/generations
      POST {api_prefix}/messages            (Anthropic adapter)
      POST /anthropic/v1/messages
      POST {api_prefix}/responses           (OpenAI Responses adapter)

    CORS headers, API-key auth (Bearer / x-api-key), TCP_NODELAY, SSE
    streaming, client-disconnect detection, and graceful shutdown are all
    supported.
    """

    def __init__(
        self,
        config: AppConfig,
        glm_client: GLMWebClient,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.glm_client = glm_client
        self.logger = logger

        self._app = web.Application()
        self._app.middlewares.append(self._options_middleware)
        self._app.on_response_prepare.append(self._on_prepare_cors)
        self._setup_routes()
        self._register_cleanup()

    # -- public helpers ---------------------------------------------------

    @property
    def app(self) -> web.Application:
        return self._app

    @staticmethod
    def create_async_application(
        config: AppConfig,
        glm_client: GLMWebClient,
        logger: logging.Logger,
    ) -> web.Application:
        """Convenience factory: build and return the configured aiohttp app."""
        server = AsyncGLM2APIServer(config=config, glm_client=glm_client, logger=logger)
        return server.app

    # -- middleware / signals ---------------------------------------------

    @staticmethod
    async def _options_middleware(
        request: web.Request, handler: Any
    ) -> web.StreamResponse:
        """Intercept CORS preflight (OPTIONS) requests and return 204."""
        if request.method == "OPTIONS":
            return web.Response(status=204)
        return await handler(request)

    async def _on_prepare_cors(
        self, _request: web.Request, response: web.StreamResponse
    ) -> None:
        """Attach CORS headers to *every* response before headers are sent.

        This fires during ``StreamResponse.prepare()`` so it works for
        both regular JSON responses and SSE streams.
        """
        response.headers["Access-Control-Allow-Origin"] = self.config.cors_allow_origin
        response.headers["Access-Control-Allow-Headers"] = (
            "Authorization, Content-Type, x-api-key, anthropic-version"
        )
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"

    # -- route table ------------------------------------------------------

    def _setup_routes(self) -> None:
        prefix = self.config.api_prefix

        # GET
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get(f"{prefix}/models", self._handle_models)
        self._app.router.add_get("/proxy/status", self._handle_proxy_status)

        # POST
        self._app.router.add_post(
            f"{prefix}/chat/completions", self._handle_chat_completions
        )
        self._app.router.add_post(
            f"{prefix}/images/generations", self._handle_images
        )
        self._app.router.add_post(f"{prefix}/messages", self._handle_anthropic)
        self._app.router.add_post(
            "/anthropic/v1/messages", self._handle_anthropic
        )
        self._app.router.add_post(
            f"{prefix}/responses", self._handle_responses
        )

    # -- lifecycle hooks --------------------------------------------------

    def _register_cleanup(self) -> None:
        self._app.on_shutdown.append(self._on_shutdown)
        self._app.on_cleanup.append(self._on_cleanup)

    @staticmethod
    async def _on_shutdown(_app: web.Application) -> None:
        logging.getLogger("glm2api.http").info("正在关闭异步 HTTP 服务...")

    @staticmethod
    async def _on_cleanup(_app: web.Application) -> None:
        logger = logging.getLogger("glm2api.http")
        logger.info("清理异步 HTTP 服务资源...")
        try:
            from .services.http_client import close_all

            close_all()
        except Exception as exc:
            logger.warning("close_all 失败: %s", exc)

    # ------------------------------------------------------------------
    #  TCP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_tcp_nodelay(request: web.Request) -> None:
        """Disable Nagle's algorithm for lower-latency streaming."""
        try:
            transport = request.transport
            if transport is not None:
                sock = transport.get_extra_info("socket")
                if sock is not None:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except (OSError, AttributeError):
            pass

    # ------------------------------------------------------------------
    #  GET handlers
    # ------------------------------------------------------------------

    @staticmethod
    async def _handle_health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_models(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {"id": model, "object": "model", "owned_by": "glm2api"}
                    for model in self.config.exposed_models
                ],
            }
        )

    @staticmethod
    async def _handle_proxy_status(_request: web.Request) -> web.Response:
        from .services.glm2api_proxy import get_pool

        pool = get_pool()
        status = pool.get_status()
        return web.json_response(status)

    # ------------------------------------------------------------------
    #  Body parsing
    # ------------------------------------------------------------------

    async def _parse_body(self, request: web.Request) -> dict[str, Any]:
        """Read, validate, and JSON-parse the request body.

        Raises ``_BodyParseError`` (a ``ValueError`` subclass) on any
        validation failure.  The caller should catch ``ValueError`` and
        check ``exc.args`` for ``(message, error_type)``.
        """
        raw_value = request.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_value or "0")
        except ValueError as exc:
            raise _BodyParseError(
                f"无效的 Content-Length: {raw_value}", "invalid_content_length"
            ) from exc
        if content_length < 0:
            raise _BodyParseError(
                "Content-Length 不能为负数。", "invalid_content_length"
            )

        raw_body = await request.read() if content_length else b"{}"
        debug_dump(
            self.logger,
            self.config.debug_dump_all,
            f"HTTP 入站原始请求体 path={request.path}",
            raw_body,
        )
        try:
            payload: dict[str, Any] = json.loads(raw_body.decode("utf-8"))
        except UnicodeDecodeError:
            raise _BodyParseError(
                "请求体必须是 UTF-8 编码。", "invalid_encoding"
            )
        except json.JSONDecodeError as exc:
            raise _BodyParseError(
                f"请求体不是合法 JSON: {exc.msg}", "invalid_json"
            ) from exc

        if not isinstance(payload, dict):
            raise _BodyParseError(
                "请求体顶层必须是 JSON 对象。", "invalid_payload"
            )

        debug_dump(
            self.logger,
            self.config.debug_dump_all,
            f"HTTP 入站解析后 JSON path={request.path}",
            payload,
        )
        return payload

    # ------------------------------------------------------------------
    #  JSON response helper
    # ------------------------------------------------------------------

    @staticmethod
    def _json_response(
        data: dict[str, Any], status: int = 200
    ) -> web.Response:
        """Build a JSON :class:`web.Response` with compact encoding."""
        body = json.dumps(
            data, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return web.Response(
            body=body,
            status=status,
            content_type="application/json; charset=utf-8",
        )

    # ------------------------------------------------------------------
    #  Auth
    # ------------------------------------------------------------------

    def _authorize(self, request: web.Request) -> bool:
        if not self.config.server_api_keys:
            return True
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            token = authorization[7:].strip()
            if token in self.config.server_api_keys:
                return True
        x_api_key = request.headers.get("x-api-key", "")
        if x_api_key and x_api_key.strip() in self.config.server_api_keys:
            return True
        return False

    # ------------------------------------------------------------------
    #  Generator iteration helper
    # ------------------------------------------------------------------

    @staticmethod
    def _next_chunk(gen: Any) -> Any:
        """Return the next item from a sync generator, or ``None`` on stop."""
        try:
            return next(gen)
        except StopIteration:
            return None

    # ------------------------------------------------------------------
    #  POST /v1/chat/completions
    # ------------------------------------------------------------------

    async def _handle_chat_completions(
        self, request: web.Request
    ) -> web.StreamResponse:
        self._set_tcp_nodelay(request)
        self.logger.debug("HTTP 入站请求 POST %s", request.path)

        if not self._authorize(request):
            self.logger.warning(
                "认证失败 path=%s ip=%s",
                request.path,
                request.remote,
            )
            return self._json_response(
                {"error": {"message": "Unauthorized"}}, 401
            )

        try:
            payload = await self._parse_body(request)
        except ValueError as exc:
            msg = exc.args[0] if exc.args else str(exc)
            etype = exc.args[1] if len(exc.args) > 1 else "invalid_request"
            return self._json_response(
                {"error": {"message": msg, "type": etype}}, 400
            )

        # inject User-Agent for client detection in tool-call format
        payload["_user_agent"] = request.headers.get("User-Agent", "")

        if payload.get("stream"):
            return await self._stream_completion(request, payload)

        # ---- non-streaming ----
        if not isinstance(payload.get("messages"), list) or not payload.get("model"):
            return self._json_response(
                {
                    "error": {
                        "message": "请求体必须包含 model 和 messages 字段。"
                    }
                },
                400,
            )

        self.logger.info(
            "收到 chat 请求 model=%s", payload.get("model")
        )
        loop = asyncio.get_event_loop()
        try:
            result, conversation_id = await loop.run_in_executor(
                None, self.glm_client.chat_completion, payload
            )
        except QueueTimeoutError as exc:
            self.logger.warning("GLM 队列等待超时 error=%s", exc)
            return self._json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": "queue_timeout",
                    }
                },
                503,
            )
        except UpstreamAPIError as exc:
            self.logger.warning(
                "上游 GLM 返回错误 status=%s error=%s",
                exc.status_code,
                exc,
            )
            status = (
                exc.status_code
                if 400 <= exc.status_code < 600
                else 502
            )
            return self._json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": "upstream_error",
                        "details": exc.payload,
                    }
                },
                status,
            )
        except ValueError as exc:
            self.logger.warning(
                "请求参数错误 path=%s error=%s", request.path, exc
            )
            return self._json_response(
                {"error": {"message": str(exc), "type": "invalid_request"}},
                400,
            )
        except Exception as exc:
            self.logger.error(
                "处理请求失败 error=%s\n%s", exc, traceback.format_exc()
            )
            return self._json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": exc.__class__.__name__,
                    }
                },
                502,
            )

        return self._json_response(result)

    # ------------------------------------------------------------------
    #  SSE streaming  /v1/chat/completions?stream=true
    # ------------------------------------------------------------------

    async def _stream_completion(
        self, request: web.Request, payload: dict[str, Any]
    ) -> web.StreamResponse:
        model = str(payload.get("model", "unknown"))
        self.logger.info("开始流式响应 model=%s", model)

        loop = asyncio.get_event_loop()

        # Acquire generator *before* starting SSE so early errors (queue
        # timeout, auth failure, ...) still produce a JSON error response.
        try:
            gen = await loop.run_in_executor(
                None, self.glm_client.stream_chat_completion, payload
            )
        except QueueTimeoutError as exc:
            self.logger.warning("GLM 队列等待超时 error=%s", exc)
            return self._json_response(
                {"error": {"message": str(exc), "type": "queue_timeout"}},
                503,
            )
        except UpstreamAPIError as exc:
            self.logger.warning(
                "上游 GLM 返回错误 status=%s error=%s",
                exc.status_code,
                exc,
            )
            status = (
                exc.status_code
                if 400 <= exc.status_code < 600
                else 502
            )
            return self._json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": "upstream_error",
                        "details": exc.payload,
                    }
                },
                status,
            )
        except ValueError as exc:
            return self._json_response(
                {"error": {"message": str(exc), "type": "invalid_request"}},
                400,
            )

        # ---- start SSE stream ----
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        sent_done = False
        completed_normally = False
        chunk_count = 0

        try:
            while True:
                chunk = await loop.run_in_executor(
                    None, self._next_chunk, gen
                )
                if chunk is None:
                    completed_normally = True
                    break

                chunk_count += 1
                debug_dump(
                    self.logger,
                    self.config.debug_dump_all,
                    f"HTTP 出站流式分片 model={model}",
                    chunk,
                )
                await response.write(chunk)

                if b"data: [DONE]\n\n" in chunk:
                    sent_done = True
        except UpstreamAPIError as exc:
            self.logger.warning(
                "流式请求中途收到上游错误 status=%s error=%s",
                exc.status_code,
                exc,
            )
            await self._write_sse_error(response, str(exc), "upstream_error")
        except _CLIENT_DISCONNECTED as exc:
            self.logger.warning(
                "客户端在流式响应过程中断开 model=%s error=%s",
                model,
                exc,
            )
            return response
        except Exception as exc:
            self.logger.error(
                "流式请求失败 model=%s error=%s\n%s",
                model,
                exc,
                traceback.format_exc(),
            )
            await self._write_sse_error(
                response, str(exc), exc.__class__.__name__
            )
        finally:
            if not sent_done and completed_normally:
                try:
                    await response.write(b"data: [DONE]\n\n")
                except _CLIENT_DISCONNECTED:
                    pass
            try:
                await response.write_eof()
            except Exception:
                pass

        self.logger.info(
            "流式请求完成 model=%s chunk_count=%s", model, chunk_count
        )
        return response

    @staticmethod
    async def _write_sse_error(
        response: web.StreamResponse, message: str, error_type: str
    ) -> None:
        """Write an SSE error event to an active stream."""
        event = json.dumps(
            {"error": {"message": message, "type": error_type}},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            await response.write(f"data: {event}\n\n".encode("utf-8"))
        except _CLIENT_DISCONNECTED:
            pass

    # ------------------------------------------------------------------
    #  POST /v1/images/generations
    # ------------------------------------------------------------------

    async def _handle_images(
        self, request: web.Request
    ) -> web.StreamResponse:
        self._set_tcp_nodelay(request)

        if not self._authorize(request):
            return self._json_response(
                {"error": {"message": "Unauthorized"}}, 401
            )

        try:
            payload = await self._parse_body(request)
        except ValueError as exc:
            msg = exc.args[0] if exc.args else str(exc)
            etype = exc.args[1] if len(exc.args) > 1 else "invalid_request"
            return self._json_response(
                {"error": {"message": msg, "type": etype}}, 400
            )

        payload["_user_agent"] = request.headers.get("User-Agent", "")

        if not payload.get("prompt"):
            return self._json_response(
                {
                    "error": {
                        "message": "图片生成请求必须包含 prompt 字段。"
                    }
                },
                400,
            )

        self.logger.info(
            "收到绘图请求 model=%s prompt=%s",
            payload.get("model"),
            payload.get("prompt"),
        )

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, self.glm_client.generate_images, payload
            )
        except QueueTimeoutError as exc:
            return self._json_response(
                {"error": {"message": str(exc), "type": "queue_timeout"}},
                503,
            )
        except UpstreamAPIError as exc:
            status = (
                exc.status_code
                if 400 <= exc.status_code < 600
                else 502
            )
            return self._json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": "upstream_error",
                        "details": exc.payload,
                    }
                },
                status,
            )
        except ValueError as exc:
            return self._json_response(
                {"error": {"message": str(exc), "type": "invalid_request"}},
                400,
            )
        except Exception as exc:
            self.logger.error(
                "处理绘图请求失败 error=%s\n%s",
                exc,
                traceback.format_exc(),
            )
            return self._json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": exc.__class__.__name__,
                    }
                },
                502,
            )

        return self._json_response(result)

    # ------------------------------------------------------------------
    #  Anthropic Messages API  (/v1/messages  &  /anthropic/v1/messages)
    # ------------------------------------------------------------------

    async def _handle_anthropic(
        self, request: web.Request
    ) -> web.StreamResponse:
        self._set_tcp_nodelay(request)

        if not self._authorize(request):
            return self._json_response(
                {"error": {"message": "Unauthorized"}}, 401
            )

        try:
            payload = await self._parse_body(request)
        except ValueError as exc:
            msg = exc.args[0] if exc.args else str(exc)
            etype = exc.args[1] if len(exc.args) > 1 else "invalid_request"
            return self._json_response(
                {"error": {"message": msg, "type": etype}}, 400
            )

        payload["_user_agent"] = request.headers.get("User-Agent", "")

        model = str(payload.get("model", "glm-4"))
        openai_payload = anthropic_to_openai(payload)

        if payload.get("stream"):
            return await self._stream_anthropic(
                request, openai_payload, model
            )

        # ---- non-streaming ----
        self.logger.info(
            "收到 Anthropic 请求 model=%s path=%s", model, request.path
        )
        loop = asyncio.get_event_loop()
        try:
            result, _ = await loop.run_in_executor(
                None, self.glm_client.chat_completion, openai_payload
            )
        except QueueTimeoutError as exc:
            return self._json_response(
                {"error": {"message": str(exc), "type": "queue_timeout"}},
                503,
            )
        except UpstreamAPIError as exc:
            status = (
                exc.status_code
                if 400 <= exc.status_code < 600
                else 502
            )
            return self._json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": "upstream_error",
                        "details": exc.payload,
                    }
                },
                status,
            )
        except ValueError as exc:
            return self._json_response(
                {"error": {"message": str(exc), "type": "invalid_request"}},
                400,
            )
        except Exception as exc:
            self.logger.error(
                "处理 Anthropic 请求失败 error=%s\n%s",
                exc,
                traceback.format_exc(),
            )
            return self._json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": exc.__class__.__name__,
                    }
                },
                502,
            )

        response = openai_to_anthropic_response(result, model)
        return self._json_response(response)

    async def _stream_anthropic(
        self,
        request: web.Request,
        openai_payload: dict[str, Any],
        model: str,
    ) -> web.StreamResponse:
        """SSE streaming for Anthropic Messages API."""
        openai_payload["stream"] = True
        accumulator = AnthropicStreamAccumulator(model=model)

        loop = asyncio.get_event_loop()

        try:
            gen = await loop.run_in_executor(
                None, self.glm_client.stream_chat_completion, openai_payload
            )
        except QueueTimeoutError as exc:
            return self._json_response(
                {"error": {"message": str(exc), "type": "queue_timeout"}},
                503,
            )
        except UpstreamAPIError as exc:
            status = (
                exc.status_code
                if 400 <= exc.status_code < 600
                else 502
            )
            return self._json_response(
                {"error": {"message": str(exc), "type": "upstream_error"}},
                status,
            )
        except ValueError as exc:
            return self._json_response(
                {"error": {"message": str(exc), "type": "invalid_request"}},
                400,
            )

        # ---- start SSE stream ----
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        try:
            while True:
                chunk = await loop.run_in_executor(
                    None, self._next_chunk, gen
                )
                if chunk is None:
                    break
                if not chunk:
                    continue

                if not accumulator.started:
                    start_event = accumulator.start_message()
                    await response.write(start_event.encode("utf-8"))

                events = accumulator.feed_chunk(chunk)
                for event in events:
                    await response.write(event.encode("utf-8"))
        except _CLIENT_DISCONNECTED as exc:
            self.logger.warning(
                "客户端在 Anthropic 流式响应过程中断开 model=%s error=%s",
                model,
                exc,
            )
            return response
        except Exception as exc:
            self.logger.error(
                "Anthropic 流式请求失败 model=%s error=%s\n%s",
                model,
                exc,
                traceback.format_exc(),
            )
            error_event = json.dumps(
                {
                    "type": "error",
                    "error": {
                        "message": str(exc),
                        "type": "upstream_error",
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            try:
                await response.write(
                    f"data: {error_event}\n\n".encode("utf-8")
                )
            except _CLIENT_DISCONNECTED:
                pass

        # Ensure message_stop is always sent (idempotent via _finished flag)
        if accumulator.started:
            try:
                for event in accumulator._finish():
                    await response.write(event.encode("utf-8"))
            except _CLIENT_DISCONNECTED:
                pass

        await response.write_eof()
        self.logger.info("Anthropic 流式请求完成 model=%s", model)
        return response

    # ------------------------------------------------------------------
    #  OpenAI Responses API  (/v1/responses)
    # ------------------------------------------------------------------

    async def _handle_responses(
        self, request: web.Request
    ) -> web.StreamResponse:
        self._set_tcp_nodelay(request)

        if not self._authorize(request):
            return self._json_response(
                {"error": {"message": "Unauthorized"}}, 401
            )

        try:
            payload = await self._parse_body(request)
        except ValueError as exc:
            msg = exc.args[0] if exc.args else str(exc)
            etype = exc.args[1] if len(exc.args) > 1 else "invalid_request"
            return self._json_response(
                {"error": {"message": msg, "type": etype}}, 400
            )

        payload["_user_agent"] = request.headers.get("User-Agent", "")

        model = str(payload.get("model", "glm-4"))
        openai_payload = responses_to_openai(payload)

        if payload.get("stream"):
            return await self._stream_responses(
                request, openai_payload, model
            )

        # ---- non-streaming ----
        self.logger.info(
            "收到 Responses 请求 model=%s", model
        )
        loop = asyncio.get_event_loop()
        try:
            result, _ = await loop.run_in_executor(
                None, self.glm_client.chat_completion, openai_payload
            )
        except QueueTimeoutError as exc:
            return self._json_response(
                {"error": {"message": str(exc), "type": "queue_timeout"}},
                503,
            )
        except UpstreamAPIError as exc:
            status = (
                exc.status_code
                if 400 <= exc.status_code < 600
                else 502
            )
            return self._json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": "upstream_error",
                        "details": exc.payload,
                    }
                },
                status,
            )
        except ValueError as exc:
            return self._json_response(
                {"error": {"message": str(exc), "type": "invalid_request"}},
                400,
            )
        except Exception as exc:
            self.logger.error(
                "处理 Responses 请求失败 error=%s\n%s",
                exc,
                traceback.format_exc(),
            )
            return self._json_response(
                {
                    "error": {
                        "message": str(exc),
                        "type": exc.__class__.__name__,
                    }
                },
                502,
            )

        response = openai_to_responses(result, model)
        return self._json_response(response)

    async def _stream_responses(
        self,
        request: web.Request,
        openai_payload: dict[str, Any],
        model: str,
    ) -> web.StreamResponse:
        """SSE streaming for OpenAI Responses API with keep-alive heartbeat."""
        openai_payload["stream"] = True
        accumulator = ResponsesStreamAccumulator(model=model)

        loop = asyncio.get_event_loop()

        try:
            gen = await loop.run_in_executor(
                None, self.glm_client.stream_chat_completion, openai_payload
            )
        except QueueTimeoutError as exc:
            return self._json_response(
                {"error": {"message": str(exc), "type": "queue_timeout"}},
                503,
            )
        except UpstreamAPIError as exc:
            status = (
                exc.status_code
                if 400 <= exc.status_code < 600
                else 502
            )
            return self._json_response(
                {"error": {"message": str(exc), "type": "upstream_error"}},
                status,
            )
        except ValueError as exc:
            return self._json_response(
                {"error": {"message": str(exc), "type": "invalid_request"}},
                400,
            )

        # ---- start SSE stream ----
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        await response.prepare(request)

        # Async queue decouples the upstream reader (which runs the sync
        # generator in an executor thread) from the async SSE writer.
        chunk_queue: asyncio.Queue[Any] = asyncio.Queue()
        _SENTINEL = object()

        async def _read_upstream() -> None:
            """Read chunks from the sync generator in executor → async queue."""
            try:
                while True:
                    chunk = await loop.run_in_executor(
                        None, self._next_chunk, gen
                    )
                    if chunk is None:
                        break
                    await chunk_queue.put(chunk)
            except BaseException as exc:
                await chunk_queue.put(exc)
            finally:
                await chunk_queue.put(_SENTINEL)

        reader_task = asyncio.ensure_future(_read_upstream())

        try:
            while True:
                try:
                    queued = await asyncio.wait_for(
                        chunk_queue.get(),
                        timeout=_RESPONSES_STREAM_HEARTBEAT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    # No data from upstream within heartbeat window — send
                    # a keep-alive comment so proxies / load-balancers don't
                    # drop the connection.
                    await response.write(b": keep-alive\n\n")
                    continue

                if queued is _SENTINEL:
                    break
                if isinstance(queued, BaseException):
                    raise queued  # type: ignore[arg-type]
                chunk = queued
                if not chunk:
                    continue

                if not accumulator.started:
                    start_events = accumulator.start_response()
                    for event in start_events:
                        await response.write(event.encode("utf-8"))

                events = accumulator.feed_chunk(chunk)
                for event in events:
                    await response.write(event.encode("utf-8"))
        except _CLIENT_DISCONNECTED as exc:
            self.logger.warning(
                "客户端在 Responses 流式响应过程中断开 model=%s error=%s",
                model,
                exc,
            )
            return response
        except Exception as exc:
            self.logger.error(
                "Responses 流式请求失败 model=%s error=%s\n%s",
                model,
                exc,
                traceback.format_exc(),
            )
            error_event = json.dumps(
                {
                    "type": "error",
                    "error": {
                        "message": str(exc),
                        "type": "upstream_error",
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            try:
                await response.write(
                    f"data: {error_event}\n\n".encode("utf-8")
                )
            except _CLIENT_DISCONNECTED:
                pass
        finally:
            reader_task.cancel()

        # Ensure response.completed is always sent (idempotent via _finished)
        if accumulator.started:
            try:
                for event in accumulator._finish():
                    await response.write(event.encode("utf-8"))
            except _CLIENT_DISCONNECTED:
                pass

        await response.write_eof()
        self.logger.info("Responses 流式请求完成 model=%s", model)
        return response


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def create_async_application(
    config: AppConfig,
    glm_client: GLMWebClient,
    logger: logging.Logger,
) -> web.Application:
    """Create an aiohttp :class:`~aiohttp.web.Application` wired to the async
    GLM2API server.

    Usage::

        server = AsyncGLM2APIServer(config, client, logger)
        web.run_app(server.app, host=config.host, port=config.port)
    """
    return AsyncGLM2APIServer.create_async_application(
        config=config, glm_client=glm_client, logger=logger
    )
