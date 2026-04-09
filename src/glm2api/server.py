from __future__ import annotations

import json
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging import Logger

from .config import AppConfig
from .services.glm_client import GLMWebClient


class GLM2APIServer:
    def __init__(self, config: AppConfig, glm_client: GLMWebClient, logger: Logger) -> None:
        self.config = config
        self.glm_client = glm_client
        self.logger = logger
        handler_cls = self._build_handler()
        self._server = ThreadingHTTPServer((config.host, config.port), handler_cls)

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def _build_handler(self):
        config = self.config
        glm_client = self.glm_client
        logger = self.logger

        class RequestHandler(BaseHTTPRequestHandler):
            server_version = "glm2api/0.1.0"

            def do_OPTIONS(self) -> None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self._send_common_headers()
                self.end_headers()

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._write_json(HTTPStatus.OK, {"status": "ok"})
                    return

                if self.path == f"{config.api_prefix}/models":
                    self._write_json(
                        HTTPStatus.OK,
                        {
                            "object": "list",
                            "data": [
                                {"id": model, "object": "model", "owned_by": "glm2api"}
                                for model in config.exposed_models
                            ],
                        },
                    )
                    return

                self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not Found"}})

            def do_POST(self) -> None:
                try:
                    if self.path != f"{config.api_prefix}/chat/completions":
                        self._write_json(HTTPStatus.NOT_FOUND, {"error": {"message": "Not Found"}})
                        return

                    if not self._authorize():
                        self._write_json(HTTPStatus.UNAUTHORIZED, {"error": {"message": "Unauthorized"}})
                        return

                    content_length = int(self.headers.get("Content-Length", "0"))
                    raw_body = self.rfile.read(content_length) if content_length else b"{}"
                    payload = json.loads(raw_body.decode("utf-8"))

                    if not isinstance(payload.get("messages"), list) or not payload.get("model"):
                        self._write_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": {"message": "请求体必须包含 model 和 messages 字段。"}},
                        )
                        return

                    if payload.get("stream"):
                        self._stream_completion(payload)
                        return

                    result, conversation_id = glm_client.chat_completion(payload)
                    self._write_json(HTTPStatus.OK, result)
                    glm_client.delete_conversation(conversation_id or "")
                except Exception as exc:
                    logger.error("处理请求失败 error=%s\n%s", exc, traceback.format_exc())
                    self._write_json(
                        HTTPStatus.BAD_GATEWAY,
                        {"error": {"message": str(exc), "type": exc.__class__.__name__}},
                    )

            def _stream_completion(self, payload: dict[str, object]) -> None:
                self.send_response(HTTPStatus.OK)
                self._send_common_headers()
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                for chunk in glm_client.stream_chat_completion(payload):
                    if chunk:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                logger.info("流式请求完成 model=%s", payload.get("model"))

            def _authorize(self) -> bool:
                if not config.server_api_keys:
                    return True
                authorization = self.headers.get("Authorization", "")
                if not authorization.startswith("Bearer "):
                    return False
                token = authorization[7:].strip()
                return token in config.server_api_keys

            def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
                body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self._send_common_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_common_headers(self) -> None:
                self.send_header("Access-Control-Allow-Origin", config.cors_allow_origin)
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

            def log_message(self, format: str, *args) -> None:
                logger.info("%s - %s", self.address_string(), format % args)

        return RequestHandler
