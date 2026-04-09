from __future__ import annotations

import hashlib
import gzip
import json
import random
import threading
import time
import uuid
import urllib.request
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

from ..config import AppConfig


SIGN_SECRET = "8a1317a7468aa3ad86e997d08f3f31cb"
ACCESS_TOKEN_EXPIRES_SECONDS = 3600


def build_sign() -> tuple[str, str, str]:
    now = str(int(time.time() * 1000))
    digits = [int(char) for char in now]
    checksum = (sum(digits) - digits[-2]) % 10
    timestamp = now[:-2] + str(checksum) + now[-1]
    nonce = uuid.uuid4().hex
    sign = hashlib.md5(f"{timestamp}-{nonce}-{SIGN_SECRET}".encode("utf-8")).hexdigest()
    return timestamp, nonce, sign


@dataclass(slots=True)
class AccessToken:
    access_token: str
    refresh_token: str
    expires_at: float


class GLMAccessTokenManager:
    def __init__(self, config: AppConfig, logger: Logger) -> None:
        self.config = config
        self.logger = logger
        self._cached: AccessToken | None = None
        self._lock = threading.Lock()
        self._persist_lock = threading.Lock()

    def get_browser_headers(self) -> dict[str, str]:
        return {
            "Accept": "text/event-stream",
            "Accept-Encoding": "identity",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "App-Name": "chatglm",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "Origin": "https://chatglm.cn",
            "Pragma": "no-cache",
            "Priority": "u=1, i",
            "Sec-Ch-Ua": '"Microsoft Edge";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": self.config.glm_user_agent,
            "X-App-Fr": "browser_extension",
            "X-App-Platform": "pc",
            "X-App-Version": "0.0.1",
            "X-Device-Brand": "",
            "X-Device-Model": "",
            "X-Lang": "zh",
        }

    def read_json_response(self, response) -> dict[str, object]:
        raw_body = response.read()
        content_encoding = response.headers.get("Content-Encoding", "").lower()

        if content_encoding == "gzip":
            raw_body = gzip.decompress(raw_body)

        return json.loads(raw_body.decode("utf-8"))

    def get_access_token(self) -> str:
        with self._lock:
            if self._cached and time.time() < self._cached.expires_at - 60:
                return self._cached.access_token
            self._cached = self._refresh_access_token()
            return self._cached.access_token

    def _refresh_access_token(self) -> AccessToken:
        timestamp, nonce, sign = build_sign()
        request = urllib.request.Request(
            self.config.refresh_url,
            data=b"{}",
            method="POST",
            headers={
                **self.get_browser_headers(),
                "Authorization": f"Bearer {self.config.glm_refresh_token}",
                "X-Device-Id": uuid.uuid4().hex,
                "X-Nonce": nonce,
                "X-Request-Id": uuid.uuid4().hex,
                "X-Sign": sign,
                "X-Timestamp": timestamp,
            },
        )
        with urllib.request.urlopen(request, timeout=self.config.request_timeout) as response:
            payload = self.read_json_response(response)
        code = payload.get("code", payload.get("status"))
        result = payload.get("result") or {}
        access_token = result.get("access_token")
        refresh_token = result.get("refresh_token", self.config.glm_refresh_token)
        if response.status != 200 or code not in {0, None} or not access_token:
            raise RuntimeError(f"刷新 GLM token 失败: {payload}")
        if refresh_token != self.config.glm_refresh_token:
            self._persist_refresh_token(refresh_token)
            self.config.glm_refresh_token = refresh_token
            self.logger.info("GLM refresh_token 已自动刷新并写回 .env")
        return AccessToken(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=time.time() + ACCESS_TOKEN_EXPIRES_SECONDS - random.randint(10, 30),
        )

    def _persist_refresh_token(self, refresh_token: str) -> None:
        env_path = self.config.env_file_path
        if not env_path.exists():
            self.logger.warning(".env 文件不存在，无法自动写回新的 refresh_token")
            return

        with self._persist_lock:
            content = env_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            updated = False

            for index, line in enumerate(lines):
                if line.startswith("GLM_REFRESH_TOKEN="):
                    lines[index] = f"GLM_REFRESH_TOKEN={refresh_token}"
                    updated = True
                    break

            if not updated:
                if lines and lines[-1].strip():
                    lines.append("")
                lines.append(f"GLM_REFRESH_TOKEN={refresh_token}")

            new_content = "\n".join(lines) + "\n"
            env_path.write_text(new_content, encoding="utf-8")
