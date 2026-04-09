from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ASSISTANT_ID = "65940acff94777010aa6b796"
DEFAULT_IMAGE_ASSISTANT_ID = "65a232c082ff90a2ad2f15e2"
DEFAULT_IMAGE_MODEL_NAME = "glm-image-1"
DEFAULT_GLM_BASE_URL = "https://chatglm.cn/chatglm"
DEFAULT_EXPOSED_MODELS = (
    "glm-5.1",
    "glm-5",
    "glm-4",
    "glm-4-flash",
    "glm-4-air",
    "glm-4v",
    "glm-zero-preview",
    "glm-deep-research",
    DEFAULT_IMAGE_MODEL_NAME,
)


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        value = raw_value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def parse_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def parse_list(value: str | None, default: tuple[str, ...] = ()) -> list[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_aliases(value: str | None) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if not value:
        return aliases
    for pair in value.split(","):
        item = pair.strip()
        if not item:
            continue
        if "=" not in item:
            aliases[item] = item
            continue
        public_name, upstream_name = item.split("=", 1)
        aliases[public_name.strip()] = upstream_name.strip()
    return aliases


@dataclass(slots=True)
class AppConfig:
    env_file_path: Path
    host: str
    port: int
    api_prefix: str
    log_level: str
    request_timeout: int
    glm_base_url: str
    glm_refresh_token: str
    glm_assistant_id: str
    glm_image_assistant_id: str
    glm_image_model_name: str
    glm_user_agent: str
    glm_delete_conversation: bool
    glm_queue_wait_timeout: int
    glm_busy_max_retries: int
    glm_busy_retry_interval: float
    exposed_models: list[str]
    model_aliases: dict[str, str]
    server_api_keys: list[str]
    cors_allow_origin: str

    @property
    def refresh_url(self) -> str:
        return f"{self.glm_base_url}/user-api/user/refresh"

    @property
    def chat_stream_url(self) -> str:
        return f"{self.glm_base_url}/backend-api/assistant/stream"

    @property
    def delete_conversation_url(self) -> str:
        return f"{self.glm_base_url}/backend-api/assistant/conversation/delete"


def load_config(env_file: str = ".env") -> AppConfig:
    file_values = parse_dotenv(Path(env_file))
    values = {**file_values, **os.environ}
    image_model_name = values.get("GLM_IMAGE_MODEL_NAME", DEFAULT_IMAGE_MODEL_NAME).strip() or DEFAULT_IMAGE_MODEL_NAME
    model_aliases = parse_aliases(values.get("GLM_MODEL_ALIASES"))
    exposed_models = parse_list(values.get("EXPOSED_MODELS"), DEFAULT_EXPOSED_MODELS)
    if image_model_name not in exposed_models:
        exposed_models.append(image_model_name)
    if not model_aliases:
        model_aliases = {name: name for name in exposed_models}
    elif image_model_name not in model_aliases:
        model_aliases[image_model_name] = image_model_name

    config = AppConfig(
        env_file_path=Path(env_file),
        host=values.get("HOST", "127.0.0.1"),
        port=parse_int(values.get("PORT"), 8000),
        api_prefix=values.get("API_PREFIX", "/v1").rstrip("/") or "/v1",
        log_level=values.get("LOG_LEVEL", "INFO").upper(),
        request_timeout=parse_int(values.get("REQUEST_TIMEOUT_SECONDS"), 120),
        glm_base_url=values.get("GLM_BASE_URL", DEFAULT_GLM_BASE_URL).rstrip("/"),
        glm_refresh_token=values.get("GLM_REFRESH_TOKEN", "").strip(),
        glm_assistant_id=values.get("GLM_ASSISTANT_ID", DEFAULT_ASSISTANT_ID).strip(),
        glm_image_assistant_id=values.get("GLM_IMAGE_ASSISTANT_ID", DEFAULT_IMAGE_ASSISTANT_ID).strip(),
        glm_image_model_name=image_model_name,
        glm_user_agent=values.get(
            "GLM_USER_AGENT",
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36 Edg/143.0.0.0"
            ),
        ).strip(),
        glm_delete_conversation=parse_bool(values.get("GLM_DELETE_CONVERSATION"), True),
        glm_queue_wait_timeout=parse_int(values.get("GLM_QUEUE_WAIT_TIMEOUT_SECONDS"), 600),
        glm_busy_max_retries=parse_int(values.get("GLM_BUSY_MAX_RETRIES"), 30),
        glm_busy_retry_interval=parse_float(values.get("GLM_BUSY_RETRY_INTERVAL_SECONDS"), 2.0),
        exposed_models=exposed_models,
        model_aliases=model_aliases,
        server_api_keys=parse_list(values.get("SERVER_API_KEYS")),
        cors_allow_origin=values.get("CORS_ALLOW_ORIGIN", "*").strip() or "*",
    )
    if not config.glm_refresh_token:
        raise ValueError("缺少必填配置：GLM_REFRESH_TOKEN，请先复制 .env.example 为 .env 后填写你的 refresh_token 值,使用F12开发者工具在 Local Storage Cookie 中获取。")
    return config
