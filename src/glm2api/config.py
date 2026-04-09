from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ASSISTANT_ID = "65940acff94777010aa6b796"
DEFAULT_IMAGE_ASSISTANT_ID = "65a232c082ff90a2ad2f15e2"
DEFAULT_IMAGE_MODEL_NAME = "glm-image-1"
DEFAULT_GLM_BASE_URL = "https://chatglm.cn/chatglm"
BUILTIN_EXPOSED_MODELS = (
    "glm-5.1",
    "glm-5-turbo",
    "glm-5",
    "glm-4.7",
    "glm-4.6",
    "glm-4.5",
    "glm-4",
    "glm-4-flash",
    "glm-4-air",
    "glm-4v",
    "glm-zero-preview",
    "glm-deep-research",
    DEFAULT_IMAGE_MODEL_NAME,
)
BUILTIN_MODEL_ALIASES = {name: name for name in BUILTIN_EXPOSED_MODELS}


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


def load_refresh_tokens(token_file_path: Path) -> list[str]:
    if not token_file_path.exists():
        return []
    tokens: list[str] = []
    for raw_line in token_file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.append(line)
    return tokens


@dataclass(slots=True)
class AppConfig:
    env_file_path: Path
    token_file_path: Path
    host: str
    port: int
    api_prefix: str
    log_level: str
    request_timeout: int
    glm_base_url: str
    glm_refresh_token: str
    glm_refresh_tokens: list[str]
    glm_assistant_id: str
    glm_image_assistant_id: str
    glm_image_model_name: str
    glm_user_agent: str
    glm_delete_conversation: bool
    glm_max_concurrency: int
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
    env_path = Path(env_file)
    file_values = parse_dotenv(env_path)
    values = {**file_values, **os.environ}
    token_file_path = Path(values.get("GLM_TOKEN_FILE", "token.txt"))
    if not token_file_path.is_absolute():
        token_file_path = (env_path.parent / token_file_path).resolve()
    refresh_tokens = load_refresh_tokens(token_file_path)
    single_refresh_token = values.get("GLM_REFRESH_TOKEN", "").strip()
    if not refresh_tokens and single_refresh_token:
        refresh_tokens = [single_refresh_token]
    image_model_name = DEFAULT_IMAGE_MODEL_NAME
    exposed_models = list(BUILTIN_EXPOSED_MODELS)
    model_aliases = dict(BUILTIN_MODEL_ALIASES)

    config = AppConfig(
        env_file_path=env_path,
        token_file_path=token_file_path,
        host=values.get("HOST", "127.0.0.1"),
        port=parse_int(values.get("PORT"), 8000),
        api_prefix=values.get("API_PREFIX", "/v1").rstrip("/") or "/v1",
        log_level=values.get("LOG_LEVEL", "INFO").upper(),
        request_timeout=parse_int(values.get("REQUEST_TIMEOUT_SECONDS"), 120),
        glm_base_url=values.get("GLM_BASE_URL", DEFAULT_GLM_BASE_URL).rstrip("/"),
        glm_refresh_token=single_refresh_token,
        glm_refresh_tokens=refresh_tokens,
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
        glm_max_concurrency=max(1, parse_int(values.get("GLM_MAX_CONCURRENCY"), 3)),
        glm_queue_wait_timeout=parse_int(values.get("GLM_QUEUE_WAIT_TIMEOUT_SECONDS"), 600),
        glm_busy_max_retries=parse_int(values.get("GLM_BUSY_MAX_RETRIES"), 30),
        glm_busy_retry_interval=parse_float(values.get("GLM_BUSY_RETRY_INTERVAL_SECONDS"), 2.0),
        exposed_models=exposed_models, # type: ignore
        model_aliases=model_aliases,
        server_api_keys=parse_list(values.get("SERVER_API_KEYS")),
        cors_allow_origin=values.get("CORS_ALLOW_ORIGIN", "*").strip() or "*",
    )
    if not config.glm_refresh_tokens:
        raise ValueError(
            "缺少可用账号 token：请在 token.txt 中每行填写一个 refresh_token，"
            "或在 .env 中配置 GLM_REFRESH_TOKEN 作为兜底。"
        )
    return config
