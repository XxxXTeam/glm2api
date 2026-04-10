from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ASSISTANT_ID = "65940acff94777010aa6b796"
DEFAULT_IMAGE_ASSISTANT_ID = "65a232c082ff90a2ad2f15e2"
DEFAULT_IMAGE_MODEL_NAME = "glm-image-1"
DEFAULT_GLM_BASE_URL = "https://chatglm.cn/chatglm"
GUEST_REFRESH_TOKEN_MARKER = "__glm_guest__"
BUILTIN_EXPOSED_MODELS = (
    "cogView-4-250304",
    "glm-5.1",
    "glm-5v-turbo",
    "glm-5-turbo",
    "glm-5",
    "glm-4.7-flash",
    "glm-4.7",
    "glm-4.6v-flash",
    "glm-4.6",
    "glm-4.5",
    "glm-4.1v-thinking-flashx",
    "glm-4",
    "glm-4-flash",
    "glm-4-air",
    "glm-4v",
    "glm-4-flashx-250414",
    "glm-4-flash-250414",
    "glm-zero-preview",
    "glm-deep-research",
    DEFAULT_IMAGE_MODEL_NAME,
)
BUILTIN_MODEL_ALIASES = {name: name for name in BUILTIN_EXPOSED_MODELS}


class ConfigError(ValueError):
    pass


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ConfigError(f"配置文件不是有效的 UTF-8 编码: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"读取配置文件失败: {path} error={exc}") from exc

    for raw_line in lines:
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
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"整数配置值无效: {value}") from exc


def parse_float(value: str | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"浮点配置值无效: {value}") from exc


def parse_list(value: str | None, default: tuple[str, ...] = ()) -> list[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def load_refresh_tokens(token_file_path: Path) -> list[str]:
    if not token_file_path.exists():
        return []
    tokens: list[str] = []
    try:
        lines = token_file_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ConfigError(f"token 文件不是有效的 UTF-8 编码: {token_file_path}") from exc
    except OSError as exc:
        raise ConfigError(f"读取 token 文件失败: {token_file_path} error={exc}") from exc
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens.append(line)
    return tokens


def is_guest_token_value(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized in {"guest", "guest_ck", "guest-ck", "visitor", "tourist", "游客", GUEST_REFRESH_TOKEN_MARKER}


@dataclass(slots=True)
class AppConfig:
    env_file_path: Path
    env_file_created: bool
    token_file_path: Path
    host: str
    port: int
    api_prefix: str
    log_level: str
    debug_dump_all: bool
    request_timeout: int
    glm_base_url: str
    glm_use_guest_refresh_token: bool
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
    glm_guest_max_retries: int
    exposed_models: list[str]
    model_aliases: dict[str, str]
    server_api_keys: list[str]
    cors_allow_origin: str

    @property
    def refresh_url(self) -> str:
        return f"{self.glm_base_url}/user-api/user/refresh"

    @property
    def guest_refresh_url(self) -> str:
        return f"{self.glm_base_url}/user-api/guest/access"

    @property
    def chat_stream_url(self) -> str:
        return f"{self.glm_base_url}/backend-api/assistant/stream"

    @property
    def delete_conversation_url(self) -> str:
        return f"{self.glm_base_url}/backend-api/assistant/conversation/delete"


def ensure_env_file(env_path: Path) -> bool:
    if env_path.exists():
        return False

    example_candidates = [
        env_path.with_name(".env.example"),
        env_path.parent / ".env.example",
    ]
    example_path = next((candidate for candidate in example_candidates if candidate.exists()), None)
    if example_path is None:
        return False

    try:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(example_path, env_path)
    except OSError as exc:
        raise ConfigError(f"自动创建配置文件失败: source={example_path} target={env_path} error={exc}") from exc
    return True


def load_config(env_file: str = ".env") -> AppConfig:
    env_path = Path(env_file)
    env_file_created = ensure_env_file(env_path)
    file_values = parse_dotenv(env_path)
    values = {**file_values, **os.environ}
    token_file_path = Path(values.get("GLM_TOKEN_FILE", "token.txt"))
    if not token_file_path.is_absolute():
        token_file_path = (env_path.parent / token_file_path).resolve()
    refresh_tokens = load_refresh_tokens(token_file_path)
    single_refresh_token = values.get("GLM_REFRESH_TOKEN", "").strip()
    explicit_guest_mode = parse_bool(values.get("GLM_USE_GUEST_REFRESH_TOKEN"), False) or is_guest_token_value(single_refresh_token)
    if explicit_guest_mode:
        refresh_tokens = [GUEST_REFRESH_TOKEN_MARKER]
        single_refresh_token = GUEST_REFRESH_TOKEN_MARKER
    elif not refresh_tokens and single_refresh_token:
        refresh_tokens = [single_refresh_token]
    elif not refresh_tokens:
        refresh_tokens = [GUEST_REFRESH_TOKEN_MARKER]
        single_refresh_token = GUEST_REFRESH_TOKEN_MARKER
        explicit_guest_mode = True
    host = values.get("HOST", "127.0.0.1").strip() or "127.0.0.1"
    api_prefix = values.get("API_PREFIX", "/v1").strip()
    if not api_prefix:
        api_prefix = "/v1"
    if not api_prefix.startswith("/"):
        api_prefix = f"/{api_prefix}"
    api_prefix = api_prefix.rstrip("/") or "/v1"
    log_level = values.get("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    debug_dump_all = parse_bool(values.get("DEBUG_DUMP_ALL"), False)
    if debug_dump_all:
        log_level = "DEBUG"
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        log_level = "INFO"
    image_model_name = DEFAULT_IMAGE_MODEL_NAME
    exposed_models = list(BUILTIN_EXPOSED_MODELS)
    model_aliases = dict(BUILTIN_MODEL_ALIASES)

    config = AppConfig(
        env_file_path=env_path,
        env_file_created=env_file_created,
        token_file_path=token_file_path,
        host=host,
        port=parse_int(values.get("PORT"), 8000),
        api_prefix=api_prefix,
        log_level=log_level,
        debug_dump_all=debug_dump_all,
        request_timeout=parse_int(values.get("REQUEST_TIMEOUT_SECONDS"), 120),
        glm_base_url=values.get("GLM_BASE_URL", DEFAULT_GLM_BASE_URL).rstrip("/"),
        glm_use_guest_refresh_token=explicit_guest_mode,
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
        glm_guest_max_retries=max(0, parse_int(values.get("GLM_GUEST_MAX_RETRIES"), 3)),
        exposed_models=exposed_models, # type: ignore
        model_aliases=model_aliases,
        server_api_keys=parse_list(values.get("SERVER_API_KEYS")),
        cors_allow_origin=values.get("CORS_ALLOW_ORIGIN", "*").strip() or "*",
    )
    if not (1 <= config.port <= 65535):
        raise ConfigError(f"端口配置超出范围: PORT={config.port}")
    if config.request_timeout <= 0:
        raise ConfigError(f"请求超时必须大于 0: REQUEST_TIMEOUT_SECONDS={config.request_timeout}")
    if config.glm_queue_wait_timeout <= 0:
        raise ConfigError(f"队列等待时间必须大于 0: GLM_QUEUE_WAIT_TIMEOUT_SECONDS={config.glm_queue_wait_timeout}")
    if config.glm_busy_retry_interval < 0:
        raise ConfigError(f"忙碌重试间隔不能小于 0: GLM_BUSY_RETRY_INTERVAL_SECONDS={config.glm_busy_retry_interval}")
    if not config.glm_base_url.startswith(("http://", "https://")):
        raise ConfigError(f"GLM_BASE_URL 必须以 http:// 或 https:// 开头: {config.glm_base_url}")
    return config
