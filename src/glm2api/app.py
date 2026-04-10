from __future__ import annotations

import errno
import signal
import traceback

from .config import AppConfig, ConfigError, load_config
from .logging_utils import get_logger, setup_logging
from .server import GLM2APIServer
from .services.glm_client import GLMWebClient


class StartupError(RuntimeError):
    pass


class Application:
    def __init__(self, config: AppConfig) -> None:
        setup_logging(config.log_level)
        self.config = config
        self.logger = get_logger("glm2api.app")
        self.client = GLMWebClient(config=config, logger=get_logger("glm2api.glm"))
        try:
            self.server = GLM2APIServer(
                config=config,
                glm_client=self.client,
                logger=get_logger("glm2api.http"),
            )
        except OSError as exc:
            raise self._wrap_server_error(exc) from exc
        except Exception as exc:
            raise StartupError(f"初始化 HTTP 服务失败: {exc}") from exc
        self._stopping = False
        self._install_signal_handlers()

    def run(self) -> None:
        self.logger.info(
            "启动服务 host=%s port=%s prefix=%s concurrency=%s accounts=%s models=%s",
            self.config.host,
            self.config.port,
            self.config.api_prefix,
            self.config.glm_max_concurrency,
            len(self.config.glm_refresh_tokens),
            ",".join(self.config.exposed_models),
        )
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            self.logger.info("收到 Ctrl+C，正在优雅关闭服务...")
        except OSError as exc:
            self.logger.error("HTTP 服务运行时异常 error=%s", exc)
            raise StartupError(f"HTTP 服务运行失败: {exc}") from exc
        except Exception as exc:
            self.logger.error("服务异常退出 error=%s\n%s", exc, traceback.format_exc())
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self.logger.info("停止 HTTP 服务并释放监听端口...")
        try:
            self.server.shutdown()
        except Exception as exc:
            self.logger.warning("关闭 HTTP 服务时出现异常 error=%s", exc)
        self.logger.info("glm2api 已退出")

    def _install_signal_handlers(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(signum, self._handle_signal)
            except (ValueError, AttributeError):
                continue

    def _handle_signal(self, signum: int, frame) -> None:
        signal_name = signal.Signals(signum).name
        self.logger.info("收到退出信号 %s，准备关闭服务...", signal_name)
        raise KeyboardInterrupt

    def _wrap_server_error(self, exc: OSError) -> StartupError:
        if exc.errno in {errno.EADDRINUSE, 10048}:
            return StartupError(f"端口已被占用: {self.config.host}:{self.config.port}")
        if exc.errno in {errno.EACCES, 10013}:
            return StartupError(f"没有权限监听地址: {self.config.host}:{self.config.port}")
        if exc.errno in {errno.EADDRNOTAVAIL, 10049}:
            return StartupError(f"监听地址不可用: {self.config.host}")
        return StartupError(f"启动 HTTP 服务失败: {exc}")


def create_application() -> Application:
    try:
        config = load_config()
    except ConfigError:
        raise
    except Exception as exc:
        raise StartupError(f"读取配置失败: {exc}") from exc
    return Application(config)
