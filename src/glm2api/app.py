from __future__ import annotations

import signal

from .config import AppConfig, load_config
from .logging_utils import get_logger, setup_logging
from .server import GLM2APIServer
from .services.glm_client import GLMWebClient


class Application:
    def __init__(self, config: AppConfig) -> None:
        setup_logging(config.log_level)
        self.config = config
        self.logger = get_logger("glm2api.app")
        self.client = GLMWebClient(config=config, logger=get_logger("glm2api.glm"))
        self.server = GLM2APIServer(
            config=config,
            glm_client=self.client,
            logger=get_logger("glm2api.http"),
        )
        self._stopping = False
        self._install_signal_handlers()

    def run(self) -> None:
        self.logger.info(
            "启动服务 host=%s port=%s prefix=%s models=%s",
            self.config.host,
            self.config.port,
            self.config.api_prefix,
            ",".join(self.config.exposed_models),
        )
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            self.logger.info("收到 Ctrl+C，正在优雅关闭服务...")
        finally:
            self.stop()

    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        self.logger.info("停止 HTTP 服务并释放监听端口...")
        self.server.shutdown()
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


def create_application() -> Application:
    config = load_config()
    return Application(config)
