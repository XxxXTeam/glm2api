from __future__ import annotations

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

    def run(self) -> None:
        self.logger.info(
            "启动服务 host=%s port=%s prefix=%s models=%s",
            self.config.host,
            self.config.port,
            self.config.api_prefix,
            ",".join(self.config.exposed_models),
        )
        self.server.serve_forever()


def create_application() -> Application:
    config = load_config()
    return Application(config)
