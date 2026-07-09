from __future__ import annotations

import platform
import sys
import traceback

from .app import StartupError, create_application, create_async_application
from .config import ConfigError
from .logging_utils import get_logger, setup_logging


def main() -> int:
    # Windows event loop compatibility
    if platform.system() == "Windows":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    use_async = "--async" in sys.argv
    try:
        if use_async:
            application = create_async_application()
        else:
            application = create_application()
    except (ConfigError, StartupError) as exc:
        # Logging may not be set up yet — fall back to plain print for early errors
        print(f"[glm2api] 启动失败: {exc}")
        return 2
    except KeyboardInterrupt:
        print("[glm2api] 已中断退出")
        return 130
    except Exception as exc:
        print(f"[glm2api] 未处理异常: {exc}")
        print(traceback.format_exc())
        return 1

    logger = get_logger("glm2api.main")
    try:
        application.run()
        return 0
    except KeyboardInterrupt:
        logger.info("已中断退出")
        return 130
    except Exception as exc:
        logger.error("未处理异常: %s\n%s", exc, traceback.format_exc())
        return 1
