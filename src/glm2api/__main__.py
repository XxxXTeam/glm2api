from __future__ import annotations

import traceback

from .app import StartupError, create_application
from .config import ConfigError


def main() -> int:
    try:
        application = create_application()
        application.run()
        return 0
    except (ConfigError, StartupError) as exc:
        print(f"[glm2api] 启动失败: {exc}")
        return 2
    except KeyboardInterrupt:
        print("[glm2api] 已中断退出")
        return 130
    except Exception as exc:
        print(f"[glm2api] 未处理异常: {exc}")
        print(traceback.format_exc())
        return 1
