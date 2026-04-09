from __future__ import annotations

from .app import create_application


def main() -> int:
    application = create_application()
    application.run()
    return 0
