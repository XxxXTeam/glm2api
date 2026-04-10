from __future__ import annotations

import logging
import sys


RESET = "\033[0m"
COLORS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
}


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        color = COLORS.get(original_levelname, "")
        if color:
            record.levelname = f"{color}{original_levelname}{RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColorFormatter(
            fmt="[%(asctime)s][%(levelname)s]%(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(resolved_level)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
