from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


# ── ANSI colour palette ──────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_C_FG = {
    "grey": "\033[38;5;245m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "white": "\033[37m",
    "bright_cyan": "\033[96m",
    "bright_green": "\033[92m",
    "bright_yellow": "\033[93m",
    "bright_red": "\033[91m",
    "bright_magenta": "\033[95m",
}

_C_BG = {
    "cyan": "\033[46m",
    "green": "\033[42m",
    "yellow": "\033[43m",
    "red": "\033[41m",
    "magenta": "\033[45m",
    "grey": "\033[48;5;240m",
}

# ── Level styling ────────────────────────────────────────────────────────────
# Use ASCII-safe icons on Windows to avoid codec errors with legacy code pages.
_IS_WINDOWS = sys.platform.startswith("win")

_LEVEL_STYLES: dict[str, dict[str, str]] = {
    "DEBUG": {
        "icon": "*" if _IS_WINDOWS else "◆",
        "fg": _C_FG["bright_cyan"],
        "bg": _C_BG["cyan"],
        "name_fg": _C_FG["cyan"],
    },
    "INFO": {
        "icon": ">" if _IS_WINDOWS else "●",
        "fg": _C_FG["bright_green"],
        "bg": _C_BG["green"],
        "name_fg": _C_FG["green"],
    },
    "WARNING": {
        "icon": "!" if _IS_WINDOWS else "▲",
        "fg": _C_FG["bright_yellow"],
        "bg": _C_BG["yellow"],
        "name_fg": _C_FG["yellow"],
    },
    "ERROR": {
        "icon": "x" if _IS_WINDOWS else "■",
        "fg": _C_FG["bright_red"],
        "bg": _C_BG["red"],
        "name_fg": _C_FG["red"],
    },
    "CRITICAL": {
        "icon": "X" if _IS_WINDOWS else "◈",
        "fg": _C_FG["bright_magenta"],
        "bg": _C_BG["magenta"],
        "name_fg": _C_FG["magenta"],
    },
}

_DATE_FMT = "%H:%M:%S"


class _TUIFormatter(logging.Formatter):
    """Terminal-UI inspired formatter with icons, colours and aligned columns."""

    def __init__(self, use_colour: bool = True) -> None:
        super().__init__()
        self.use_colour = use_colour
        # Name column width — adjusts lazily up to a cap
        self._name_width = 16
        self._name_max = 22

    def _colorize(self, text: str, codes: str) -> str:
        if not self.use_colour:
            return text
        return f"{codes}{text}{_RESET}"

    def _pad_name(self, name: str) -> str:
        # Shorten common prefixes to keep output compact
        name = name.replace("glm2api.", "")
        if len(name) > self._name_width:
            self._name_width = min(len(name), self._name_max)
        return name.ljust(self._name_width)

    def format(self, record: logging.LogRecord) -> str:
        style = _LEVEL_STYLES.get(record.levelname, _LEVEL_STYLES["INFO"])
        time_str = self.formatTime(record, _DATE_FMT)

        # colourised components
        time_part = self._colorize(time_str, _C_FG["grey"] + _DIM)
        icon_part = self._colorize(f" {style['icon']} ", style["fg"] + _BOLD)
        level_badge = self._colorize(
            f" {record.levelname:<7} ",
            _C_FG["white"] + style["bg"] + _BOLD,
        )
        name_part = self._colorize(self._pad_name(record.name), style["name_fg"] + _DIM)

        # message — keep newlines but indent continuations
        message = record.getMessage()
        lines = message.splitlines()
        indent = " " * (len(time_str) + 1 + 3 + 1 + 9 + 1 + self._name_width + 3)
        formatted_lines: list[str] = []
        for idx, line in enumerate(lines):
            if idx == 0:
                formatted_lines.append(
                    f"{time_part} {icon_part} {level_badge} │ {name_part} │ {line}"
                )
            else:
                formatted_lines.append(f"{indent}{line}")

        return "\n".join(formatted_lines)


class _PlainFormatter(logging.Formatter):
    """Plain-text formatter for file logs (no colour, no icons)."""

    def __init__(self) -> None:
        super().__init__()
        self._name_width = 16
        self._name_max = 22

    def _pad_name(self, name: str) -> str:
        name = name.replace("glm2api.", "")
        if len(name) > self._name_width:
            self._name_width = min(len(name), self._name_max)
        return name.ljust(self._name_width)

    def format(self, record: logging.LogRecord) -> str:
        time_str = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        icon = _LEVEL_STYLES.get(record.levelname, _LEVEL_STYLES["INFO"])["icon"]
        message = record.getMessage()
        lines = message.splitlines()
        indent = " " * (len(time_str) + 1 + 3 + 1 + 9 + 1 + self._name_width + 3)
        formatted: list[str] = []
        for idx, line in enumerate(lines):
            if idx == 0:
                formatted.append(
                    f"{time_str} {icon} {record.levelname:<7} │ "
                    f"{self._pad_name(record.name)} │ {line}"
                )
            else:
                formatted.append(f"{indent}{line}")
        return "\n".join(formatted)


def _should_use_colour() -> bool:
    """Heuristic: use colour when stdout is a TTY."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def setup_logging(level: str) -> None:
    # Attempt to force UTF-8 on Windows consoles so Unicode icons survive emit.
    if _IS_WINDOWS:
        import io
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

    root = logging.getLogger()
    root.handlers.clear()
    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(resolved_level)

    # ── Console handler (TUI style) ──────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_TUIFormatter(use_colour=_should_use_colour()))
    root.addHandler(console)

    # ── File handler (plain text, only when DEBUG) ───────────────────────────
    if resolved_level <= logging.DEBUG:
        log_dir = Path(os.environ.get("GLM2API_LOG_DIR", "log"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "glm2api_debug.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(_PlainFormatter())
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def serialize_for_debug(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, str):
        return value
    try:
        import orjson

        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2).decode("utf-8")
    except Exception:
        return repr(value)


def debug_dump(logger: logging.Logger, enabled: bool, title: str, value: Any) -> None:
    if not enabled:
        return
    logger.debug("%s\n%s", title, serialize_for_debug(value))
