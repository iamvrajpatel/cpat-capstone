"""
CPAT — Structured Logging Infrastructure
==========================================
Configures ``structlog`` with ``rich`` for development and JSON output
for production. All CPAT components use Python's standard ``logging``
module; structlog provides the structured formatting layer.

Usage::

    from cpat.infrastructure.logging import setup_logging
    setup_logging(log_level="INFO", log_format="console")
    # Then use standard logging in any module:
    logger = logging.getLogger(__name__)
    logger.info("System started", extra={"capital": 1_000_000})
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "console",
    log_dir: str | None = None,
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> None:
    """Configure the CPAT logging stack.

    Args:
        log_level: Minimum log level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
        log_format: ``"console"`` for human-readable output, ``"json"`` for
                    structured JSON (production).
        log_dir: Directory for rotating log files. If ``None``, file logging
                 is disabled.
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of rotated log files to retain.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    # ── Console handler ────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    if log_format == "console":
        formatter = _ColorFormatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = _JSONFormatter()

    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # ── File handler (rotating) ────────────────────────────────────────────────
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / "cpat.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(_JSONFormatter())
        handlers.append(file_handler)

    # ── Root logger configuration ──────────────────────────────────────────────
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate logs
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy_lib in ("yfinance", "urllib3", "requests", "peewee", "appdirs"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


class _ColorFormatter(logging.Formatter):
    """Console formatter with ANSI colour codes for log levels."""

    _COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self._RESET}"
        return super().format(record)


class _JSONFormatter(logging.Formatter):
    """Minimal JSON log formatter for structured production logging."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        log_entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)
