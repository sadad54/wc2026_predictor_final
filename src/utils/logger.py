"""
src/utils/logger.py

Logging configuration for the World Cup Prediction Engine.

Uses Loguru instead of Python's built-in logging because it:
  - Requires zero boilerplate
  - Has beautiful colour-coded console output by default
  - Handles log rotation and compression automatically
  - Gives you file + line numbers in every message for free

Usage in any other module:
    from src.utils.logger import logger
    logger.info("Training model...")
    logger.warning("Missing data for team: Spain")
    logger.error("File not found: data/raw/results.csv")
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logger(
    log_dir: str = "logs",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Configure the global Loguru logger with two handlers:
      - Console (stdout): colourised, human-readable
      - File: rotating, compressed, machine-parseable

    This function is called once at startup via initialize_project().
    After that, every module just imports `logger` and uses it directly.

    Args:
        log_dir:   Directory where log files are written.
        level:     Minimum log level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        rotation:  When to start a new log file ('10 MB', '1 day', etc.).
        retention: How long to keep old log files ('1 week', '30 days', etc.).
    """
    # Ensure the logs directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Remove Loguru's default stderr handler so we control the output format
    logger.remove()

    # ── Console handler ──────────────────────────────────────────────────────
    # {level: <8} pads the level name to 8 characters so columns align neatly
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "{message}"
    )
    logger.add(
        sys.stdout,
        level=level,
        format=console_format,
        colorize=True,
    )

    # ── File handler ─────────────────────────────────────────────────────────
    # Plain format (no colour codes) because log files are parsed by machines
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} — {message}"
    )
    logger.add(
        Path(log_dir) / "worldcup_predictor.log",
        level=level,
        format=file_format,
        rotation=rotation,       # Start a new file when this size is reached
        retention=retention,     # Delete files older than this
        compression="zip",       # Compress old log files to save space
        encoding="utf-8",
    )

    logger.info("Logger configured — console + file handlers active")


# Re-export the configured logger so every module can do:
#   from src.utils.logger import logger
__all__ = ["logger", "setup_logger"]