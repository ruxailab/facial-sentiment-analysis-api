"""
Centralized logging configuration for the RuxaiLab Facial Sentiment API.

Provides a single `get_logger()` factory so every module uses a consistent
format, level, and handler setup — regardless of whether the app is run
locally or deployed on Cloud Run.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import os
import sys

_configured = False

def _configure_root_logger() -> None:
    """
    Configure the root logger once at import time.
    Level is read from the LOG_LEVEL environment variable (default: INFO).
    Outputs structured plaintext to stdout so Cloud Run can ingest it.
    """
    global _configured
    if _configured:
        return

    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if called multiple times
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with guaranteed root configuration applied.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured logging.Logger instance.
    """
    _configure_root_logger()
    return logging.getLogger(name)
