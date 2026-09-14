"""Logging setup: one place that decides where log lines go."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import Config

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to the console and to a rotating file."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(Config.LOG_LEVEL)
    formatter = logging.Formatter(FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_dir = Path(Config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_dir / Config.LOG_FILE,
        maxBytes=Config.LOG_MAX_BYTES,
        backupCount=Config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
