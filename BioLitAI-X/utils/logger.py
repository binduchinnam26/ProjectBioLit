"""Logging configuration for BioLitAI-X."""

import logging
import logging.handlers
import sys

import config


def setup_logging():
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler
    try:
        fh = logging.handlers.RotatingFileHandler(
            config.LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass  # Skip file logging if path is not writable
