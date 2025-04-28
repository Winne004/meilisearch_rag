import logging
import sys


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with a specific name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if the logger already has handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
