"""Logging utilities for optimization template pipeline."""

import logging


def get_dq_warnings_logger() -> logging.Logger:
    """Get dq warnings logger."""
    return logging.getLogger("dq-warnings")


def config_logger_for_file_handler(
    logger: logging.Logger, logger_format: str, logger_level: int, file_name: str
) -> logging.FileHandler:
    """This config allows to save logs to file."""
    fh = logging.FileHandler(file_name)
    fh.setLevel(logger_level)
    formatter = logging.Formatter(logger_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return fh
