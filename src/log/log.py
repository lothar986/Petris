"""Logger configuration package."""

import logging
import os
import sys
from pathlib import Path


class LogFormatter(logging.Formatter):
    """Custom formatter to display colors"""
    
    _grey = "\x1b[38;20m"
    _yellow = "\x1b[33;20m"
    _red = "\x1b[31;20m"
    _bold_red = "\x1b[31;1m"
    _reset = "\x1b[0m"
    _format = "%(asctime)s.%(msecs)-3d %(levelname)-8s %(name)s:%(lineno)-10s \t %(message)s"
    
    _formatter_map = {
        logging.DEBUG: _grey + _format + _reset,
        logging.INFO: _grey + _format + _reset,
        logging.WARNING: _yellow + _format + _reset,
        logging.ERROR: _red + _format + _reset,
        logging.CRITICAL: _bold_red + _format + _reset
    }
    
    def format(self, record):
        log_fmt = self._formatter_map.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record=record)


def initialize_logger(log_path: Path) -> None:
    """
    Initializes and configures the logger for the entire application.

    Args:
        log_path (Path): Path to logs directory.
    """
    
    # Logging format
    formatter = LogFormatter()
    
    
    # Want the root logger to log from the DEBUG level for developers.
    logging.root.setLevel(logging.DEBUG)
    
    # STDOUT for the logger to log to the console
    stream_out_handler = logging.StreamHandler(sys.stdout)
    stream_out_handler.setLevel(logging.DEBUG)
    stream_out_handler.setFormatter(formatter)
    logging.root.addHandler(stream_out_handler)
    
    # Create logs/ directory if is does not already exist
    os.makedirs(name=log_path.parent, exist_ok=True)
    
    # For the logger to create and log to files.
    file_handler = logging.FileHandler(filename=log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logging.root.addHandler(file_handler)
