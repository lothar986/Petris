"""Logger configuration package."""

import logging
import os
import sys
from pathlib import Path


def initialize_logger(log_path: Path) -> None:
    """
    Initializes and configures the logger for the entire application.

    Args:
        log_path (Path): Path to logs directory.
    """
    
    # Logging format
    formatter = logging.Formatter("%(asctime)s.%(msecs)-3d %(levelname)-8s %(name)s:%(lineno)-10s"
                                  "\t %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    
    
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
