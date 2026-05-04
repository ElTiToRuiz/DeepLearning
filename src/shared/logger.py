import logging
import os
from datetime import datetime

LOGGER_NAME = "dl"


def setup_logger(results_dir: str, name: str = LOGGER_NAME) -> logging.Logger:
    """
    Configure the shared logger with a console handler and a timestamped file
    handler inside `results_dir`. Call this once at the start of each activity.
    Subsequent calls are a no-op so modules can safely import `logger`.
    """
    log = logging.getLogger(name)
    if log.handlers:
        return log

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(results_dir, f"run_{timestamp}.log")

    log.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    log.addHandler(console_handler)
    log.addHandler(file_handler)

    return log


# Module-level handle so modules can `from src.shared.logger import logger`.
# Stays unconfigured until the activity calls setup_logger().
logger = logging.getLogger(LOGGER_NAME)
