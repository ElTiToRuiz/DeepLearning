import logging
import os
from datetime import datetime
from src.config.config import RESULTS_DIR
 
 
def get_logger(name: str = "insurance_nn") -> logging.Logger:
    """
    Logger that writes to console (INFO+) and to a file (DEBUG+).
    The file includes a timestamp to avoid overwriting previous executions.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(RESULTS_DIR, f"run_{timestamp}.log")
 
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
 
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
 
    # CONSOLE — only INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
 
    # FILE — everything, including DEBUG
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
 
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
 
    return logger
 
 
# GLOBAL INSTANCE — imported directly with: from src.logger.logger import logger
logger = get_logger()