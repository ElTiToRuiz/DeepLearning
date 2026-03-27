import logging
import os
from datetime import datetime
from src.config.config import RESULTS_DIR
 
 
def get_logger(name: str = "insurance_nn") -> logging.Logger:
    """
    Sets up a logger that talks to the console (vocal) and a file (quiet listener).
    The file gets a timestamp so we don't overwrite our history.
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
 
    # Console picks up INFO and louder
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
 
    # File catches everything, even the DEBUG whispers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
 
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
 
    return logger
 
 
# Shared instance so we can just import 'logger' anywhere
logger = get_logger()