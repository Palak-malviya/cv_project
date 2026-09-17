import logging
import time
from functools import wraps
import os

class InvalidImageError(Exception):
    """Raised when an image cannot be loaded or is invalid."""
    pass

class NoKeypointsFoundError(Exception):
    """Raised when SIFT fails to extract any keypoints."""
    pass

def setup_logger(name: str) -> logging.Logger:
    """
    Configures a logger that writes to both the console and a log file.
    The log file is placed in the 'output' directory.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        return logger
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    os.makedirs('output', exist_ok=True)
    fh = logging.FileHandler('output/forgery_detection.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def timeit(func):
    """Decorator to measure and log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger = logging.getLogger(func.__module__)
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper
