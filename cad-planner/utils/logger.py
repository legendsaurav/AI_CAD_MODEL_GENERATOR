import logging
import os
from typing import Optional

# Resolve log path relative to the cad-planner root, not the caller's CWD
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO):
    """
    Standardized hierarchical experiment tracking logger.
    """
    if log_file is None:
        log_file = os.path.join(_ROOT_DIR, "logs", "planner.log")
        
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger
