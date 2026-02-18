import logging
import sys
from pathlib import Path


def get_logger(name: str = "CAMRegularization", log_file: str = "log.txt") -> logging.Logger:
    """
    Configure and return a logger that writes to both console and log file.
    
    Args:
        name: Name of the logger (typically __name__ of the module)
        log_file: Path to the log file (default: log.txt in current directory)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if the logger doesn't have handlers yet (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter(
            '%(message)s'
        )
        
        # File handler
        log_path = Path(log_file)
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger
