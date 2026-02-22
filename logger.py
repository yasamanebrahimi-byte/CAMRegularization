import logging
import sys
from pathlib import Path
from typing import Optional

# Global chosen log file path for this process (shared across modules)
_GLOBAL_LOG_PATH: Optional[Path] = None

def _ensure_log_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _attach_file_handler(logger: logging.Logger, path: Path, mode: str = "a") -> None:
    """Remove existing FileHandler(s) on logger and attach a new one for `path`."""
    # Remove existing FileHandler instances
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)

    _ensure_log_dir(path)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(path, mode=mode)
    fh.setLevel(logging.INFO)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)


"""
Set a global log file path and attach it to the root logger only.
Child loggers will propagate messages up to the root, preventing duplicates.
If console=False, removes console handlers from all existing loggers.
"""
def set_global_log_file(path: Path, mode: str = "a", console: bool = True) -> None:
    global _GLOBAL_LOG_PATH
    _GLOBAL_LOG_PATH = Path(path)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    _attach_file_handler(root, _GLOBAL_LOG_PATH, mode=mode)
    
    # If console output is disabled, remove console handlers from all loggers
    if not console:
        root_handlers = list(root.handlers)
        for h in root_handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                root.removeHandler(h)
        
        # Also remove console handlers from any existing named loggers
        for name, obj in list(logging.Logger.manager.loggerDict.items()):
            if isinstance(obj, logging.Logger):
                for h in list(obj.handlers):
                    if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                        obj.removeHandler(h)


def _ensure_console_handler(logger: logging.Logger) -> None:
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_formatter = logging.Formatter("%(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)

"""
Return a logger configured with a console handler and a shared file handler.
- If `log_file` is provided, it becomes the global log file for this process
    and will be attached to all existing loggers.
- If a global log file has already been set, it will be attached to the
    returned logger as well.
"""
def get_logger(name: str = "CAMRegularization", log_file: Optional[Path] = None, console: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True  # Ensure messages propagate to root logger
    
    if console:
        _ensure_console_handler(logger)

    # If caller provided a specific file, set it globally
    if log_file is not None:
        path = Path(log_file)
        set_global_log_file(path, mode="a", console=console)

    return logger

class SimpleLogger:
    def info(self, msg):
        print(msg)