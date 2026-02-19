import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


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
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)


def set_global_log_file(path: Path, mode: str = "a") -> None:
    """Set a global log file path and attach it to all existing loggers.

    Subsequent calls to get_logger that provide no explicit log_file will
    also attach handlers pointing to this path.
    """
    global _GLOBAL_LOG_PATH
    _GLOBAL_LOG_PATH = Path(path)
    # Attach file handler to the root logger only (no console handler on root)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    _attach_file_handler(root, _GLOBAL_LOG_PATH, mode=mode)

    # Also attach to all existing named loggers and ensure they don't propagate unnecessarily
    for name, obj in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(obj, logging.Logger):
            _attach_file_handler(obj, _GLOBAL_LOG_PATH, mode=mode)


def _ensure_console_handler(logger: logging.Logger) -> None:
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_formatter = logging.Formatter("%(message)s")
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)


def get_logger(name: str = "CAMRegularization", log_file: Optional[Path] = None) -> logging.Logger:
    """Return a logger configured with a console handler and (optionally)
    a shared file handler.

    - If `log_file` is provided, it becomes the global log file for this process
      and will be attached to all existing loggers.
    - If a global log file has already been set, it will be attached to the
      returned logger as well.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Ensure console handler exists
    _ensure_console_handler(logger)

    # If caller provided a specific file, set it globally and attach handlers
    if log_file is not None:
        path = Path(log_file)
        set_global_log_file(path, mode="a")
    elif _GLOBAL_LOG_PATH is not None:
        # Attach the global file handler to this logger (if not already)
        _attach_file_handler(logger, _GLOBAL_LOG_PATH, mode="a")

    return logger
