import logging
import sys
import io
from pathlib import Path
from typing import Optional


def _ensure_log_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _attach_file_handler(logger: logging.Logger, path: Path, mode: str = "a") -> None:
    """Remove existing FileHandler(s) on logger and attach a new one for the path."""
    # Remove existing FileHandler instances
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)

    _ensure_log_dir(path)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(path, mode=mode, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)


def _utf8_stdout_stream():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        return sys.stdout
    except Exception:
        pass

    try:
        return io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        return sys.stdout


def _ensure_console_handler(logger: logging.Logger) -> None:
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        console_formatter = logging.Formatter("%(message)s")
        ch = logging.StreamHandler(_utf8_stdout_stream())
        ch.setLevel(logging.INFO)
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)


def _remove_console_handlers(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            logger.removeHandler(h)


"""
Return a logger configured with optional console and file handlers.
Each logger instance is configured independently (no global shared log file).
"""
def get_logger(name: str = "CAMRegularization", log_file: Optional[Path] = None, console: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if console:
        _ensure_console_handler(logger)
    else:
        _remove_console_handlers(logger)

    if log_file is not None:
        path = Path(log_file)
        _attach_file_handler(logger, path, mode="a")

    return logger


class SimpleLogger:
    def _format(self, msg, *args):
        if args:
            return str(msg) % args
        return str(msg)

    def info(self, msg, *args):
        print(self._format(msg, *args))

    def warning(self, msg, *args):
        print(self._format(msg, *args))

    def error(self, msg, *args):
        print(self._format(msg, *args))
