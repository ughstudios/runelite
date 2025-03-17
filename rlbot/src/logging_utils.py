import logging
import time
import inspect
from pathlib import Path
from typing import Optional, Tuple

from rich.console import Console
from rich.logging import RichHandler


def get_logger(
    debug: bool = False, 
    verbose: bool = False, 
    logger_name: Optional[str] = None
) -> logging.Logger:
    """
    Configure unified logging with both file output and rich-colored console formatting.
    
    Args:
        debug (bool): Whether to enable debug logging.
        verbose (bool): Whether to enable verbose output.
        logger_name (Optional[str]): The name for the logger. If None, it is inferred from the caller's module.
        
    Returns:
        Tuple[logging.Logger, Console]: The configured logger and rich console.
    """
    if logger_name is None:
        caller_frame = inspect.stack()[1]
        logger_name = caller_frame.frame.f_globals.get("__name__", "RLBot")
    
    log_level = logging.DEBUG if debug or verbose else logging.INFO
    log_dir = Path("./rlbot/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"rlbot_{timestamp}.log"

    console = Console()
    logger = logging.getLogger(logger_name)
    
    if not logger.handlers:
        logger.setLevel(log_level)
        rich_handler = RichHandler(rich_tracebacks=True, markup=True, console=console)
        rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(rich_handler)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)
        
        logger.info("Logging initialized at level %s", log_level)
        logger.info("Log file: %s", log_file)
    
    return logger
