# config/logging_config.py
import logging
import sys
from pathlib import Path
from config import constants, switch_properties

def setup_logger(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    root = logging.getLogger()  # root logger
    if root.handlers:
        return
    root.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not(log_file):
        log_file = switch_properties.SWITCH_PROPERTIES[constants.log_path]

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    root.addHandler(fh)

def get_logger(name: str | None = None) -> logging.Logger:
    """Use in any module: logger = get_logger(__name__)."""
    return logging.getLogger(name or constants.credit_assessment)

if __name__ == "__main__":
    setup_logger()
    logger = get_logger()
    logger.info("Hello, world!")
    logger1 = get_logger("data_cleaning")
    logger1.info("Hello, world! from data_cleaning")
