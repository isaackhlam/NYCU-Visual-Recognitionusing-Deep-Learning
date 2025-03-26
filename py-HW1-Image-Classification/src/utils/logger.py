import logging
from typing import Optional


def setup_logger(
    name: Optional[str] = None,
    output_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(level)

    log_format = "%(asctime)s:%(name)s:%(levelname)s:%(message)s"
    formatter = logging.Formatter(log_format)

    if output_file is not None:
        file_handler = logging.FileHandler(output_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
