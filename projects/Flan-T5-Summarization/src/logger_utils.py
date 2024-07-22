# src/logger_utils.py

import os
import logging


def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("caption_logger")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(
        logging.WARNING
    )  # Set console handler to WARNING to avoid info logs in terminal
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
