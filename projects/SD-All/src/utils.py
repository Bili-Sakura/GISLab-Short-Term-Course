# src/utils.py

import logging
from datetime import datetime
from pytz import timezone

def time_in_utc_plus_8(*args):
    """
    Convert the current UTC time to UTC+8.
    """
    tz = timezone('Asia/Shanghai')
    return datetime.now(tz).timetuple()

def setup_logging(log_file_path, log_level=logging.INFO):
    """
    Set up logging configuration with UTC+8 time zone.

    Args:
        log_file_path (str): The file path for the log file.
        log_level (int): The logging level (default is logging.INFO).
    """
    logging.basicConfig(
        filename=log_file_path,
        level=log_level,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    # Create a custom formatter with the time conversion
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    formatter.converter = time_in_utc_plus_8

    # Update the logging handlers with the new formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
