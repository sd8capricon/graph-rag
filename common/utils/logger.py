import logging
import sys


def setup_logger(level=logging.INFO, fmt="%(asctime)s - %(levelname)s - %(message)s"):
    logger = logging.getLogger()

    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    formatter = logging.Formatter(fmt)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.setLevel(level)
