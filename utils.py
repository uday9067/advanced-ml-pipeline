# Utility Functions for Logging and Helpers

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def log_info(message):
    logging.info(message)


def log_error(message):
    logging.error(message)


def log_debug(message):
    logging.debug(message)


def helper_function():
    # Add your helper functionality here
    pass