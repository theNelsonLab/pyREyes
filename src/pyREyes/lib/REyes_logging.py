import os
import logging
import sys

def setup_logging(log_file: str, dir_name: str = "REyes_logs", debug: bool = False) -> None:
    """Configure logging with both console and file output.
    
    Args:
        log_file: Name of the log file to create in the grid_squares directory
        dir_name: Name of directory to create for logs
    """
    log_level = logging.DEBUG if debug else logging.INFO

    # Create output directory
    log_dir = os.path.join(os.getcwd(), dir_name)
    os.makedirs(log_dir, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(log_level)

    # Configure formatters
    console_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Set up console output for user feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    console_handler.flush = sys.stdout.flush
    root_logger.addHandler(console_handler)

    # Set up file output for persistent logging
    log_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

def log_print(message: str, level: int = logging.INFO) -> None:
    """Helper function to log messages at specified level (defaults to INFO)."""
    logging.log(level, message)