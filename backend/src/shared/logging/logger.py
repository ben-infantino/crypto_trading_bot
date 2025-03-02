import logging
import os
import sys
from datetime import datetime

def setup_logger(name, log_level=logging.INFO, log_to_file=True, log_dir="logs"):
    """
    Set up a logger with console and file handlers.
    
    Args:
        name (str): Logger name
        log_level (int): Logging level
        log_to_file (bool): Whether to log to a file
        log_dir (str): Directory for log files
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if requested
    if log_to_file:
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Example usage
# logger = setup_logger("trading_bot")
# logger.info("Application started")
# logger.error("An error occurred", exc_info=True) 