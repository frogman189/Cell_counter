import logging
import os 
import time


def setup_logging(train_cfg, timestamp, model_args=None, optuna=False, run_exp=True):
    # Create timestamp for the current run and create the directories if they don't exist
    run_dir = train_cfg['result_dir']
    model = model_args['model_type']

    if optuna:
        logger = logging.getLogger(f"optuna_logger_{timestamp}")
        logger.addHandler(logging.NullHandler())
        return logger, None, None
    
    if not run_exp:
        # When optuna or run_exp is True, we still want to log to console but not to a file
        logger = logging.getLogger(f"exp_logger_{timestamp}")
        # Ensure we don't have any handlers attached before adding a StreamHandler for console
        if not logger.hasHandlers():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)  # Log all messages to the console
            logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        return logger, None, None
    
    # Create a unique folder for each run based on model and timestamp
    run_subdir = f"{model}_{timestamp}"
    full_run_dir = os.path.join(run_dir, run_subdir)
    os.makedirs(full_run_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(full_run_dir, f"run_log_{model}_{timestamp}.log")
    logger = logging.getLogger(f"my_custom_logger_{model}_{timestamp}")

    # Ensure handlers are not duplicated
    if logger.hasHandlers():
        logger.handlers.clear()  # Remove any existing handlers
        
    
    logger.setLevel(logging.DEBUG)  # Set the logger level
    
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    
    file_handler.setLevel(logging.WARNING)  # File logs warnings and above
    console_handler.setLevel(logging.DEBUG)  # Console shows all debug and above

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Log the starting information
    date_str = time.strftime("%d/%m/%Y")
    time_str = time.strftime("%H:%M:%S")
    logger.warning(f"Date: {date_str}")
    logger.warning(f"Start Time: {time_str}")
    logger.warning(f"Model: {model}")
    logger.warning(f"Number of epochs: {train_cfg['num_epochs']}")
    logger.warning(f"Learning rate: {train_cfg['learning_rate']}")

    return logger, log_file, full_run_dir