import logging

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Returns a logger for the specified module
    This function DOES NOT configure logging handlers or formatters
    Logging configuration can be done using 'configs/logging.yaml' and initialized once in the main module
    Parameters:
        name : str, optional
            Name of the logger (default is the current module)
    Returns:
        logging.Logger
            Logger instance for the given module
    """
    return logging.getLogger(name)