import logging

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Return a logger for the given module name.
    This function DOESN'T configure logging handlers/formatters.
    Configure logging centrally in the program entrypoint (main.py).
    """
    return logging.getLogger(name)