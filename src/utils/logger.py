import logging
import logging.config
import yaml
import os

config_load = False

def get_logger(path: str = '../configs/logging.yaml') -> logging.Logger:
    '''Load YAML config and return a configured logger'''

    global config_load
    if not config_load:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Config file <{path}> not found')

        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logging.config.dictConfig(config)
        config_load = True

    logger = logging.getLogger(__name__)

    return logger