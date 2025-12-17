import logging.config
import yaml
import os

from src.loader import DataLoader
from src.preprocessing.simple import SimplePreprocessor


def setup_logging(path = 'configs/logging.yaml'):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'File {path} not found')

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config)

def main():
    setup_logging()
    loader = DataLoader()
    df = loader.load()

    simple_preprocessor = SimplePreprocessor(df)
    simple_preprocessor.run_simple_preprocessor()


if __name__ == '__main__':
    main()
