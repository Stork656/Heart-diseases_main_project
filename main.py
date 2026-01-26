import logging.config

import pandas as pd
import yaml
import os
from pathlib import Path

from src.utils.logger import get_logger
from src.loader import DataLoader
from src.preprocessing.simple import SimplePreprocessor
from src.preprocessing.standard import StandardPreprocessor
from src.preprocessing.advanced import AdvancedPreprocessor
from src.utils.splitter import splitter
from src.models.training import Models
from src.models.evaluation import Evaluate

def setup_logging(path = 'configs/logging.yaml'):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'File {path} not found')

    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logging.config.dictConfig(config)
    global logger
    logger = get_logger()


def run_preprocessing(PreprocessorClass, df, file_name, processed_dir=Path('data/processed')):
    try:
        logger.info(f'Starting preprocessing - {PreprocessorClass.__name__}.')
        preprocessor = PreprocessorClass(df)
        preprocessor.run()
        preprocessor.df.to_csv(processed_dir / file_name, index=False)
        logger.info(f'Processed file "{file_name}" saved to "{processed_dir}".\n'
                        f'{PreprocessorClass.__name__} finished successfully.\n')
    except Exception as e:
        logger.error(f'{PreprocessorClass.__name__} failed with an error: \n{e}')


def main():
    setup_logging()
    loader = DataLoader()
    df = loader.load()

    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Run preprocessing
    run_preprocessing(SimplePreprocessor, df, 'simple.csv')
    run_preprocessing(StandardPreprocessor, df, 'standard.csv')
    run_preprocessing(AdvancedPreprocessor, df, 'advanced.csv')

    # Run splitting
    file_names = ['simple.csv', 'standard.csv', 'advanced.csv']
    for name in file_names:
        file_path = processed_dir / name
        splitter(file_path, name.replace('.csv', ''))

    split_dir = Path('data/splits')

    for name in file_names:
        name = name.replace('.csv', '')
        X_train = pd.read_csv(split_dir / f'{name}_X_train.csv')
        X_test = pd.read_csv(split_dir / f'{name}_X_test.csv')
        y_train = pd.read_csv(split_dir / f'{name}_y_train.csv').squeeze()
        y_test = pd.read_csv(split_dir / f'{name}_y_test.csv').squeeze()

        logger.info(f'Start train {name} pipline.')
        models = Models(X_train, y_train, preprocessing_type=name)
        models.train_models()

        logger.info(f'Start evaluate {name} pipline.')
        ev = Evaluate(X_test, y_test, preprocessing_type=name)
        ev.evaluate()





if __name__ == '__main__':
    main()

