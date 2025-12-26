import logging.config
import yaml
import os
from pathlib import Path
from src.utils.logger import get_logger
from src.loader import DataLoader
from src.preprocessing.simple import SimplePreprocessor
from src.preprocessing.standart import StandartPreprocessor
from src.preprocessing.advanced import AdvancedPreprocessor


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
    run_preprocessing(StandartPreprocessor, df, 'standart.csv')
    run_preprocessing(AdvancedPreprocessor, df, 'advanced.csv')



if __name__ == '__main__':
    main()

