from pathlib import Path
from src.utils.logger import get_logger
import yaml
import logging.config
from typing import Type
import pandas as pd
from src.loader import DataLoader
from src.preprocessing.base import BasePreprocessor
from src.preprocessing.simple import SimplePreprocessor
from src.preprocessing.standard import StandardPreprocessor
from src.preprocessing.advanced import AdvancedPreprocessor
from src.utils.splitter import splitter
from src.models.training import Models
from src.models.evaluation import Evaluate


def setup_logging(path: Path = Path("configs/logging.yaml")) -> logging.Logger:
    """
    Initializes logging configuration from a YAML file
    Should be called once
    Parameters:
        path : Path, optional
            Path to the logging YAML file (default is 'configs/logging.yaml')
    Raises:
        FileNotFoundError: If the YAML file does not exist
    Returns:
        logging.Logger
            Logger instance configured according to the YAML file
    """
    # If dir not exists
    Path("logs").mkdir(parents=True, exist_ok=True)

    # File exist check
    if not path.is_file():
        raise FileNotFoundError(f"File {path} not found")

    # Reading the configuration file
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Logging initializing
    logging.config.dictConfig(config)

    return get_logger()


def run_preprocessing(PreprocessorClass: Type[BasePreprocessor],
                      df: pd.DataFrame,
                      file_name: str,
                      logger: logging.Logger,
                      processed_dir: Path=Path("data/processed")):
    """
    Executes the data preprocessing pipeline using the specified preprocessor
    Parameters:
        PreprocessorClass : Type[BasePreprocessor]
            Preprocessor class to use
        df : pd.DataFrame
            DataFrame to be processed
        file_name : str
            Name of the file to be processed
        processed_dir : Path, optional
            Directory where the processed data will be stored (default is 'data/processed')
        logger : logging.Logger
            Logger instance for logging messages and saving logs
    """
    try:
        # Launching the pipeline preprocessing data
        logger.info(f"{PreprocessorClass.__name__} is starting")
        preprocessor = PreprocessorClass(df)
        preprocessor.run()
        preprocessor.df.to_csv(processed_dir / file_name, index=False)
        logger.info(f"Processed file {file_name} saved to {processed_dir}\n"
                        f"{PreprocessorClass.__name__} finished successfully\n")
    except Exception as e:
        logger.error(f"{PreprocessorClass.__name__} failed with an error: \n{e}")


def main():
    """
    Performs data preprocessing, trains machine learning models, and evaluates their performance
    """
    # Component initialization
    logger = setup_logging()
    loader = DataLoader()

    # Loading data
    df = loader.load()

    # Setup directory for saving processed files
    processed_dir: Path = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Run preprocessing pipelines
    run_preprocessing(SimplePreprocessor, df, "simple.csv", logger)
    run_preprocessing(StandardPreprocessor, df, "standard.csv", logger)
    run_preprocessing(AdvancedPreprocessor, df, "advanced.csv", logger)

    file_names = ["simple.csv", "standard.csv", "advanced.csv"]

    # Run splitting
    for name in file_names:
        file_path = processed_dir / name
        splitter(file_path, name.replace(".csv", ""))

    # Run training and evaluation
    split_dir = Path("data/splits")
    for name in file_names:
        name = name.replace(".csv", '')
        X_train = pd.read_csv(split_dir / f"{name}_X_train.csv")
        X_test = pd.read_csv(split_dir / f"{name}_X_test.csv")
        y_train = pd.read_csv(split_dir / f"{name}_y_train.csv").squeeze()
        y_test = pd.read_csv(split_dir / f"{name}_y_test.csv").squeeze()

        logger.info(f"Start train {name} pipeline")
        models = Models(X_train, y_train, preprocessing_type=name)
        models.train_models()

        logger.info(f"Start evaluate {name} pipeline")
        ev = Evaluate(X_test, y_test, preprocessing_type=name)
        ev.evaluate()


if __name__ == "__main__":
    main()

