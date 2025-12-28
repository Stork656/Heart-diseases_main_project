import pandas as pd
from src.utils.validator import Validator
from logging import Logger
from src.utils.logger import get_logger
from pathlib import Path


class DataLoader:
    """
    Responsible for loading dataset from CSV.
    """

    def __init__(self, path: Path = Path('data/raw/heart-diseases.csv')):
        self.path: Path = path.resolve()

        self.validator = Validator()
        self.validator.check_type_path(self.path)
        self.validator.check_file_exists(self.path)

        self.df: pd.DataFrame | None = None
        self.logger: Logger = get_logger()


    def load(self) -> pd.DataFrame:
        """
        Load CSV file into pandas DataFrame.
        """

        self.df = pd.read_csv(self.path)
        self.logger.info(f"DataFrame is loaded. \nShape: {self.df.shape}\n")
        return self.df
