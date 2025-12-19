import pandas as pd
import os.path
from src.utils.validator import Validator
from logging import Logger
from src.utils.logger import get_logger

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DataLoader:
    """
    Responsible for loading dataset from CSV.
    """

    def __init__(self, path: str = 'data/raw/heart-diseases.csv'):
        full_path = os.path.join(project_root, path)
        self.validator = Validator()
        self.validator.check_type_path(full_path)
        self.validator.check_file_exists(full_path)

        self.path: str = full_path
        self.df: pd.DataFrame | None = None
        self.logger: Logger = get_logger()


    def load(self) -> pd.DataFrame:
        """
        Load CSV file into pandas DataFrame.
        """

        self.df = pd.read_csv(self.path)
        self.logger.info(f"DataFrame is loaded. \nShape: {self.df.shape}\n")
        return self.df
