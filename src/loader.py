import pandas as pd
from src.utils.validator import Validator
from logging import Logger
from src.utils.logger import get_logger
from pathlib import Path


class DataLoader:
    """
    Loads dataset from a CSV file.

    Attributes:
        logger : Logger
            Logger instance for logging messages and saving logs

        validator : Validator
            Validator instance for validating input data

        path : Path
            Path to CSV file

        df : pd.DataFrame or None
            Loaded DataFrame (None until loaded)
    """

    def __init__(self, path: Path = Path("data/raw/heart-diseases.csv")):
        """
        Initializes DataLoader.

        Parameters:
            path : Path, optional
                Path to the CSV file (default "data/raw/heart-diseases.csv")
        """

        # Component initialization
        self.logger: Logger = get_logger()
        self.validator = Validator()

        # Variables initialization
        self.path: Path = path.resolve()
        self.validator.check_type_path(self.path)
        self.validator.check_file_exists(self.path)
        self.df: pd.DataFrame | None = None


    def load(self) -> pd.DataFrame:
        """
        Load CSV file into pandas DataFrame.

        Returns:
            pd.DataFrame
                Loaded DataFrame
        """

        self.df = pd.read_csv(self.path)
        self.logger.info(f"DataFrame is loaded. \nShape: {self.df.shape}\n")
        return self.df
