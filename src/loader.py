import pandas as pd
import os.path
from src.utils.validator import Validator

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DataLoader:
    """
    Responsible for loading dataset from CSV.
    """

    def __init__(self, path: str = 'data/raw/heart-diseases.csv'):
        full_path = os.path.join(project_root, path)
        self.validator = Validator()
        self.validator.check_file_path(full_path)

        self.path: str = full_path
        self.df: pd.DataFrame | None = None


    def load(self) -> pd.DataFrame:
        """
        Load CSV file into pandas DataFrame.
        """

        self.df = self.validator.load_csv(self.path)
        return self.df


if __name__ == '__main__':
    loader = DataLoader()
    df = loader.load()
    print(df.head())