import pandas as pd
import os.path
from src.utils.logger import get_logger


class DataLoader:
    def __init__(self, path: str = '../data/raw/heart-diseases.csv'):
        if not isinstance(path, str):
            raise TypeError('Path must be a string')

        if not os.path.isfile(path):
            raise FileNotFoundError(f'File <{path}> does not exist')

        self.path = os.path.abspath(path)
        self.df = None
        self.logger = get_logger()


    def load(self) -> pd.DataFrame:
        '''Convert to a dataframe'''
        try:
            self.df = pd.read_csv(self.path)
            self.logger.info(f'Data successfully loaded. Shape: {self.df.shape}')
        except Exception as e:
            self.logger.error(f'Failed reading {self.path}. Error: {e}')
            self.logger.info(f'Failed reading <{self.path}>. Error: {e}')

        return self.df

