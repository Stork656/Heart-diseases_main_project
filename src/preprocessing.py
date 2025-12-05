import pandas as pd
from data.loader import DataLoader
from src.utils.logger import get_logger

class Preprocessor:
    '''Data preprocessing class'''

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError('Input must be a pandas DataFrame')

        self.df = df.copy()
        self.logger = get_logger()

        self.logger.info(f'Data preprocessor initialized. Shape: {self.df.shape}')



loader = DataLoader()
df = loader.load()
pre = Preprocessor(df)
print(df.head())





