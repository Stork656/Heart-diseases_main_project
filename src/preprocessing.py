import pandas as pd
from src.loader import DataLoader
from pandas.api.types import is_numeric_dtype
from pandas.core.dtypes.common import is_numeric_dtype
from src.utils.logger import get_logger
from collections import defaultdict


class Preprocessor:
    '''Data preprocessing class'''

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError('Input must be a pandas DataFrame')

        self.df = df.copy()
        self.logger = get_logger()
        self.logger.info(f'Data preprocessor initialized. Shape: {self.df.shape}')

    def split_by_feature_type(self) -> dict:
        """Detect numeric, categorical, binary and target features."""

        target = 'HeartDisease'
        feature_types = defaultdict(list)
        feature_types['target'] = target

        for col in self.df.drop(columns=[target]).columns:
            if self.df[col].nunique() == 2:
                feature_types['binary'].append(col)
            elif is_numeric_dtype(self.df[col]):
                feature_types['numeric'].append(col)
            else:
                feature_types['categorical'].append(col)

        return feature_types




if __name__ == '__main__':
    loader = DataLoader()
    df = loader.load()

    pre = Preprocessor(df)
    pre.split_by_feature_type()







