import pandas as pd
from pandas.api.types import is_numeric_dtype
from logging import Logger
from src.loader import DataLoader
from src.utils.logger import get_logger
from collections import defaultdict


class Preprocessor:
    """
    ***
    """

    def __init__(self, df: pd.DataFrame, target: str = 'HeartDisease'):
        if not isinstance(df, pd.DataFrame):
            raise TypeError('Input must be a pandas DataFrame')

        self.df: pd.DataFrame = df.copy()
        self.target: str = target
        self.feature_types: dict | None = None
        self.logger: Logger = get_logger()

        self.logger.info(f'Data preprocessor initialized. \nShape: {self.df.shape}\n')


    def split_feature_types(self) -> dict:
        """
        Classify dataset columns into feature types.
        Returns a dict with keys:
            - target
            - binary
            - numeric
            - categorical
        """

        feature_types = defaultdict(list)
        feature_types['target'] = [self.target]

        for col in self.df.drop(columns=[self.target]).columns:
            if self.df[col].nunique() == 2:
                feature_types['binary'].append(col)
            elif is_numeric_dtype(self.df[col]):
                feature_types['numeric'].append(col)
            else:
                feature_types['categorical'].append(col)

        self.feature_types = dict(feature_types)
        self.logger.info(f'Feature types: \n{dict(self.feature_types)}\n')
        return self.feature_types


    def missing_values(self, way: str = 'simple') -> None:
        """
        Handle missing values in 3 ways:
            - drop rows
            - SimpleImputer
            - KNNImputer
        """

        if way == 'simple':
            self.logger.info("Applying SimpleImputer for missing values")
            result = self._simple_imputer()

        elif way == 'drop':
            self.logger.info("Dropping rows with missing values")
            result = self._drop_rows()

        elif way == 'knn':
            self.logger.info("Applying KNNImputer for missing values")
            result = self._knn_imputer()

        else:
            self.logger.error(f"Unknown missing-values strategy: {way}")
            raise ValueError()

        self.df = result


    def _drop_rows():
        pass


    def _simple_imputer():
        pass


    def _knn_imputer():
        pass




if __name__ == '__main__':
    loader = DataLoader()
    df = loader.load()

    pre = Preprocessor(df)
    pre.split_by_feature_type()









