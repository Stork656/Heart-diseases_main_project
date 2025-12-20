import pandas as pd
from pandas.api.types import is_numeric_dtype
from logging import Logger
from src.loader import DataLoader
from src.utils.logger import get_logger
from collections import defaultdict
from src.utils.validator import Validator

class BasePreprocessor:
    """
    ***
    """

    def __init__(self, df: pd.DataFrame, target: str = 'HeartDisease'):
        #Checking the correctness of the data
        self.validator = Validator()
        self.validator.check_df_type(df)
        self.validator.check_target(target, df)

        #initializing variables
        self.df: pd.DataFrame = df.copy()
        self.target: str = target
        self.feature_types: dict | None = None

        #Enabling logging
        self.logger: Logger = get_logger()
        self.logger.info(f'Base preprocessor initialized. '
                         f'\nShape: {self.df.shape}\n'
                         f'Target: {self.target}\n')


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
        features = '\n'.join(f'{key.title()}: {', '.join(map(str, value))}' for key, value in self.feature_types.items())
        self.logger.info(f'The features are distributed. \n{features}')
        return self.feature_types


    def remove_duplicates(self) -> None:
        """
        Deletes complete duplicate rows
        """

        if self.validator.check_duplicates(self.df):
            self.df = self.df.drop_duplicates()


    def remove_missing(self) -> bool:
        """
        Checks for missing values
        """

        if self.validator.check_missing(self.df):
            return True













