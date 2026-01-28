import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from logging import Logger
from src.utils.logger import get_logger
from src.utils.validator import Validator

class BasePreprocessor:
    """
    The parent class for 3 data processing pipelines:
    - Simple
    - Standard
    - Advanced
    Performs general preprocessing steps:
    - Converts missing values encoded as zeros in the 'Cholesterol' column to NaN
    - Classifies features by type (numeric, categorical, binary)
    - Removes duplicate rows and checks for missing values
    """
    def __init__(self, df: pd.DataFrame, target: str = "HeartDisease"):
        # Component initialization
        self.validator = Validator()
        self.logger: Logger = get_logger()

        # Checking the correctness of the data
        self.validator.check_df_type(df)
        self.validator.check_target(target, df)

        # initializing variables
        self.df: pd.DataFrame = df.copy()
        self.target: str = target
        self.feature_types: dict | None = None
        self.numeric_cols: list | None = None
        self.categorical_cols: list | None = None
        self.binary_cols: list | None = None

        # Logging
        self.logger.info(f"Base preprocessor initialized, shape: {self.df.shape}, target - {self.target}")

        # Calling a method for converts missing values encoded as zeros in the 'Cholesterol' column to NaN
        self.replace_cholesterol_zeros()


    def replace_cholesterol_zeros(self) -> None:
        """
        Converts missing values encoded as zeros in the 'Cholesterol' column to NaN
        """
        if self.validator.check_column_exist(self.df, ["Cholesterol"]):
            n_zeros = (self.df["Cholesterol"] == 0).sum()
            self.df["Cholesterol"].replace(0, np.nan, inplace=True)
            self.logger.info(f"Replaced {n_zeros} zeros with NaN in Cholesterol column")


    def split_feature_types(self) -> dict:
        """
        Classifies features by type (numeric, categorical, binary)
        Returns:
            dict
             Dictionary with keys:
             'target' : list containing target column name
             'binary' : list of binary feature column names
             'numeric' : list of numeric feature column names
             'categorical' : list of categorical feature column names
        """
        # Creating and filling in a dictionary
        self.feature_types = {
            "target": [self.target],
            "binary": [],
            "numeric": [],
            "categorical": []
        }
        for col in self.df.drop(columns=[self.target]).columns:
            if self.df[col].nunique() == 2:
                self.feature_types['binary'].append(col)
            elif is_numeric_dtype(self.df[col]):
                self.feature_types['numeric'].append(col)
            else:
                self.feature_types['categorical'].append(col)

        # initializing variables (for use in child classes)
        self.numeric_cols = self.feature_types['numeric']
        self.categorical_cols = self.feature_types['categorical']
        self.binary_cols = self.feature_types['binary']
        self.target_col = self.feature_types['target'][0]

        # Check splitting features by type
        self.validator.check_split_features(self)
        return self.feature_types


    def remove_duplicates(self) -> None:
        """
        Remove complete duplicate rows
        """
        if self.validator.check_duplicates(self.df):
            self.df = self.df.drop_duplicates()


    def remove_missing(self) -> bool:
        """
        Checks for missing values in the DataFrame
        """
        return self.validator.check_missing(self.df)