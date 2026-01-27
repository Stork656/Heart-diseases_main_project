import pandas as pd
from logging import Logger
from typing import Type
from src.utils.logger import get_logger
from pathlib import Path


class Validator:
    """
    Checks the input data

    Attributes:
        logger : Logger
            Logger instance for logging messages and saving logs
    """
    def __init__(self) -> None:
        """
        Initializes Validator
        """
        self.logger: Logger = get_logger()


    def check_type_path(self, path: Path) -> None:
        """
        Checks that the path is a Path object

        Parameters:
            path : Path
                Path object to check

        Raises:
            TypeError: If the path type is not pathlib.Path
        """

        if not isinstance(path, Path):
            self.logger.error(f"Path must be a pathlib.Path, got: {type(path)}")
            raise TypeError(f"Path must be a pathlib.Path, got: {type(path)}")

        self.logger.info(f"Path type - {type(path)} is valid")


    def check_file_exists(self, path: Path) -> None:
        """
        Checks if the file exists

        Parameters:
            path : Path
                Path object to check

        Raises:
            FileNotFoundError: If the file does not exist
        """

        if not path.is_file():
            self.logger.error(f"File {path} does not exist")
            raise FileNotFoundError(f"File {path} does not exist")

        self.logger.info(f"File {path} exist")


    def check_df_type(self, df: pd.DataFrame) -> None:
        """
        Checks if the input is a pandas DataFrame

        Parameters:
            df : pd.DataFrame
            Pandas DataFrame to check

        Raises:
            TypeError: If the df type is not pandas.DataFrame
        """

        if not isinstance(df, pd.DataFrame):
            self.logger.error(f"Input must be a pandas DataFrame: {type(df)}")
            raise TypeError(f"Input must be a pandas DataFrame: {type(df)}")

        self.logger.info(f"Input type - {type(df)} is valid")


    def check_target(self, target: str, df: pd.DataFrame) -> None:
        """
        Checks that the target column exists in the DataFrame

        Parameters:
            target : str
                Name of the target column to check
            df : pd.DataFrame
                Pandas DataFrame to check

        Raises:
            ValueError: If the target column is not found in the DataFrame
        """

        if target not in df.columns:
            self.logger.error(f"Target column {target} not found")
            raise ValueError(f"Target column {target} not found")

        self.logger.info(f"Target column {target} found")


    def check_split_features(self, preprocessor) -> None:
        """
        Verifies that all required split features are correctly set

        Raises:
            ValueError: If the split features is not correct
        """

        empty = []

        if not preprocessor.target_col:
            empty.append('target_col')
        if not preprocessor.binary_cols:
            empty.append('binary_cols')
        if not preprocessor.numeric_cols:
            empty.append('numeric_cols')
        if not preprocessor.categorical_cols:
            empty.append('categorical_cols')

        if not empty:
            features = '\n'.join(f"{key.title()}: {', '.join(map(str, value))}" for key, value in preprocessor.feature_types.items())
            self.logger.info(f"The features are distributed: \n{features}")
        else:
            self.logger.info(f"The following feature lists are empty: {', '.join(empty)}")
            raise ValueError(f"The following feature lists are empty: {', '.join(empty)}")


    def check_duplicates(self, df: pd.DataFrame) -> bool:
        """
        Checks for duplicate values
        """

        count = df.duplicated().sum()

        if count > 0:
            self.logger.warning(f'{count} duplicates found.')
            return True
        else:
            self.logger.info(f'No duplicates found.')
            return False


    def check_missing(self, df: pd.DataFrame) -> bool:
        """
        Checks for missing values
        """

        missing = df.isna().sum()
        missing = missing[missing>0]
        if not missing.empty:
            self.logger.warning(f'Missing values found:\n{missing}')
            return True
        else:
            self.logger.info('No missing values found.')
            return False


    def check_column_exist(self, df: pd.DataFrame, columns: list) ->bool:
        """
        Checks column exist
        """

        missings = []
        for column in columns:
            if column not in df.columns:
                missings.append(column)

        if missings:
            self.logger.error(f'Columns not found in the dataset: {missings}')
            raise TypeError(f'Columns not found in the dataset: {missings}')
        else:
            self.logger.info(f'All columns in the dataset are found: {columns}')
            return True










