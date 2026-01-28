import pandas as pd
from logging import Logger
from src.utils.logger import get_logger
from pathlib import Path


class Validator:
    """
    Performs validation checks on input data
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
                Input path to validate
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
                Input path to validate
        Raises:
            FileNotFoundError: If the file does not exist
        """
        if not path.is_file():
            self.logger.error(f"File {path} does not exist")
            raise FileNotFoundError(f"File {path} does not exist")

        self.logger.info(f"File {path} exists")


    def check_df_type(self, df: pd.DataFrame) -> None:
        """
        Checks if the input is a pandas DataFrame
        Parameters:
            df : pd.DataFrame
                Input DataFrame to validate
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
                Input DataFrame to validate
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
        # Check which feature lists are empty. empty = [] means all features are present.
        empty = []
        if not preprocessor.target_col:
            empty.append('target_col')
        if not preprocessor.binary_cols:
            empty.append('binary_cols')
        if not preprocessor.numeric_cols:
            empty.append('numeric_cols')
        if not preprocessor.categorical_cols:
            empty.append('categorical_cols')

        # Log the distribution of feature types
        if not empty:
            features = "\n".join(f"{key.title()}: {', '.join(map(str, value))}" for key, value in preprocessor.feature_types.items())
            self.logger.info(f"The features are distributed: \n{features}")
        # Raise error if any feature list is empty
        else:
            self.logger.info(f"The following feature lists are empty: {', '.join(empty)}")
            raise ValueError(f"The following feature lists are empty: {', '.join(empty)}")


    def check_duplicates(self, df: pd.DataFrame) -> bool:
        """
        Checks for duplicate rows in the DataFrame
        Parameters:
            df : pd.DataFrame
                Input DataFrame to validate
        Returns:
            bool
                True if duplicate values were found, False otherwise
        """
        count = df.duplicated().sum()
        if count > 0:
            self.logger.warning(f"{count} duplicates found")
            return True

        self.logger.info(f"No duplicates found")
        return False


    def check_missing(self, df: pd.DataFrame) -> bool:
        """
        Checks for missing values in the DataFrame
        Parameters:
            df : pd.DataFrame
                Input DataFrame to validate
        Returns:
            bool
                True if missing values were found, False otherwise
        """
        # Count missing values per column and keep only columns with missing data
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            self.logger.warning(f"Missing values found:\n{missing}")
            return True

        self.logger.info("No missing values found")
        return False


    def check_column_exist(self, df: pd.DataFrame, columns: list[str]) -> bool:
        """
        Checks if columns exist in the DataFrame
        Parameters:
            df : pd.DataFrame
                Input DataFrame to validate
            columns : list[str]
                List of columns to check for existence in the DataFrame
        Returns:
            bool
                True if columns exist in the DataFrame
        Raises:
            ValueError: If one or more columns do not exist
        """
        # Identify columns that are missing from the DataFrame
        missings = []
        for column in columns:
            if column not in df.columns:
                missings.append(column)

        if missings:
            self.logger.error(f'Columns not found in the dataset: {missings}')
            raise ValueError(f'Columns not found in the dataset: {missings}')

        self.logger.info(f'All columns in the dataset are found: {columns}')
        return True