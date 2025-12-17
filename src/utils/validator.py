import os
import pandas as pd
from logging import Logger
from src.utils.logger import get_logger

class Validator:
    def __init__(self):
        self.logger: Logger = get_logger()

    def check_file_path(self, path: str) -> None:
        """
        Checks file path and path type is correct
        """
        if not isinstance(path, str):
            self.logger.error(f"Path must be a string: {type(path)}")
            raise TypeError(f"Path must be a string: {type(path)}")

        if not os.path.isfile(path):
            self.logger.error(f"File {path} does not exist.")
            raise FileNotFoundError(f"File {path} does not exist.")


        self.logger.info('File path is valid.')


    def load_csv(self, path: str) -> pd.DataFrame:
        """
        Checks the csv file upload
        """
        try:
            df = pd.read_csv(path)
            self.logger.info(f'Data successfully loaded. \nShape: {df.shape}\n')
            return df

        except Exception as e:
            self.logger.error(f'Failed reading {path}. Error: {e}')
            raise


    def check_df_type(self, df: pd.DataFrame) -> None:
        """
        Checks the df type is correct
        """
        if not isinstance(df, pd.DataFrame):
            self.logger.error(f"File must be a DataFrame: {type(df)}")
            raise TypeError(f"File must be a DataFrame: {type(df)}")

        self.logger.info('File path is valid.')


    def check_target(self, target: str, df: pd.DataFrame) -> None:
        """
        Checks target is correct
        """

        if target not in df.columns:
            self.logger.error(f'Target column "{target}" not found')
            raise ValueError(f'Target column "{target}" not found')
        self.logger.info(f'Target column "{target}" found.')


    def check_duplicates(self, df) -> bool:
        """
        Checks for duplicate values
        """
        count = df.duplicated().sum()

        if count > 0:
            self.logger.warning(f'{count} duplicates found')
            return True
        else:
            self.logger.info(f'No duplicates found')
            return False


    def check_missing(self, df) -> bool:
        """
        Checks for missing values
        """

        missing = df.isna().sum()
        missing = missing[missing>0]
        if not missing.empty:
            self.logger.warning(f'Missing values found:\n{missing}\n')
            return True
        else:
            self.logger.info('No missing values found.')
            return False










