from src.preprocessing.base import BasePreprocessor
import pandas as pd


class SimplePreprocessor(BasePreprocessor):
    """
    SimplePreprocessor is a subclass of BasePreprocessor
    Performs a minimal preprocessing pipeline:
    - Removes rows with missing values
    - Filters outliers using predefined thresholds
    - Does not perform feature scaling
    - Applies one-hot encoding using pandas
    """
    def __init__(self, df: pd.DataFrame, target: str = 'HeartDisease') -> None:
        """
        Initializes SimplePreprocessor
        Parameters:
            df : pd.DataFrame
                Input DataFrame to preprocess from parent class
            target : str, optional
                Target column name from parent class (default is 'HeartDisease')
        """
        super().__init__(df, target)


    def remove_missing(self) -> None:
        """
        Removes rows with missing values
        """
        if super().check_missing():
            self.df.dropna(inplace = True)
            self.logger.info(f'Rows with missing values deleted')


    def remove_outliers(self) -> None:
        """
        Filters outliers using predefined thresholds (see EDA)
        """
        if self.validator.check_column_exist(self.df, ['RestingBP']):
            self.df = self.df.loc[(self.df['RestingBP'] >= 50)]


    def scaling(self) -> None:
        """
        There is no scaling in a simple pipeline
        """
        pass


    def encoding(self) -> None:
        """
        Applies one-hot encoding using pandas for categorical and binary features
        """
        columns_to_encode = self.categorical_cols + self.binary_cols
        self.df = pd.get_dummies(self.df, columns=columns_to_encode)


    def run(self) -> None:
        """
        Run full simple preprocessing pipeline
        """
        self.logger.info(f'Running simple preprocessor')
        self.remove_duplicates()
        self.remove_missing()
        self.remove_outliers()
        super().split_feature_types()
        self.encoding()

        counts = self.df[self.target].value_counts()
        self.logger.info(f'Target balance after simple preprocessing:\n{counts}.')