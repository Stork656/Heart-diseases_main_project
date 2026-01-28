from src.preprocessing.base import BasePreprocessor
import pandas as pd


class SimplePreprocessor(BasePreprocessor):
    """
    Child class of the base class
    Performs a simple data processing pipeline and contains minimal processing steps:
    - Rows with missing values are removed
    - Outliers are filtered using a predefined threshold
    - No feature scaling
    - One-Hot Encoding using Pandas
    """
    def __init__(self, df: pd.DataFrame, target: str = 'HeartDisease'):
        """

        """
        super().__init__(df, target)


    def remove_missing(self) -> None:
        """
        Deletes a row if it finds missing values
        """

        if super().remove_missing():
            self.df.dropna(inplace = True)
            self.logger.info(f'Rows with missing values deleted')


    def remove_outliers(self) -> None:
        """
        Deletes unrealistic values
        EDA showed:
        - Cholesterol
        - RestingBP
        """
        if self.validator.check_column_exist(self.df, ['RestingBP']):
            self.df = self.df.loc[(self.df['RestingBP'] >= 50)]


    def scaling(self) -> None:
        """
        There is no scaling in a simple pipeline
        """

        pass


    def encoding(self):
        """
        Encoding of categorical and binary-categorical values
        """

        columns_to_encode = self.categorical_cols + self.binary_cols
        self.df = pd.get_dummies(self.df, columns=columns_to_encode)


    def run(self):
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