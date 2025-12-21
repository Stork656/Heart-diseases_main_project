from src.preprocessing.base import BasePreprocessor
import pandas as pd

class SimplePreprocessor(BasePreprocessor):
    def __init__(self, df: pd.DataFrame, target: str = 'HeartDisease'):
        super().__init__(df, target)


    def remove_missing(self) -> None:
        """
        Deletes a row if it finds missing values
        """

        if super().remove_missing():
            self.df.dropna(inplace = True)
            self.logger.info(f'Rows with missing values deleted')


    def remove_emissions(self) -> None:
        """
        Deletes unrealistic values
        EDA showed:
        - Cholesterol
        - RestingBP
        """

        emissions = ['Cholesterol', 'RestingBP']
        for em in emissions:
            self.df = self.df.loc[(self.df[em] >= 50)]


    def scaling(self) -> None:
        """
        There is no scaling in a simple pipeline
        """

        pass


    def encoding(self):
        """
        Encoding of categorical and binary-categorical values
        """

        columns_to_encode = self.feature_types['categorical'] + self.feature_types['binary']
        self.df = pd.get_dummies(self.df, columns=columns_to_encode)


    def run_simple_preprocessor(self):
        """
        Run full simple preprocessing pipeline
        """

        self.remove_duplicates()
        self.remove_missing()
        super().split_feature_types()
        self.remove_emissions()
        self.encoding()
















