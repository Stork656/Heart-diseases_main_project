from src.preprocessing.base import BasePreprocessor
import pandas as pd
from sklearn.preprocessing import RobustScaler
from  sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest


class AdvancedPreprocessor(BasePreprocessor):
    """

    """


    def __init__(self, df: pd.DataFrame, target: str = 'HeartDisease'):
        super().__init__(df, target)


    def encoding(self) -> None:
        """

        """

        encoding_data = self.categorical_cols + self.binary_cols
        for column in encoding_data:
            freq = self.df[column].value_counts(normalize=True)
            self.df[column] = self.df[column].map(freq)


    def remove_missing(self) -> None:
        """
        Remove the missing values from dataframe with KNNImputer
        (number of neighbors = 5)
        """
        self.df = self.df.dropna(subset=[self.target])

        if super().remove_missing():
            imputer = KNNImputer(n_neighbors=5)
            self.df.loc[:, :] = imputer.fit_transform(self.df)


    def scaling(self) -> None:
        """
        Scaling of numerical features with RobustScaler
        """

        scaler = RobustScaler()
        self.df[self.numeric_cols] = self.df[self.numeric_cols].astype(float)
        self.df.loc[:, self.numeric_cols] = scaler.fit_transform(self.df.loc[:, self.numeric_cols])


    def remove_outliers(self) -> None:
        """

        """
        iso = IsolationForest(contamination='auto', random_state=42)
        mask = iso.fit_predict(self.df[self.numeric_cols])
        self.df = self.df[mask == 1]


    def run(self) -> None:
        """
        Run full advansed preprocessing pipeline
        """

        self.remove_duplicates()
        super().split_feature_types()
        self.encoding()
        self.remove_missing()
        self.scaling()
        self.remove_outliers()

        counts = self.df[self.target].value_counts()
        self.logger.info(f'Target balance after advanced preprocessing:\n{counts}.')

