from src.preprocessing.base import BasePreprocessor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class StandartPreprocessor(BasePreprocessor):
    def __init__(self, df: pd.DataFrame, target: str = 'HeartDisease'):
        super().__init__(df, target)


    def remove_missing(self) -> None:
        """
        Replaces the gaps with the average for numeric features
        and the mode for categorical, binary, target
        """

        if super().remove_missing():
            for feature in self.feature_types['numeric']:
                self.df[feature].fillna(self.df[feature].mean(), inplace=True)

            features_moda = self.feature_types['categorical'] + self.feature_types['binary'] + self.feature_types['target']
            for feature in features_moda:
                self.df[feature].fillna(self.df[feature].mode()[0], inplace=True)


    def remove_emissions(self) -> None:
        """
        Removing percentile outliers
        """

        mask = pd.Series(True, index=self.df.index)
        for feature in self.feature_types['numeric']:
            q1 = self.df[feature].quantile(0.25)
            q3 = self.df[feature].quantile(0.75)
            margin = (q3 - q1) * 1.5
            mask &= (self.df[feature] >= q1 - margin) & (self.df[feature] <= q3 + margin)
        self.df = self.df.loc[mask]


    def scaling(self) -> None:
        """
        Scaling of numerical features with StandardScaler
        """

        for feature in self.feature_types['numeric']:
            scaler = StandardScaler()
            fit_df = self.df[feature].to_frame()
            scaler.fit(fit_df)
            self.df[feature] = scaler.transform(fit_df)


    def encoding(self) -> None:
        columns_to_encode = self.feature_types['categorical'] + self.feature_types['binary']

        encoder = OneHotEncoder()
        encoded_data = encoder.fit_transform(self.df[columns_to_encode])

        encoded_df = pd.DataFrame.sparse.from_spmatrix(
            encoded_data,
            columns = encoder.get_feature_names_out(columns_to_encode),
            index = self.df.index,
        )

        self.df = pd.concat([self.df.drop(columns=columns_to_encode), encoded_df], axis=1)


    def run_standart_preprocessor(self) -> None:
        """
        Run full standart preprocessing pipeline
        """
        super().split_feature_types()
        self.remove_duplicates()
        self.remove_missing()
        self.remove_emissions()
        self.scaling()
        self.encoding()















