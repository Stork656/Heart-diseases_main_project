from src.preprocessing.base import BasePreprocessor
import pandas as pd
from sklearn.preprocessing import RobustScaler
from  sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest


class AdvancedPreprocessor(BasePreprocessor):
    """
    AdvancedPreprocessor is a subclass of BasePreprocessor
    Performs a sophisticated steps of preprocessing pipeline:
    - Applies encoded by frequency
    - Missing values are imputed using KNNImputer
    - Feature scaling with RobustScaler
    - Filters outliers using IsolationForest
    """
    def __init__(self, df: pd.DataFrame, target: str = 'HeartDisease') -> None:
        """
        Initializes StandardPreprocessor
        Parameters:
            df : pd.DataFrame
                Input DataFrame to preprocess from parent class
            target : str, optional
                Target column name from parent class (default is 'HeartDisease')
        """
        super().__init__(df, target)


    def encoding(self) -> None:
        """
        Applies encoded by frequency
        """
        encoding_data = self.categorical_cols + self.binary_cols
        for column in encoding_data:
            freq = self.df[column].value_counts(normalize=True)
            self.df[column] = self.df[column].map(freq)


    def remove_missing(self) -> None:
        """
        Missing values are imputed using KNNImputer
        (number of neighbors = 5)
        Rows with missing values in the target column are dropped
        """
        # Drop missing values in the target column
        self.df = self.df.dropna(subset=[self.target])

        # Impute missing values in the DataFrame
        if super().check_missing():
            features = self.df.drop(columns=[self.target])
            imputer = KNNImputer(n_neighbors=5)
            features_imputed = pd.DataFrame(
                imputer.fit_transform(features),
                columns = features.columns,
                index = features.index
            )

            self.df[features.columns] = features_imputed


    def scaling(self) -> None:
        """
        Scaling of numerical features with RobustScaler
        """
        scaler = RobustScaler()
        self.df[self.numeric_cols] = scaler.fit_transform(self.df[self.numeric_cols])


    def remove_outliers(self) -> None:
        """
        Filters outliers using IsolationForest
        """
        iso = IsolationForest(contamination=0.05, random_state=42)
        mask = iso.fit_predict(self.df[self.numeric_cols])
        self.df = self.df[mask == 1].reset_index(drop=True)


    def run(self) -> None:
        """
        Run full advanced preprocessing pipeline
        """

        self.remove_duplicates()
        super().split_feature_types()
        self.encoding()
        self.remove_missing()
        self.scaling()
        self.remove_outliers()

        counts = self.df[self.target].value_counts()
        self.logger.info(f'Target balance after advanced preprocessing:\n{counts}.')

