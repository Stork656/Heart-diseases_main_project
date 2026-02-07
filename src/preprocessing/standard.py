from src.preprocessing.base import BasePreprocessor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class StandardPreprocessor(BasePreprocessor):
    """
    StandardPreprocessor is a subclass of BasePreprocessor
    Performs a standard steps of preprocessing pipeline:
    - Replaced missing values with mean (for numerical) or mode (for categorical)
    - Filters outliers using percentiles
    - Applies one-hot encoding using Scikit-Learn
    - Feature scaling with StandardScaler
    """
    def __init__(self, df: pd.DataFrame, target: str = "HeartDisease") -> None:
        """
        Initializes StandardPreprocessor
        Parameters:
            df : pd.DataFrame
                Input DataFrame to preprocess from parent class
            target : str, optional
                Target column name from parent class (default is 'HeartDisease')
        """
        super().__init__(df, target)


    def remove_missing(self) -> None:
        """
        Replaces missing values with mean for numeric features
        and the mode for categorical, binary, target
        """
        if self.df[self.target].isna().sum() > 0:
            self.df = self.df.dropna(subset=[self.target])

        if super().check_missing():
            for feature in self.numeric_cols:
                self.df[feature].fillna(self.df[feature].mean(), inplace=True)

            features_moda = self.categorical_cols + self.binary_cols
            for feature in features_moda:
                self.df[feature].fillna(self.df[feature].mode()[0], inplace=True)


    def remove_outliers(self) -> None:
        """
        Filters outliers using percentiles
        IQR = 0.75 - 0.25
        margin = IQR*1.5
        """
        mask = pd.Series(True, index=self.df.index)
        for feature in self.numeric_cols:
            q1 = self.df[feature].quantile(0.25)
            q3 = self.df[feature].quantile(0.75)
            margin = (q3 - q1) * 1.5
            # Comparing the Series of each numeric feature
            mask &= (self.df[feature] >= q1 - margin) & (self.df[feature] <= q3 + margin)
        self.df = self.df.loc[mask].reset_index(drop=True)


    def encoding(self) -> None:
        """
        Applies one-hot encoding using Scikit-Learn
        """
        columns_to_encode = self.categorical_cols + self.binary_cols

        # Get sparse matrix
        encoder = OneHotEncoder()
        encoded_data = encoder.fit_transform(self.df[columns_to_encode])

        # Convert to data frame
        encoded_df = pd.DataFrame.sparse.from_spmatrix(
            encoded_data,
            columns = encoder.get_feature_names_out(columns_to_encode),
            index = self.df.index,
        )

        # Replacing encoding columns
        self.df = pd.concat([self.df.drop(columns=columns_to_encode), encoded_df], axis=1)


    def scaling(self) -> None:
        """
        Scaling of numerical features with StandardScaler
        """
        scaler = StandardScaler()
        self.df[self.numeric_cols] = scaler.fit_transform(self.df[self.numeric_cols])


    def run(self) -> None:
        """
        Run full standard preprocessing pipeline
        """
        self.remove_duplicates()
        super().split_feature_types()
        self.remove_missing()
        self.remove_outliers()
        self.encoding()
        self.scaling()

        counts = self.df[self.target].value_counts()
        self.logger.info(f'Target balance after standard preprocessing:\n{counts}')