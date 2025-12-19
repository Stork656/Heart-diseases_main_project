from src.preprocessing.base import BasePreprocessor

class SimplePreprocessor(BasePreprocessor):
    def __init__(self, df: pd.DataFrame, target: str = 'HeartDisease'):
        super().__init__(df, target)



    def remove_missing(self) -> None:
        """
        Deletes a row if it finds missing values
        """

        if super().check_missing():
            self.df = self.df.dropna(inplace = True)
            self.logger.info(f'Rows with missing values deleted')


    def run_simple_preprocessor(self):
        """
        Run full simple preprocessing pipeline
        """

        self.remove_duplicates()
        self.remove_missing()














