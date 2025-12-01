import pandas as pd
from sklearn.preprocessing import StandartScaler
from pandas.api.types import is_numeric_dtype

class Preprocessing:
    def __init__(self, df):
        self.df = df

    def splitting(self) -> tuple[list[str]]:
        '''Разделение на числовые и категориальные признаки'''

        lst_cat_features = []
        lst_num_features = []
        for column in df.columns:
            if not(is_numeric_dtype(df[column])):
                lst_cat_features.append(column)
            else:
                lst_num_features.append(column)

        return lst_cat_features, lst_num_features




