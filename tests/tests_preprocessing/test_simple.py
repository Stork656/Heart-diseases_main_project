import pytest
import pandas as pd
from src.preprocessing.simple import SimplePreprocessor


def test_remove_missing_positive(data_test):
    df = data_test.copy()
    df.iloc[0,0] = None
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.remove_missing()

    assert not sp.df.isnull().values.any()
    assert len(sp.df) == len(df) - 1


def test_remove_missing_negative(data_test):
    df = data_test.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.remove_missing()

    assert not sp.df.isnull().values.any()
    assert len(sp.df) == len(df)


def test_remove_missing_real_data(real_data):
    df = real_data.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.remove_missing()

    assert not sp.df.isnull().values.any()


def test_remove_emissions_positive(data_test):
    df = data_test.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')
    before = len(sp.df)
    sp.remove_emissions()
    after = len(sp.df)
    assert before > after


def test_remove_emissions_negative(data_test):
    df = data_test.copy()
    df.loc[1, 'Cholesterol'] = 300
    df.loc[1, 'RestingBP'] = 125
    sp = SimplePreprocessor(df, 'HeartDisease')
    before = len(sp.df)
    sp.remove_emissions()
    after = len(sp.df)
    assert before == after


def test_remove_emissions_real_data(real_data):
    df = real_data.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')
    before = len(sp.df)
    sp.remove_emissions()
    after = len(sp.df)
    assert before > after


def test_encoding_test_data(data_test):
    df = data_test.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.split_feature_types()
    encoding_cols = sp.categorical_cols + sp.binary_cols

    sp.encoding()

    dummies_col = [f'{col}_{val}' for col in encoding_cols for val in df[col].unique()]

    assert all(dummy in sp.df.columns for dummy in dummies_col)
    assert len(sp.df) == len(df)
    assert all(num_feat in sp.df.columns for num_feat in sp.numeric_cols)


def test_encoding_real_data(real_data):
    df = real_data.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')

    sp.remove_duplicates()
    sp.remove_missing()
    sp.remove_emissions()
    sp.split_feature_types()

    encoding_cols = sp.categorical_cols + sp.binary_cols
    df = sp.df

    sp.encoding()

    dummies_col = [f'{col}_{val}' for col in encoding_cols for val in df[col].unique()]

    assert all(dummy in sp.df.columns for dummy in dummies_col)
    assert len(sp.df) == len(df)
    assert all(num_feat in sp.df.columns for num_feat in sp.numeric_cols)


@pytest.mark.parametrize("data", ["data_test", "real_data"])
def test_run(request, data):
    df = request.getfixturevalue(data)
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.run()

    assert len(sp.df) <= len(df)
    assert 'HeartDisease' in sp.df.columns
    assert isinstance(sp.df, pd.DataFrame)




