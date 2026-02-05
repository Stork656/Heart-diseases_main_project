import pytest
import pandas as pd
from src.preprocessing.simple import SimplePreprocessor


def test_remove_missing_positive(data_test: pd.DataFrame) -> None:
    """
    Check that the method delete row when there are missing values in the test data
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    df = data_test.copy()
    df.iloc[0,0] = None
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.remove_missing()

    assert not sp.df.isnull().values.any()
    assert len(sp.df) == len(df) - 1


def test_remove_missing_negative(data_test: pd.DataFrame) -> None:
    """
    Check that the method doesn't remove any rows when there are no missing values
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    df = data_test.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.remove_missing()

    assert not sp.df.isnull().values.any()
    assert len(sp.df) == len(df)


def test_remove_missing_real_data(real_data: pd.DataFrame) -> None:
    """
    Check that method delete row when there are missing values in the real data
    Parameters:
        real_data : pd.DataFrame
            Real DataFrame provided by a fixture
    """
    df = real_data.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.remove_missing()

    assert not sp.df.isnull().values.any()


def test_remove_outliers_positive(data_test: pd.DataFrame) -> None:
    df = data_test.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')
    before = len(sp.df)
    sp.remove_outliers()
    after = len(sp.df)
    assert before > after


def test_remove_outliers_negative(data_test: pd.DataFrame):
    df = data_test.copy()
    df.loc[1, 'Cholesterol'] = 300
    df.loc[1, 'RestingBP'] = 125
    sp = SimplePreprocessor(df, 'HeartDisease')
    before = len(sp.df)
    sp.remove_outliers()
    after = len(sp.df)
    assert before == after


def test_remove_outliers_real_data(real_data: pd.DataFrame):
    df = real_data.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')
    before = len(sp.df)
    sp.remove_outliers()
    after = len(sp.df)
    assert before > after


def test_encoding_test_data(data_test: pd.DataFrame):
    df = data_test.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.split_feature_types()
    encoding_cols = sp.categorical_cols + sp.binary_cols

    sp.encoding()

    dummies_col = [f'{col}_{val}' for col in encoding_cols for val in df[col].unique()]

    assert all(dummy in sp.df.columns for dummy in dummies_col)
    assert len(sp.df) == len(df)
    assert all(num_feat in sp.df.columns for num_feat in sp.numeric_cols)


@pytest.mark.parametrize('data', ['data_test', 'real_data'])
def test_encoding_data(request, data) -> None:
    df = request.getfixturevalue(data).copy()
    sp = SimplePreprocessor(df, 'HeartDisease')

    sp.remove_duplicates()
    sp.remove_missing()
    sp.remove_emissions()
    sp.split_feature_types()
    cleaned_df = sp.df.copy()

    encoding_cols = sp.categorical_cols + sp.binary_cols
    numerical_cols = sp.numeric_cols.copy()

    dummies_col = [f'{col}_{val}' for col in encoding_cols for val in cleaned_df[col].unique()]

    sp.encoding()

    assert all(dummy in sp.df.columns for dummy in dummies_col)
    assert all(col in sp.df.columns for col in numerical_cols)
    assert all(col not in sp.df.columns for col in encoding_cols)
    assert sp.df.shape[1] > cleaned_df.shape[1] and sp.df.shape[0] == cleaned_df.shape[0]
    assert not sp.df.isna().values.any()


@pytest.mark.parametrize("data", ["data_test", "real_data"])
def test_run(request, data):
    df = request.getfixturevalue(data)
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.run()

    assert len(sp.df) <= len(df)
    assert 'HeartDisease' in sp.df.columns
    assert isinstance(sp.df, pd.DataFrame)




