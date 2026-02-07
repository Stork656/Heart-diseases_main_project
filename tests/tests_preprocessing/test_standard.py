import numpy as np
import pytest
import pandas as pd
from src.preprocessing.standard import StandardPreprocessor


def test_remove_missing_positive(data_test) -> None:
    """
    Check that the method replace missing values correctly when missing values are present in the test data
    mean - for numeric features
    mode - for categorical features
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    df = data_test.copy()
    df.iloc[1, 1] = None
    stp = StandardPreprocessor(df)
    stp.split_feature_types()
    stp.remove_missing()

    assert not stp.df.isna().values.any()
    assert stp.df.iloc[1, 1] == stp.df['ExerciseAngina'].mode()[0]


def test_remove_missing_negative(data_test) -> None:
    """
    Check that the method doesn't replace missing values when missing values are not present in the test data
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    df = data_test.copy()
    stp = StandardPreprocessor(df)
    stp.split_feature_types()
    stp.remove_missing()

    assert not stp.df.isna().values.any()


def test_remove_missing_real_data(real_data) -> None:
    """
    Check that the method replace all missing values correctly
    mean - for numeric features
    mode - for categorical features
    Parameters:
        real_data : pd.DataFrame
            Real DataFrame provided by a fixture
    """
    df = real_data.copy()
    mask = {col: df[col].isna() for col in df.columns}

    stp = StandardPreprocessor(df)
    stp.split_feature_types()
    stp.remove_missing()

    assert not stp.df.isna().values.any()

    # Сhecks the replacement with the average value
    for col in stp.numeric_cols:
        if mask[col].any():
            mean_val = df[col].mean()
            replaced_vals = stp.df.loc[mask[col], col].values
            assert all(replaced_vals == mean_val)

    # Сhecks the replacement with the mode
    for col in stp.categorical_cols + stp.binary_cols:
        if mask[col].any():
            moda_val = df[col].mode()[0]
            replaced_vals = stp.df.loc[mask[col], col].values
            assert all(replaced_vals == moda_val)


def test_remove_outliers_positive(data_test) -> None:
    """
    Check that the method removes outliers when outliers are present in the test data
    filters by percentiles
    IQR = 0.75 - 0.25
    margin = IQR*1.5
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    df = data_test.copy()
    stp = StandardPreprocessor(df)
    stp.split_feature_types()

    outliers = {'Cholesterol': [1000], 'RestingBP': [0]}
    stp.remove_outliers()

    for col, values in outliers.items():
        assert all(val not in stp.df[col].values for val in values)
    assert all(val in stp.df['Cholesterol'].values for val in [280, 310, 300])
    assert all(val in stp.df['RestingBP'].values for val in [130, 120, 135])


def test_remove_outliers_negative(data_test) -> None:
    """
    Check that the method doesn't remove outliers when outliers are not present in the test data
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    df = data_test.copy()
    stp = StandardPreprocessor(df)
    stp.split_feature_types()
    stp.df.loc[1, 'Cholesterol'] = 290
    stp.df.loc[1, 'RestingBP'] = 125

    stp.remove_outliers()

    assert len(df) == len(stp.df)


def test_remove_outliers_real_data(real_data) -> None:
    """
    Check that the method removes outliers when outliers are present in the test data
    filters by percentiles
    IQR = 0.75 - 0.25
    margin = IQR*1.5
    Parameters:
        real_data : pd.DataFrame
            Real DataFrame provided by a fixture
    """
    df = real_data.copy()
    stp = StandardPreprocessor(df)
    stp.split_feature_types()

    # This value is a clear outlier (see EDA)
    outlier = {'RestingBP': [0]}
    stp.remove_outliers()

    assert outlier not in stp.df['RestingBP'].values


@pytest.mark.parametrize('data', ['data_test', 'real_data'])
def test_encoding_data(request, data) -> None:
    """
    Check that the method encodes test data correctly
    Parameters:
        data : pd.DataFrame
            data_test - Test DataFrame provided by a fixture
            real_data - Real DataFrame provided by a fixture
    """
    df = request.getfixturevalue(data).copy()
    stp = StandardPreprocessor(df)
    stp.split_feature_types()

    encoding_cols = stp.categorical_cols + stp.binary_cols
    numerical_cols = stp.numeric_cols.copy()
    stp.remove_missing()
    cleaned_df = stp.df

    encoded_cols = [f'{col}_{val}' for col in encoding_cols for val in cleaned_df[col].unique()]

    stp.encoding()

    assert all(col in stp.df.columns for col in encoded_cols)
    assert all(col in stp.df.columns for col in numerical_cols)
    assert all(col not in stp.df.columns for col in encoding_cols)
    assert stp.df.shape[1] > cleaned_df.shape[1] and stp.df.shape[0] == cleaned_df.shape[0]
    assert not stp.df.isna().values.any()


@pytest.mark.parametrize('data', ['data_test', 'real_data'])
def test_scaling(request, data) -> None:
    """
    Check that the method scales data correctly
    Parameters:
        data : pd.DataFrame
            data_test - Test DataFrame provided by a fixture
            real_data - Real DataFrame provided by a fixture
    """
    df = request.getfixturevalue(data).copy()
    stp = StandardPreprocessor(df)
    stp.split_feature_types()
    stp.remove_missing()
    stp.remove_outliers()
    stp.encoding()
    stp.scaling()

    for col in stp.numeric_cols:
        mean = stp.df[col].mean()
        std = stp.df[col].std(ddof=0)
        assert np.isclose(mean, 0, atol=1e-7)
        assert np.isclose(std, 1, atol=1e-7)


@pytest.mark.parametrize("data", ["data_test", "real_data"])
def test_run(request, data) -> None:
    """
    Check that the method runs correctly
    Parameters:
        data : pd.DataFrame
            data_test - Test DataFrame provided by a fixture
            real_data - Real DataFrame provided by a fixture
    """
    df = request.getfixturevalue(data)
    stp = StandardPreprocessor(df, 'HeartDisease')
    stp.run()

    assert len(stp.df) <= len(df)
    assert 'HeartDisease' in stp.df.columns
    assert isinstance(stp.df, pd.DataFrame)