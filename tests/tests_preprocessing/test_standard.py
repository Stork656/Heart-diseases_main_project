import numpy as np
import pytest
import pandas as pd
from src.preprocessing.standard import StandardPreprocessor


def test_remove_missing_positive(data_test) -> None:
    df = data_test.copy()
    df.iloc[0, 0] = None
    stp = StandardPreprocessor(df)
    stp.split_feature_types()
    stp.remove_missing()

    assert not stp.df.isnull().values.any()
    assert stp.df.iloc[0, 0] == stp.df['HeartDisease'].mode()[0]



def test_remove_missing_negative(data_test) -> None:
    df = data_test.copy()
    stp = StandardPreprocessor(df)
    stp.split_feature_types()
    stp.remove_missing()

    assert not stp.df.isnull().values.any()


def test_remove_missing_real_data(real_data) -> None:
    df = real_data.copy()
    mask = {col: df[col].isna() for col in df.columns}

    stp = StandardPreprocessor(df)
    stp.split_feature_types()
    stp.remove_missing()

    assert not stp.df.isnull().values.any()

    # Сhecks the replacement with the average value
    for col in stp.numeric_cols:
        if mask[col].any():
            mean_val = df[col].mean()
            replaced_vals = stp.df.loc[mask[col], col].values
            assert all(replaced_vals == mean_val)

    # Сhecks the replacement with the mode
    for col in stp.categorical_cols + stp.binary_cols + [stp.target_col]:
        if mask[col].any():
            moda_val = df[col].mode()[0]
            replaced_vals = stp.df.loc[mask[col], col].values
            assert all(replaced_vals == moda_val)


def test_remove_emissions_positive(data_test) -> None:
    df = data_test.copy()
    stp = StandardPreprocessor(df)
    stp.split_feature_types()

    outliers = {'Cholesterol': [1000], 'RestingBP': [0]}

    stp.remove_emissions()

    for col, values in outliers.items():
        assert all(val not in stp.df[col].values for val in values)

    assert all(val in stp.df['Cholesterol'].values for val in [280, 310, 300])
    assert all(val in stp.df['RestingBP'].values for val in [130, 120, 135])


def test_remove_emissions_negative(data_test) -> None:
    df = data_test.copy()
    stp = StandardPreprocessor(df)
    stp.split_feature_types()
    stp.df.loc[1, 'Cholesterol'] = 290
    stp.df.loc[1, 'RestingBP'] = 125

    stp.remove_emissions()

    assert len(df) == len(stp.df)


def test_remove_emissions_real_data(real_data) -> None:
    df = real_data.copy()
    stp = StandardPreprocessor(df)
    stp.split_feature_types()

    outliers = {'Cholesterol': [0], 'RestingBP': [0]}

    stp.remove_emissions()

    for col, values in outliers.items():
        assert all(val not in stp.df[col].values for val in values)


@pytest.mark.parametrize('data', ['data_test', 'real_data'])
def test_encoding_data(request, data) -> None:
    df = request.getfixturevalue(data).copy()
    stp = StandardPreprocessor(df)
    stp.split_feature_types()

    encoding_cols = stp.categorical_cols + stp.binary_cols
    numerical_cols = stp.numeric_cols.copy()
    stp.remove_missing()
    cleaned_df = stp.df

    dummies_col = [f'{col}_{val}' for col in encoding_cols for val in cleaned_df[col].unique()]

    stp.encoding()

    assert all(dummy in stp.df.columns for dummy in dummies_col)
    assert all(col in stp.df.columns for col in numerical_cols)
    assert all(col not in stp.df.columns for col in encoding_cols)
    assert stp.df.shape[1] > cleaned_df.shape[1] and stp.df.shape[0] == cleaned_df.shape[0]
    assert not stp.df.isna().values.any()


@pytest.mark.parametrize('data', ['data_test', 'real_data'])
def test_scaling(request, data) -> None:
    df = request.getfixturevalue(data).copy()
    stp = StandardPreprocessor(df)
    stp.split_feature_types()

    stp.remove_missing()
    stp.remove_emissions()
    stp.encoding()

    stp.scaling()

    for col in stp.numeric_cols:
        mean = stp.df[col].mean()
        std = stp.df[col].std(ddof=0)
        assert np.isclose(mean, 0, atol=1e-7)
        assert np.isclose(std, 1, atol=1e-7)


@pytest.mark.parametrize("data", ["data_test", "real_data"])
def test_run(request, data):
    df = request.getfixturevalue(data)
    stp = StandardPreprocessor(df, 'HeartDisease')
    stp.run()

    assert len(stp.df) <= len(df)
    assert 'HeartDisease' in stp.df.columns
    assert isinstance(stp.df, pd.DataFrame)