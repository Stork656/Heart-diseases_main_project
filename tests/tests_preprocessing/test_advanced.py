import pytest
import numpy as np
import pandas as pd
from src.preprocessing.advanced import AdvancedPreprocessor


@pytest.mark.parametrize("data", ["data_test", "real_data"])
def test_encoding(request, data):

    df = request.getfixturevalue(data).copy()
    adp = AdvancedPreprocessor(df)
    adp.split_feature_types()

    encoding_data = adp.categorical_cols + adp.binary_cols
    df_before = adp.df[encoding_data].copy()

    adp.encoding()

    assert all(pd.api.types.is_numeric_dtype(adp.df[col]) for col in encoding_data)

    for col in encoding_data:
        assert all(adp.df[col].dropna().between(0, 1))

    for col in encoding_data:
        original_uniq = df_before[col].dropna().unique()
        encoded = adp.df[col]
        assert all(encoded[df_before[col] == val].nunique() == 1 for val in original_uniq)

    assert adp.df.shape[0] == df.shape[0]


def test_remove_missing_negative(data_test):
    df = data_test.copy()
    mask_before = df.notna().all().all()
    adp = AdvancedPreprocessor(df)
    adp.split_feature_types()
    adp.encoding()
    adp.remove_missing()

    mask_after = adp.df.notna().all().all()

    assert mask_before == mask_after
    assert adp.df.shape == df.shape


def test_remove_missing_positive(data_test):
    df = data_test.copy()
    df.iloc[0:0] = np.nan
    adp = AdvancedPreprocessor(df)
    adp.split_feature_types()
    adp.encoding()
    adp.remove_missing()

    assert adp.df.shape == df.shape
    assert adp.df.notna().all().all()


def test_remove_missing_real_data(real_data):
    df = real_data.copy()
    adp = AdvancedPreprocessor(df)
    adp.split_feature_types()
    adp.encoding()
    adp.remove_missing()

    assert adp.df.shape[0] == df.shape[0] - 5 # Because there are 5 missing in the target and they were simply deleted.
    assert adp.df.notna().all().all()


@pytest.mark.parametrize("data", ["data_test", "real_data"])
def test_scaling(request, data):
    df = request.getfixturevalue(data).copy()
    adp = AdvancedPreprocessor(df)
    adp.split_feature_types()
    adp.encoding()
    adp.remove_missing()

    numeric_cols = adp.numeric_cols.copy()
    df_before = adp.df.copy()

    adp.scaling()

    assert all(np.issubdtype(adp.df[col].dtype, float) for col in numeric_cols)
    assert all(col in adp.df.columns for col in numeric_cols)

    for col in numeric_cols:
        mediana = np.median(adp.df[col])
        assert np.isclose(mediana, 0, atol=1e-7)

    assert adp.df.shape == df_before.shape

@pytest.mark.parametrize("data", ["data_test", "real_data"])
def test_remove_outliers(request, data):
    df = request.getfixturevalue(data).copy()
    adp = AdvancedPreprocessor(df)

    adp.split_feature_types()
    adp.encoding()
    adp.remove_missing()
    adp.scaling()

    df_before = adp.df.copy()

    adp.remove_emissions()

    assert adp.df.shape[0] < df_before.shape[0]
    assert all(col in adp.df.columns for col in df_before.columns)


@pytest.mark.parametrize("data", ["data_test", "real_data"])
def test_run(request, data):
    df = request.getfixturevalue(data)
    adp = AdvancedPreprocessor(df, 'HeartDisease')
    adp.run()

    assert len(adp.df) <= len(df)
    assert 'HeartDisease' in adp.df.columns
    assert isinstance(adp.df, pd.DataFrame)