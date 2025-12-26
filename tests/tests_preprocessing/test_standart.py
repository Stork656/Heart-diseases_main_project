import pytest
import pandas as pd
from src.preprocessing.standart import StandartPreprocessor


def test_remove_missing_positive(data_test) -> None:
    df = data_test.copy()
    df.iloc[0, 0] = None
    stp = StandartPreprocessor(df)
    stp.split_feature_types()
    stp.remove_missing()

    assert not stp.df.isnull().values.any()
    assert stp.df.iloc[0, 0] == stp.df['HeartDisease'].mode()[0]



def test_remove_missing_negative(data_test) -> None:
    df = data_test.copy()
    stp = StandartPreprocessor(df)
    stp.split_feature_types()
    stp.remove_missing()

    assert not stp.df.isnull().values.any()


def test_remove_missing_real_data(real_data) -> None:
    df = real_data.copy()
    mask = {col: df[col].isna() for col in df.columns}

    stp = StandartPreprocessor(df)
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
    stp = StandartPreprocessor(df)
    stp.split_feature_types()

    outliers = {'Cholesterol': [1000], 'RestingBP': [0]}

    stp.remove_emissions()

    for col, values in outliers.items():
        assert all(val not in stp.df[col].values for val in values)

    assert all(val in stp.df['Cholesterol'].values for val in [280, 310, 300])
    assert all(val in stp.df['RestingBP'].values for val in [130, 120, 135])


def test_remove_emissions_negative(data_test) -> None:
    df = data_test.copy()
    stp = StandartPreprocessor(df)
    stp.split_feature_types()
    stp.df.loc[1, 'Cholesterol'] = 290
    stp.df.loc[1, 'RestingBP'] = 125

    stp.remove_emissions()

    assert len(df) == len(stp.df)


def test_remove_emissions_real_data(real_data) -> None:
    df = real_data.copy()
    stp = StandartPreprocessor(df)
    stp.split_feature_types()

    outliers = {'Cholesterol': [0], 'RestingBP': [0]}

    stp.remove_emissions()

    for col, values in outliers.items():
        assert all(val not in stp.df[col].values for val in values)


def test_encoding_test_data(data_test) -> None:
    pass


def test_encoding_real_data(real_data) -> None:
    pass


def test_scaling_test_data(data_test) -> None:
    pass


def test_scaling_real_data(real_data) -> None:
    pass


@pytest.mark.parametrize("data", ["data_test", "real_data"])
def test_run(request, data):
    df = request.getfixturevalue(data)
    sp = StandartPreprocessor(df, 'HeartDisease')
    sp.run()

    assert len(sp.df) <= len(df)
    assert 'HeartDisease' in sp.df.columns
    assert isinstance(sp.df, pd.DataFrame)