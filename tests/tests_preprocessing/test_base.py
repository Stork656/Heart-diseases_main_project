import pytest
from src.preprocessing.base import BasePreprocessor
import pandas as pd


def test_split_feature_types_binary_str(data_test: pd.DataFrame, expected: dict) -> None:
    """
    Checks that binary features represented as strings are correctly identified
    and that all feature types match the expected distribution in the test data
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
        expected : dict
            Expected feature type distribution
    """
    df = data_test.copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    feature_types = bp.split_feature_types()
    assert feature_types == expected


def test_split_feature_types_binary_num(data_test: pd.DataFrame, expected: dict) -> None:
    """
    Checks that binary features represented as numeric (0/1) are correctly identified
    and that all feature types match the expected distribution in the test data
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
        expected : dict
            Expected feature type distribution
    """
    df = data_test.copy()
    df['ExerciseAngina'] = [1, 0, 0, 1, 0]
    bp = BasePreprocessor(df, target='HeartDisease')
    feature_types = bp.split_feature_types()
    assert feature_types == expected


def test_split_feature_types_binary_bool(data_test: pd.DataFrame, expected: dict) -> None:
    """
    Checks that binary features represented as boolean (True/False) are correctly identified
    and that all feature types match the expected distribution in the test data
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
        expected : dict
            Expected feature type distribution
    """
    df = data_test.copy()
    df['ExerciseAngina'] = [True, False, False, True, False]
    bp = BasePreprocessor(df, target='HeartDisease')
    feature_types = bp.split_feature_types()
    assert feature_types == expected


def test_split_features_types_real_data(real_data: pd.DataFrame, expected_types: dict) -> None:
    """
    Checks that all feature types in the real dataset match the expected distribution
    Parameters:
        real_data : pd.DataFrame
            Real DataFrame provided by a fixture
        expected_types : dict
            Expected feature type distribution in the real data
    """
    df = real_data.copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    feature_types = bp.split_feature_types()
    for key, values in expected_types.items():
        assert all(value in feature_types[key] for value in values)


def test_remove_duplicates_positive(data_test: pd.DataFrame) -> None:
    """
    Checks that all duplicate rows are removed from the test data
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    df = data_test.copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    before = len(bp.df)
    bp.remove_duplicates()
    after = len(bp.df)
    assert after < before


def test_remove_duplicates_negative(data_test: pd.DataFrame) -> None:
    """
    Checks that no rows are removed from the test data if there are no duplicates
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    df = data_test.copy()
    df = df.drop_duplicates()
    bp = BasePreprocessor(df, target='HeartDisease')
    before = len(bp.df)
    bp.remove_duplicates()
    after = len(bp.df)
    assert after == before


def test_remove_duplicates_real_data(real_data: pd.DataFrame) -> None:
    """
    Checks that no rows are removed from the real data
    Parameters:
        real_data : pd.DataFrame
            Real DataFrame provided by a fixture
    """
    df = real_data.copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    before = len(bp.df)
    bp.remove_duplicates()
    after = len(bp.df)
    assert after == before


def test_check_missing_negative(data_test: pd.DataFrame) -> None:
    """
    Checks that method returns False when there are no missing values in the test data
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    df = data_test.copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    assert not bp.check_missing()


def test_check_missing_positive(data_test: pd.DataFrame) -> None:
    """
    Checks that method returns True
    when missing values are present in the test data
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    df = data_test.copy()
    df.iloc[0, 0] = None
    bp = BasePreprocessor(df, target='HeartDisease')
    assert bp.check_missing()


def test_check_missing_real_data(real_data: pd.DataFrame) -> None:
    """
    Checks that method returns True in the real data
    Parameters:
        real_data : pd.DataFrame
            Real DataFrame provided by a fixture
    """
    df = real_data.copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    assert bp.check_missing()