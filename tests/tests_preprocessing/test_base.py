import pytest
import pandas as pd
from src.preprocessing.base import BasePreprocessor
from src.loader import DataLoader
from src.utils.validator import Validator

#Fixtures
@pytest.fixture
def expected():
    expected = {
        'target': ['HeartDisease'],
        'binary': ['ExerciseAngina'],
        'numeric': ['age'],
        'categorical': ['ChestPainType'],
    }
    return expected


@pytest.fixture
def expected_types():
    expected_types = {
        'target': ['HeartDisease'],
        'binary': ['Sex', 'FastingBS', 'ExerciseAngina'],
        'numeric': ['Age', 'RestingBP', 'Cholesterol', 'MaxHR',  'Oldpeak'],
        'categorical': ['ChestPainType', 'RestingECG', 'ST_Slope']}
    return expected_types


#Tests
def test_split_feature_types_binary_str(data_test, expected) :
    df = data_test.copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    feature_types = bp.split_feature_types()
    assert feature_types == expected


def test_split_feature_types_binary_num(data_test, expected):
    df = data_test.copy()
    df['ExerciseAngina'] = [1, 0, 1, 0]
    bp = BasePreprocessor(df, target='HeartDisease')
    feature_types = bp.split_feature_types()
    assert feature_types == expected


def test_split_feature_types_binary_bool(data_test, expected):
    df = data_test.copy()
    df['ExerciseAngina'] = [True, False, True, False]
    bp = BasePreprocessor(df, target='HeartDisease')
    feature_types = bp.split_feature_types()
    assert feature_types == expected


def test_split_features_types_real_data(real_data, expected_types):
    df = real_data.copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    feature_types = bp.split_feature_types()
    for key, values in expected_types.items():
        assert all(value in feature_types[key] for value in values)


def test_remove_duplicates_positive(data_test):
    df = data_test.copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    before = len(bp.df)
    bp.remove_duplicates()
    after = len(bp.df)
    assert after < before


def test_remove_duplicates_negative(data_test):
    df = data_test.copy()
    df = df.drop_duplicates()
    bp = BasePreprocessor(df, target='HeartDisease')
    before = len(bp.df)
    bp.remove_duplicates()
    after = len(bp.df)
    assert after == before


def test_remove_duplicates_real_data(real_data):
    df = real_data.copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    before = len(bp.df)
    bp.remove_duplicates()
    after = len(bp.df)
    assert after == before


def test_remove_missing_negative(data_test):
    df = data_test.copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    assert not bp.remove_missing()


def test_remove_missing_positive(data_test):
    df = data_test.copy()
    df.iloc[0, 0] = None
    bp = BasePreprocessor(df, target='HeartDisease')
    assert bp.remove_missing()


def test_remove_missing_real_data(real_data):
    df = real_data.copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    assert bp.remove_missing()