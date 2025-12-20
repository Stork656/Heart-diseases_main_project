import pytest
import pandas as pd
from src.preprocessing.base import BasePreprocessor
from src.loader import DataLoader
from src.utils.validator import Validator

#Фикстура, а не parametrize т.к. так можно использовать один набор для всех тестов
#Fixtures
@pytest.fixture
def data_test():
    df = pd.DataFrame(
        {
            'HeartDisease': [1, 0, 1, 1],
            'ExerciseAngina': ['Y', 'N', 'Y', 'Y'],
            'age': [25, 30, 45, 25],
            'ChestPainType': ['ASY', 'NAP', 'ATA', 'ASY']
        }
    )
    expected = {
        'target': ['HeartDisease'],
        'binary': ['ExerciseAngina'],
        'numeric': ['age'],
        'categorical': ['ChestPainType'],
    }
    return df, expected

@pytest.fixture
def real_data():
    loader = DataLoader()
    df = loader.load()
    expected_types = {
        'target': ['HeartDisease'],
        'binary': ['Sex', 'FastingBS', 'ExerciseAngina'],
        'numeric': ['Age', 'RestingBP', 'Cholesterol', 'MaxHR',  'Oldpeak'],
        'categorical': ['ChestPainType', 'RestingECG', 'ST_Slope']}
    return df, expected_types


#Tests
def test_split_feature_types_binary_str(data_test) -> None:
    df, expected = data_test
    bp = BasePreprocessor(df, target='HeartDisease')
    feature_types = bp.split_feature_types()
    assert feature_types == expected


def test_split_feature_types_binary_num(data_test):
    df, expected = data_test
    df = df.copy()
    df['ExerciseAngina'] = [1, 0, 1, 0]
    bp = BasePreprocessor(df, target='HeartDisease')
    feature_types = bp.split_feature_types()
    assert feature_types == expected


def test_split_feature_types_binary_bool(data_test):
    df, expected = data_test
    df = df.copy()
    df['ExerciseAngina'] = [True, False, True, False]
    bp = BasePreprocessor(df, target='HeartDisease')
    feature_types = bp.split_feature_types()
    assert feature_types == expected


def test_split_features_types_real_data(real_data):
    df, expected_types = real_data
    bp = BasePreprocessor(df, target='HeartDisease')
    feature_types = bp.split_feature_types()
    for key, values in expected_types.items():
        assert all(value in feature_types[key] for value in values)


def test_remove_duplicates_positive(data_test):
    df = data_test[0]
    bp = BasePreprocessor(df, target='HeartDisease')
    before = len(bp.df)
    bp.remove_duplicates()
    after = len(bp.df)
    assert after < before


def test_remove_duplicates_negative(data_test):
    df = data_test[0].copy()
    df = df.drop_duplicates()
    bp = BasePreprocessor(df, target='HeartDisease')
    before = len(bp.df)
    bp.remove_duplicates()
    after = len(bp.df)
    assert after == before


def test_remove_duplicates_real_data(real_data):
    df = real_data[0].copy()
    bp = BasePreprocessor(df, target='HeartDisease')
    before = len(bp.df)
    bp.remove_duplicates()
    after = len(bp.df)
    assert after == before


def test_remove_missing_negative(data_test):
    df = data_test[0]
    bp = BasePreprocessor(df, target='HeartDisease')
    assert not bp.remove_missing()


def test_remove_missing_positive(data_test):
    df = data_test[0].copy()
    df.iloc[0, 0] = None
    bp = BasePreprocessor(df, target='HeartDisease')
    assert bp.remove_missing()


def test_remove_missing_real_data(real_data):
    df = real_data[0]
    bp = BasePreprocessor(df, target='HeartDisease')
    assert bp.remove_missing()