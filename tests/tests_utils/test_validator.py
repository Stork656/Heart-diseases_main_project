import pytest
import numpy as np
import pathlib
from src.utils.validator import Validator
from src.preprocessing.base import BasePreprocessor



validator = Validator()


def test_file_path_string_positive():
    path = 'строка'
    validator.check_type_path(path)


def test_file_path_string_negative():
    path = 123
    with pytest.raises(TypeError):
        validator.check_type_path(path)


def test_file_exists_positive(tmp_path: pathlib.Path):
    file = tmp_path / 'file.csv'
    file.write_text('a,b,c')
    validator.check_file_exists(file)


def test_file_exists_negative(tmp_path: pathlib.Path):
    file = tmp_path / 'no_file.csv'
    with pytest.raises(FileNotFoundError):
        validator.check_file_exists(file)


def test_df_type_positive(data_test):
    validator.check_df_type(data_test)


def test_df_type_negative():
    df = 'пупупу'
    with pytest.raises(TypeError):
        validator.check_df_type(df)


def test_check_target_positive(data_test):
    target = 'HeartDisease'
    validator.check_target(target, data_test)


def test_check_target_negative(data_test):
    target = 'пупупу'
    with pytest.raises(ValueError):
        validator.check_target(target, data_test)


def test_split_features_positive(data_test):
    """
    The validator check_split_features() is called inside split_feature_types()
    """
    bp = BasePreprocessor(data_test)
    bp.split_feature_types()


def test_split_features_negative(data_test):
    """
    The validator check_split_features() is called inside split_feature_types()
    """
    data_test = data_test.copy()
    data_test.pop('ExerciseAngina')
    bp = BasePreprocessor(data_test, 'HeartDisease')
    with pytest.raises(ValueError):
        bp.split_feature_types()


def test_split_features_real_data(real_data):
    """
    The validator check_split_features() is called inside split_feature_types()
    """
    real_data = real_data.copy()
    bp = BasePreprocessor(real_data, 'HeartDisease')
    bp.split_feature_types()


def test_check_duplicates_positive(data_test):
    assert validator.check_duplicates(data_test)


def test_check_duplicates_negative(data_test):
    df = data_test.copy()
    no_duplicates_df = df.drop_duplicates()
    assert not validator.check_duplicates(no_duplicates_df)


def test_check_missing_negative(data_test):
    assert not validator.check_missing(data_test)


def test_check_missing_positive(data_test):
    df = data_test.copy()
    df.iloc[0, 0] = None
    assert validator.check_missing(df) == True


def test_check_column_exist_positive(data_test):
    columns = ['ExerciseAngina', 'ChestPainType']
    validator.check_column_exist(data_test, columns)


def test_check_column_exist_negative(data_test):
    columns = ['non-existent column']
    with pytest.raises(TypeError):
        validator.check_column_exist(data_test, columns)


def test_check_column_exist_real_data(real_data):
    columns = ['ExerciseAngina', 'ChestPainType']
    validator.check_column_exist(real_data, columns)


