import pytest
from pathlib import Path
from src.utils.validator import Validator
from src.preprocessing.base import BasePreprocessor


# Initializing Validator
validator = Validator()


def test_file_path_Path_positive() -> None:
    """
    Checks that the validator's check_type_path method
    correctly accepts a pathlib.Path object without raising an error
    """
    path = Path('C:')
    validator.check_type_path(path)


def test_file_path_string_negative() -> None:
    """
    Checks that the validator's check_type_path method
    raises a TypeError when a non-path object is passed
    """
    path = "C:"
    with pytest.raises(TypeError):
        validator.check_type_path(path)


def test_file_exists_positive(tmp_path: Path) -> None:
    """
    Checks that the validator's check_file_exists method
    correctly accepts an existing file without raising an error
    Parameters:
        tmp_path : pathlib.Path
             Temporary directory provided by pytest for creating test files
    """
    file = tmp_path / 'file.csv'
    file.write_text('a,b,c')
    validator.check_file_exists(file)


def test_file_exists_negative(tmp_path: Path) -> None:
    """
    Checks that the validator's check_file_exists method
    raises a FileNotFoundError when a non-existent file is provided
    Parameters:
        tmp_path : pathlib.Path
            Temporary directory provided by pytest for creating test files
    """
    file = tmp_path / 'no_file.csv'
    with pytest.raises(FileNotFoundError):
        validator.check_file_exists(file)


def test_df_type_positive(data_test) -> None:
    """
    Checks that the validator's df_type method
    correctly accepts a DataFrame object without raising an error
    Parameters:
        data_test : pd.DataFrame
             pd.DataFrame object to test (fixture)
    """
    validator.check_df_type(data_test)


def test_df_type_negative() -> None:
    """
    Checks that the validator's df_type method
    raises a TypeError when a not DataFrame object is provided
    """
    df = 'пупупу'
    with pytest.raises(TypeError):
        validator.check_df_type(df)


def test_check_target_positive(data_test) -> None:
    """
    Test that check_type_path accepts a valid target data
    """
    target = 'HeartDisease'
    validator.check_target(target, data_test)


def test_check_target_negative(data_test) -> None:
    """
    Test that check_type_path accepts an invalid target data
    """
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


