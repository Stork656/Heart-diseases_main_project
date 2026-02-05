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
    Checks that the validator's check_df_type method
    correctly accepts a pandas DataFrame without raising an error
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    validator.check_df_type(data_test)


def test_df_type_negative() -> None:
    """
    Checks that the validator's check_df_type method
    raises a TypeError when an object that is not a DataFrame is provided
    """
    df = 'пупупу'
    with pytest.raises(TypeError):
        validator.check_df_type(df)


def test_check_target_positive(data_test) -> None:
    """
    Checks that the validator's check_target method
    correctly accepts a valid target column without raising an error
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    target = 'HeartDisease'
    validator.check_target(target, data_test)


def test_check_target_negative(data_test) -> None:
    """
    Checks that the validator's check_target method
    raises a ValueError when an invalid target column is provided
    """
    target = 'пупупу'
    with pytest.raises(ValueError):
        validator.check_target(target, data_test)


def test_split_features_positive(data_test) -> None:
    """
    Checks that the validator's check_split_features method
    accepts correctly split features without raising an error
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    bp = BasePreprocessor(data_test)
    bp.split_feature_types()


def test_split_features_negative(data_test) -> None:
    """
    Checks that the validator's check_split_features method raises a ValueError
    when features are incorrectly split or required features are missing
    """
    data_test = data_test.copy()
    data_test.pop('ExerciseAngina')
    bp = BasePreprocessor(data_test, 'HeartDisease')
    with pytest.raises(ValueError):
        bp.split_feature_types()


def test_split_features_real_data(real_data) -> None:
    """
    Checks that the validator's check_split_features method
    accepts correctly split features in real data without raising an error
    Parameters:
        real_data : pd.DataFrame
            Real DataFrame provided by a fixture
    """
    real_data = real_data.copy()
    bp = BasePreprocessor(real_data, 'HeartDisease')
    bp.split_feature_types()


def test_check_duplicates_positive(data_test) -> None:
    """
    Checks that the validator's check_duplicates method
    correctly detects duplicate rows
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    assert validator.check_duplicates(data_test)


def test_check_duplicates_negative(data_test) -> None:
    """
    Checks that the validator's check_duplicates method
    correctly returns False when there are no duplicate rows
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    df = data_test.copy()
    no_duplicates_df = df.drop_duplicates()
    assert not validator.check_duplicates(no_duplicates_df)


def test_check_missing_negative(data_test) -> None:
    """
    Checks that the validator's check_missing method
    correctly returns False when there are no missing values
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    assert not validator.check_missing(data_test)


def test_check_missing_positive(data_test) -> None:
    """
    Checks that the validator's check_missing method
    correctly detects missing values
    Parameters:
        data_test : pd.DataFrame
            Test DataFrame provided by a fixture
    """
    df = data_test.copy()
    df.iloc[0, 0] = None
    assert validator.check_missing(df) == True


def test_check_column_exist_positive(data_test) -> None:
    """
    Checks that the validator's check_column_exists method
    correctly detects existing columns
    """
    columns = ['ExerciseAngina', 'ChestPainType']
    validator.check_column_exist(data_test, columns)


def test_check_column_exist_negative(data_test) -> None:
    """

    """
    columns = ['non-existent column']
    with pytest.raises(TypeError):
        validator.check_column_exist(data_test, columns)


def test_check_column_exist_real_data(real_data):
    columns = ['ExerciseAngina', 'ChestPainType']
    validator.check_column_exist(real_data, columns)


