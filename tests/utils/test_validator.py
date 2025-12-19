import pytest
import pandas as pd
import pathlib
from src.utils.validator import Validator
from src.loader import DataLoader


validator = Validator()


def test_file_path_string_positive() -> None:
    path = 'строка'
    validator.check_type_path(path)


def test_file_path_string_negative() -> None:
    path = 123
    with pytest.raises(TypeError):
        validator.check_type_path(path)


def test_file_exists_positive(tmp_path: pathlib.Path) -> None:
    file = tmp_path / 'file.csv'
    file.write_text('a,b,c')
    validator.check_file_exists(file)


def test_file_exists_negative(tmp_path: pathlib.Path) -> None:
    file = tmp_path / 'no_file.csv'
    with pytest.raises(FileNotFoundError):
        validator.check_file_exists(file)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({'a': [1, 2, 1], 'b': [2, 5, 2], 'c': [3, 4, 3]})


def test_df_type_positive(sample_df) -> None:
    validator.check_df_type(sample_df)


def test_df_type_negative() -> None:
    df = 'пупупу'
    with pytest.raises(TypeError):
        validator.check_df_type(df)


def test_check_target_positive(sample_df) -> None:
    target = 'a'
    validator.check_target(target, sample_df)


def test_check_target_negative(sample_df) -> None:
    target = 'пупупу'
    with pytest.raises(ValueError):
        validator.check_target(target, sample_df)


def test_check_duplicates_positive(sample_df) -> None:
    assert validator.check_duplicates(sample_df)


def test_check_duplicates_negative(sample_df) -> None:
    no_duplicates_df = sample_df.drop_duplicates()
    assert not validator.check_duplicates(no_duplicates_df)


def test_check_missing_negative(sample_df) -> None:
    assert validator.check_missing(sample_df) == False


def test_check_missing_positive(sample_df) -> None:
    sample_df.iloc[0, 0] = None
    assert validator.check_missing(sample_df) == True

