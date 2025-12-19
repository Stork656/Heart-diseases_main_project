import pytest
import pandas as pd
from src.preprocessing.base import BasePreprocessor

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        'binary': [True, False, True],
        'categorical': ['ASY', 'NAP', 'ATA'],
        'numeric': [11, 2, 45],
        'target': [1, 0, 1],
    })


def test_split_feature_types(sample_df):
    base_preprocessor = BasePreprocessor(sample_df, target='target')
    feature_types = base_preprocessor.split_feature_types()

    expected = {
        'target': ['target'],
        'binary': ['binary'],
        'categorical': ['categorical'],
        'numeric': ['numeric'],
    }

    for key, value in feature_types.items():
        assert feature_types[key] == value


def test_remove_duplicates():
    pass


def test_remove_missing():
    pass