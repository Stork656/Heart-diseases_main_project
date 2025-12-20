import pytest
import pandas as pd
from src.loader import DataLoader
from src.preprocessing.simple import SimplePreprocessor


def test_remove_missing_positive(data_test):
    df = data_test.copy()
    df.iloc[0,0] = None
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.remove_missing()

    assert not sp.df.isnull().values.any()
    assert len(sp.df) == len(df) - 1


def test_remove_missing_negative(data_test):
    df = data_test.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.remove_missing()

    assert not sp.df.isnull().values.any()
    assert len(sp.df) == len(df)


def test_remove_missing_real_data(real_data):
    df = real_data.copy()
    sp = SimplePreprocessor(df, 'HeartDisease')
    sp.remove_missing()

    assert not sp.df.isnull().values.any()



