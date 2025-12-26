import pytest
import pandas as pd
from src.loader import DataLoader


@pytest.fixture
def data_test():
    df = pd.DataFrame(
        {
            'HeartDisease': [1, 0, 0, 1, 0],
            'ExerciseAngina': ['Y', 'N', 'N', 'Y', 'N'],
            'age': [25, 30, 45, 25, 50],
            'ChestPainType': ['ASY', 'NAP', 'ATA', 'ASY', 'NAP'],
            'Cholesterol': [280, 1000, 300, 280, 310],
            'RestingBP': [120, 0, 130, 120, 135],
        }
    )
    return df


@pytest.fixture
def real_data():
    loader = DataLoader()
    df = loader.load()
    return df


