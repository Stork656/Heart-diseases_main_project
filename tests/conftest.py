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
            'Cholesterol': [100, 0, 600, 100, 300],
            'RestingBP': [130, 0, 120, 130, 110],
        }
    )
    return df


@pytest.fixture
def real_data():
    loader = DataLoader()
    df = loader.load()
    return df


