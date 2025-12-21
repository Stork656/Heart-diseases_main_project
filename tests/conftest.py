import pytest
import pandas as pd
from src.loader import DataLoader


@pytest.fixture
def data_test():
    df = pd.DataFrame(
        {
            'HeartDisease': [1, 0, 1, 1],
            'ExerciseAngina': ['Y', 'N', 'Y', 'Y'],
            'age': [25, 30, 45, 25],
            'ChestPainType': ['ASY', 'NAP', 'ATA', 'ASY'],
            'Cholesterol': [100, 0, 600, 100],
            'RestingBP': [130, 0, 120, 130],
        }
    )
    return df

@pytest.fixture
def real_data():
    loader = DataLoader()
    df = loader.load()
    return df