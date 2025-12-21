@pytest.fixture
def scaling_df():
    df = pd.DataFrame(
        {
            'HeartDisease': [1, 0, 1, 1],
            'Cholesterol': [100, 0, 600, 100],
            'Age': [13, 50, 20, 18],
    }
    )
    return df


def test_scaling(scaling_df):
    df = scaling_df.copy()
    target = 'HeartDisease'
    sp = SimplePreprocessor(df, target)

    sp.feature_types = {'numeric': df.drop(columns=[target]).columns.tolist()}
    sp.scaling()

    for feature in df.columns:
        mean = sp.df[feature].mean()
        std = sp.df[feature].std()
        assert abs(mean) < 1e-6
        assert abs(std - 1) < 1e-6