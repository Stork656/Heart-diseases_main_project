import pandas as pd
from src.utils.validator import Validator
from pathlib import Path
from src.utils.logger import get_logger
from sklearn.model_selection import train_test_split


def splitter(file_path: Path, name: str, target: str = 'HeartDisease'):
    """

    """

    logger = get_logger()
    validator = Validator()

    validator.check_type_path(file_path)
    validator.check_file_exists(file_path)

    logger.info(f'Starting split for {name} pipline. File path: {file_path}')
    df = pd.read_csv(file_path)
    validator.check_df_type(df)
    validator.check_target(target, df)

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    save_dir = Path('data/splits')
    save_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(save_dir / f'{name}_X_train.csv', index=False)
    X_test.to_csv(save_dir / f'{name}_X_test.csv', index=False)
    y_train.to_csv(save_dir / f'{name}_y_train.csv', index=False)
    y_test.to_csv(save_dir / f'{name}_y_test.csv', index=False)

    logger.info(f'Split for {name} pipline is done. File saved to: {save_dir}.\n')











