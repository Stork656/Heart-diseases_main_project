import pandas as pd
from src.utils.logger import get_logger
from sklearn.model_selection import GridSearchCV
import yaml
from pathlib import Path
from src.utils.validator import Validator
import importlib
import joblib


class Models:
    def __init__(self,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 preprocessing_type: str,
                 config_path: Path = Path('configs/models.yaml')):
        """

        """
        self.validator = Validator()
        self.logger = get_logger()

        self.X_train = X_train
        self.y_train = y_train

        self.preprocessing_type = preprocessing_type

        self.config_path = config_path.resolve()
        self.validator.check_type_path(config_path)
        self.validator.check_file_exists(config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self._load_models()

        self.save_path = Path('models') / self.preprocessing_type
        self.save_path.mkdir(parents=True, exist_ok=True)


    def _get_class_from_string(self, class_path: str):
        """

        """
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls

    def _load_models(self):
        """

        """
        self.models = {}
        self.params = {}

        for name, info in self.config['models'].items():
            cls = self._get_class_from_string(info['class'])
            self.models[name] = cls()
            self.params[name] = info.get('params', {})


    def train_models(self):
        """

        """
        self.trained_models = {}
        self.results = {}


        for name, model in self.models.items():
            self.logger.info(f'Training {name} model')
            base_params = self.params[name]

            if name == 'LR':
                params = [
                {
                    'solver': ['lbfgs', 'liblinear'],
                    'C': base_params['C'],
                    'max_iter': base_params['max_iter']
                },
                {
                    'solver': ['liblinear', 'saga'],
                    'C': base_params['C'],
                    'max_iter': base_params['max_iter']
                },
                {
                    'solver': ['saga'],
                    'l1_ratio': base_params['l1_ratio'],
                    'C': base_params['C'],
                    'max_iter': base_params['max_iter']
                }
            ]
            else:
                params = base_params


            gs = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=self.config['gridsearch']['cv'],
                scoring=self.config['gridsearch']['scoring'],
                n_jobs=self.config['gridsearch']['n_jobs']
            )

            gs.fit(self.X_train, self.y_train)

            self.trained_models[name] = gs.best_estimator_
            self.results[name] = {
                'best_params': gs.best_params_,
                'best_score': gs.best_score_
            }

            self.logger.info(f'{name} best params: {gs.best_params_}, best CV score: {gs.best_score_:.4f}')


        for name, model in self.trained_models.items():
            file_path = self.save_path / f'{name}.joblib'
            joblib.dump(model, file_path)
            self.logger.info(f'Saving model {model} at {file_path}')
