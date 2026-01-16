import importlib
import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.validator import Validator


class Evaluate:
    def __init__(self,
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 preprocessing_type: str,
                 config_path: Path = Path('configs/metrics.yaml')):

        """

        """
        self.validator = Validator()
        self.logger = get_logger()

        self.X_test = X_test
        self.y_test = y_test
        self.preprocessing_type = preprocessing_type

        self.config_path = config_path.resolve()
        self.validator.check_type_path(config_path)
        self.validator.check_file_exists(config_path)


        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self._load_metrics()

        self.save_path = Path('results')
        self.models_path = Path('models') / self.preprocessing_type
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self._load_trained_models()


    def _get_class_from_string(self, class_path: str):
        module_name, class_name = class_path.rsplit('.',1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls

    def _load_metrics(self):
        self.metrics = {}

        for metric_name, config in self.config['metrics'].items():
            self.metrics[metric_name] = {
                'fn': self._get_class_from_string(config['class']),
                'params': config.get('params', {})
            }

    def _load_trained_models(self):
        for model in self.models_path.glob('*.joblib'):
            model_name = model.stem
            self.models[model_name] = joblib.load(model)
            self.logger.info(f'Loaded {model_name} model')


    def evaluate(self):
        '''

        '''
        metrics = []
        y_scores = {}
        y_predictions = {}

        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_predictions[model_name] = y_pred


            if hasattr(model, 'predict_proba'):
                y_score = model.predict_proba(self.X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_score = model.decision_function(self.X_test)
            else:
                y_score = None


            if y_score is not None:
                y_scores[model_name] = y_score


            row = {"model": model_name}
            for metric_name, metric_data in self.metrics.items():
                metric_fn = metric_data['fn']
                params = metric_data['params']

                if metric_name == 'roc_auc':
                    if y_score is not None:
                        row[metric_name] = metric_fn(self.y_test, y_score, **params)
                    else:
                        row[metric_name] = None
                else:
                    row[metric_name] = metric_fn(self.y_test, y_pred, **params)

            metrics.append(row)

        df_metrics = pd.DataFrame(metrics)
        file_path = self.save_path / f'{self.preprocessing_type}_metrics.csv'
        df_metrics.to_csv(file_path, index=False)
        self.logger.info(f"Metrics saved to '{file_path}':\n{df_metrics}")

        predictions_file = self.save_path / f'{self.preprocessing_type}_y_predict.npy'
        np.save(predictions_file, y_predictions)
        self.logger.info(f"Predictions saved to {predictions_file}")

        if y_scores:
            scores_file = self.save_path / f'{self.preprocessing_type}_y_scores.npy'
            np.save(scores_file, y_scores)
            self.logger.info(f"Scores saved to {scores_file}")



