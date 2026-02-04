import importlib
import numpy as np
import pandas as pd
import yaml
import joblib
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.validator import Validator


class Evaluate:
    """
    A class to evaluate trained models
    Metrics used:
    - Accuracy
    - Precision
    - Recall
    - F2
    - ROC-AUC
    - in EDA PR - curve
    Metrics configuration are defined in 'configs/metrics.yaml'
    Attributes:
        validator : Validator
            Validator instance for validating input data
        logger : Logger
            Logger instance for logging messages and saving logs
        X_test : pd.DataFrame
            Test data
        y_test : pd.Series
            Test labels
        preprocessing_type : str
            Type of preprocessing to used
        config_path : Path
            Path to the metrics YAML file (default is 'configs/metrics.yaml')
        models : dict
            Dictionary of models
        config : dict
            Dictionary of configuration parameters
        metrics : dict
            Dictionary of metrics to used

    """
    def __init__(self,
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 preprocessing_type: str,
                 config_path: Path = Path('configs/metrics.yaml')) -> None:
        """
        Initialize the Evaluate class
        Parameters:
            X_test : pd.DataFrame
                Test data
            y_test : pd.Series
                Test labels
            preprocessing_type : str
                Type of preprocessing to used
            config_path : Path
                Path to the metrics YAML file (default is 'configs/metrics.yaml')
            config :
            models_path : Path
                Path to the models (default is 'models')
            save_path : Path
                Path to save results

        """
        # Component initialization
        self.validator = Validator()
        self.logger = get_logger()

        # initializing variables
        self.X_test = X_test
        self.y_test = y_test
        self.preprocessing_type = preprocessing_type
        self.config_path = config_path.resolve()
        self.models = {}

        # Validate the configuration path
        self.validator.check_type_path(config_path)
        self.validator.check_file_exists(config_path)

        # Load configuration from YAML file
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Set path to save results and loading models
        self.models_path = Path("models") / self.preprocessing_type
        self.save_path = Path("results")
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Calling a methods for loading metrics and models
        self.load_metrics()
        self.load_trained_models()


    def get_class_from_string(self, class_path: str) -> type:
        """
        Loads the metric and the class of the metric used
        Parameters:
            class_path : str
                Full path to the class, including module and class name
                (Example: 'sklearn.metrics.accuracy_score')
        Returns:
            type
                The class of the metric
        """
        module_name, class_name = class_path.rsplit(".",1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls


    def load_metrics(self) -> None:
        """
        Loads metrics and creates a dictionary with settings
        """
        self.metrics = {}

        for metric_name, config in self.config["metrics"].items():
            self.metrics[metric_name] = {
                "fn": self.get_class_from_string(config["class"]),
                "params": config.get("params", {})
            }

    def load_trained_models(self) -> None:
        """
        Loads trained models and create dictionary with them
        """
        for model in self.models_path.glob('*.joblib'):
            model_name = model.stem
            self.models[model_name] = joblib.load(model)
            self.logger.info(f'Loaded {model_name} model')


    def evaluate(self) -> None:
        """

        """
        metrics = []
        y_scores = {}
        y_predictions = {}

        # Get predictions on th test
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_predictions[model_name] = y_pred

            #
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