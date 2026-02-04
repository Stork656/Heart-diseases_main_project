import logging
import pandas as pd
from src.utils.logger import get_logger
from sklearn.model_selection import GridSearchCV
import yaml
from pathlib import Path
from src.utils.validator import Validator
import importlib
import joblib


class Models:
    """
    A class to training models:
    - LogisticRegression
    - svm.SVC
    - KNeighborsClassifier
    - RandomForestClassifier
    - GradientBoostingClassifier
    - AdaBoostClassifier
    Hyperparameters are tuned using GridSearchCV
    Model and GridSearch configuration are defined in 'configs/models.yaml'
    Attributes:
        validator : Validator
            Validator instance for validating input data
        logger : Logger
            Logger instance for logging messages and saving logs
        X_train : pd.DataFrame
            Training data
        y_train : pd.Series
            Training labels
        preprocessing_type : str
            Type of preprocessing used
        config : dict
            Dictionary of configuration parameters
        config_path : Path
            Path to the models YAML file (default is 'configs/models.yaml')
        save_path : Path
            Path to save trained models
        models : dict or None
            Dictionary of models module and class
        params: dict or None
            Dictionary of model parameters
        trained_models : dict
            Dictionary of trained models
        results : dict
            Dictionary of results from grid search
    """
    def __init__(self,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 preprocessing_type: str,
                 config_path: Path = Path("configs/models.yaml")) -> None:
        """
        Initialize the Models class
        Parameters:
            X_train : pd.DataFrame
                Training data
            y_train : pd.Series
                Training labels
            preprocessing_type : str
                Type of preprocessing used
            config_path : Path
                Path to the models YAML file (default is 'configs/models.yaml')
        """
        # Component initialization
        self.validator = Validator()
        self.logger: logging.Logger = get_logger()

        # initializing variables
        self.X_train: pd.DataFrame = X_train
        self.y_train: pd.Series = y_train
        self.preprocessing_type: str = preprocessing_type
        self.config_path: Path = config_path.resolve()
        self.models: dict | None = None
        self.params: dict | None = None
        self.trained_models: dict | None = None
        self.results: dict | None = None

        # Validate the configuration path
        self.validator.check_type_path(config_path)
        self.validator.check_file_exists(config_path)

        # Load configuration from YAML file
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Calling a method for loading models
        self.load_models()

        # Set path to save trained models
        self.save_path = Path("models") / self.preprocessing_type
        self.save_path.mkdir(parents=True, exist_ok=True)


    def get_class_from_string(self, class_path: str) -> type:
        """
        Loads the module and the class of the model used
        Parameters:
            class_path : str
                Full path to the class, including module and class name
                (Example: 'sklearn.linear_model.LogisticRegression')
        Returns:
            type
                The class of the model
        """
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls

    def load_models(self) -> None:
        """
        Initializes the 'models' and 'params' dictionaries.
        Each model class is loaded and an instance is created.
        Parameters for each model are stored separately.
        """
        self.models = {}
        self.params = {}

        for name, info in self.config["models"].items():
            cls = self.get_class_from_string(info["class"])
            self.models[name] = cls()
            self.params[name] = info.get("params", {})


    def train_models(self) -> None:
        """
        Trains all models using GridSearchCV.
        Special handling is applied for LogisticRegression
        due to constraints with l1_ratio and solver compatibility
        """
        self.trained_models = {}
        self.results = {}

        for name, model in self.models.items():
            self.logger.info(f"Training {name} model")
            base_params = self.params[name]

            # Separate processing for Logistic regression
            if name == "LR":
                params = [
                {
                    "solver": ["lbfgs", "liblinear"],
                    "C": base_params["C"],
                    "max_iter": base_params["max_iter"]
                },
                {
                    "solver": ["liblinear", "saga"],
                    "C": base_params["C"],
                    "max_iter": base_params["max_iter"]
                },
                {
                    "solver": ["saga"],
                    "l1_ratio": base_params["l1_ratio"],
                    "C": base_params["C"],
                    "max_iter": base_params["max_iter"]
                }
            ]
            else:
                params = base_params

            # Tuning
            gs = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=self.config["gridsearch"]["cv"],
                scoring=self.config["gridsearch"]["scoring"],
                n_jobs=self.config["gridsearch"]["n_jobs"]
            )
            gs.fit(self.X_train, self.y_train)

            # Saving parameters and score
            self.trained_models[name] = gs.best_estimator_
            self.results[name] = {
                "best_params": gs.best_params_,
                "best_score": gs.best_score_
            }
            self.logger.info(f"{name} best params: {gs.best_params_}, best CV score: {gs.best_score_:.4f}")

        # Saving models
        for name, model in self.trained_models.items():
            file_path = self.save_path / f"{name}.joblib"
            joblib.dump(model, file_path)
            self.logger.info(f"Saving model {model} at {file_path}")
