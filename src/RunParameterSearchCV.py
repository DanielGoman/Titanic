# pipeline components
from src.preprocess.FeatureExtractor import FeatureExtractor
from src.preprocess.MissingFeaturesImputer import MissingFeaturesImputer
from src.preprocess.SelectKBest import SelectKBest
from src.preprocess.oversampler import Oversampler
from src.decorators import set_logger, run_all_models

# model selection
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.base import ClassifierMixin

# basic data packages
import pandas as pd

# utility
import time
import timeit


# TODO: add a graph to compare the performance of the models over different metrics


def runtime_counter(func):
    def timer(**kargs):
        """Timer decorator

        Counts the runtime of the grid search over each model

        Args:
            **kargs:
              dictionary of key worded parameters to pass to func
        """
        start_time = time.time()
        func(**kargs)
        end_time = time.time()

        time_diff = end_time - start_time
        minutes = int(time_diff // 60)
        seconds = int(time_diff % 60)
        print(f'Total run time: {minutes} minutes and {seconds} seconds\n')

    return timer


def get_model_config(model_name: str) -> dict:
    """Model parameters configuration function

    For each model configures the chosen parameters to use
    for the grid search cv

    Args:
        model_name: str
          dictionary of key worded parameters to pass to func

    Returns:
        params: dict
          dictionary of parameters for the model component
    """
    model_params = dict()
    if model_name == 'RandomForest':
        model_params['Model__n_estimators'] = [150, 300]
        model_params['Model__max_depth'] = [5, 7]
        model_params['Model__min_samples_split'] = [2]
    elif model_name == 'AdaBoost':
        model_params['Model__n_estimators'] = [200, 300]
        model_params['Model__learning_rate'] = [0.1, 0.15]
    elif model_name == 'GradientBoost':
        model_params['Model__min_samples_split'] = [2]
        model_params['Model__max_depth'] = [3, 5]
        model_params['Model__learning_rate'] = [0.05, 0.1]
        model_params['Model__n_estimators'] = [300, 400]
    elif model_name == 'XGBoost':
        model_params['Model__eta'] = [1, 2]
        model_params['Model__gamma'] = [1, 2]
        model_params['Model__max_depth'] = [4]
    elif model_name == 'SVM':
        model_params['Model__kernel'] = ['rbf', 'poly']
        model_params['Model__degree'] = [2, 3]

    return model_params


def get_initialized_model(model_name: str, random_state: int) -> ClassifierMixin:
    """Model initialization function

    Initializes the desired model with its respective parameters

    Args:
        model_name: str
          dictionary of key worded parameters to pass to func
        random_state: int
          random state parameter for all of the components of the pipeline

    Returns:
        Model:
          the model component for the pipeline, initialized
    """
    if model_name == 'RandomForest':
        return RandomForestClassifier(random_state=random_state)
    if model_name == 'AdaBoost':
        return AdaBoostClassifier(random_state=random_state)
    if model_name == 'GradientBoost':
        return GradientBoostingClassifier(random_state=random_state)
    elif model_name == 'XGBoost':
        return XGBClassifier(use_label_encoder=False, random_state=random_state, verbosity=0)
    elif model_name == 'SVM':
        return SVC(random_state=random_state)
    else:
        raise ValueError(f"Invalid model name {model_name} for parameter 'model_name'")


def add_pipeline_params(model_params: dict) -> dict:
    """Addition of pipeline parameters

    Adds to the parameter grid parameters of the pipeline

    Args:
        model_params: dict
          dict of parameters of the selected model
    Returns:
        pipe_params: dict
          dict of parameters of the selected model and of the pipeline
    """
    pipe_params = model_params.copy()
    pipe_params['MissingFeaturesImputer__n_estimators'] = [75, 150]
    pipe_params['MissingFeaturesImputer__max_depth'] = [10, 15]
    pipe_params['SelectKBest__n_features_to_select'] = [0.5, 0.75]
    return pipe_params


@run_all_models
@runtime_counter
@set_logger
def run(df: pd.DataFrame, model_name: str, search_type: str, scorers: list,
        transforms: list, n_jobs: int, random_state: int):
    """Runs the grid search cv

    Runs the grid search cv over the given model and parameters

    Args:
        df: pd.DataFrame
          the input dataframe, containing both X and y
        model_name: str
          the name of the model to be trained in this run
        search_type: str
          the type of the searchCV to be used - either GridSearchCV or RandomizedSearchCV
        scorers: list
          list of scoring functions to evaluate the performance of the model
        transforms: list
          list of selected transformations to apply to the data
        n_jobs: int
         number of jobs to run in parallel. -1 means using all processors
        random_state: int
          random state parameter for all of the components of the pipeline

    Returns:
        grid.cv_results_: dict
          dictionary that maps scoring function's name
          to the score the model had scored
        grid.best_params_: dict
          dictionary that maps parameter name to its
          optimal value found by the grid search cv

    """
    model = get_initialized_model(model_name, random_state)
    model_params = get_model_config(model_name)

    pipe = Pipeline([('FeatureExtractor', FeatureExtractor(transforms)),
                     ('MissingFeaturesImputer', MissingFeaturesImputer()),
                     ('Oversampler', Oversampler()),
                     ('SelectKBest', SelectKBest(model)),
                     ('Model', model)
                     ])

    pipe_params = add_pipeline_params(model_params)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    search_params = {'estimator': pipe,
                     'cv': 5,
                     'scoring': scorers,
                     'refit': 'roc_auc',
                     'verbose': 1,
                     'n_jobs': n_jobs,
                     }

    if search_type == 'grid':
        grid = GridSearchCV(param_grid=pipe_params, **search_params)
    elif search_type == 'randomized':
        grid = RandomizedSearchCV(param_distributions=pipe_params, **search_params)
    else:
        raise ValueError(f"Invalid type of search {search_type} for parameter 'search_type'")

    grid.fit(X, y)

    return grid.cv_results_, grid.best_params_
