# pipeline components
from src.preprocess.FeatureExtractor import FeatureExtractor
from src.preprocess.MissingFeaturesImputer import MissingFeaturesImputer
from src.preprocess.SelectKBest import SelectKBest
from src.preprocess.oversampler import Oversampler

import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import time
import os


# TODO: add a graph to compare the performance of the models over different metrics


def run_all_models(func):
    def run_loop(**kargs):
        """Decorator that runs all chosen models

        Empties the output file and iteratively calls runs a grid search cv
        for each of the models.

        Args:
            **kargs:
              dictionary of key worded parameters to pass to func
        """
        model_names = kargs.pop('model_names')
        if os.path.exists(kargs['out_path']):
            os.remove(kargs['out_path'])

        for model_name in model_names:
            print(f'Running {model_name}:')
            func(model_name=model_name, **kargs)

    return run_loop


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


def set_logger(func):
    def logger(**kargs):
        """Logger decorator

        Writes output to both the terminal and an output file

        Args:
            **kargs:
              dictionary of key worded parameters to pass to func
        """
        out_path = kargs.pop('out_path')
        model_name = kargs['model_name']
        scorers = kargs['scorers']

        score, best_params = func(**kargs)

        with open(out_path, 'a') as f:
            f.write(f'{model_name} scores:\n')
            for scorer in scorers:
                score_str = f'\t{scorer} score: {round(score[f"mean_test_{scorer}"][0], 4)}'
                f.write(f'{score_str}\n')
                print(score_str)

            f.write(f'\n\t{model_name} best params:\n')
            print(f'{model_name} best params:')
            for param_name, param_value in best_params.items():
                best_params_str = f'\t\t{param_name} = {param_value}'
                f.write(f'{best_params_str}\n')
                print(best_params_str)

            f.write('\n\n')

    return logger


def get_model_config(model_name: str, random_state: int) -> (None, dict, bool):
    """Model parameters configuration function

    For each model configures the chosen parameters to use
    for the grid search cv

    Args:
        model_name: str
          dictionary of key worded parameters to pass to func
        random_state: int
          random state parameter for all of the components of the pipeline

    Returns:
        Model:
          the model component for the pipeline
        params: dict
          dictionary of parameters for the model component
    """
    params = dict()
    if model_name == 'RandomForest':
        params['Model__n_estimators'] = [150, 300]
        params['Model__max_depth'] = [5, 7]
        params['Model__min_samples_split'] = [2, 3]
        return RandomForestClassifier(random_state=random_state), params
    if model_name == 'AdaBoost':
        params['Model__n_estimators'] = [200, 300]
        params['Model__learning_rate'] = [0.1, 0.2]
        return AdaBoostClassifier(random_state=random_state), params
    if model_name == 'GradientBoost':
        params['Model__min_samples_split'] = [2]
        params['Model__max_depth'] = [3, 5]
        params['Model__learning_rate'] = [0.05, 0.1]
        params['Model__n_estimators'] = [300, 400]
        return GradientBoostingClassifier(random_state=random_state), params
    elif model_name == 'XGBoost':
        params['Model__eta'] = [1]  # [1, 2]
        params['Model__gamma'] = [1, 2]
        params['Model__max_depth'] = [4, 6]
        return XGBClassifier(use_label_encoder=False, random_state=random_state, verbosity=0), params
    elif model_name == 'SVM':
        params['Model__kernel'] = ['rbf']
        params['Model__degree'] = [2]
        return SVC(random_state=random_state), params


@run_all_models
@runtime_counter
@set_logger
def run(df: pd.DataFrame, model_name: str, scorers: list, transforms: list, n_jobs: int, random_state: int):
    """Runs the grid search cv

    Runs the grid search cv over the given model and parameters

    Args:
        df: pd.DataFrame
          the input dataframe, containing both X and y
        model_name: str
          the name of the model to be trained in this run
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
    model, params = get_model_config(model_name, random_state)

    pipe = Pipeline([('FeatureExtractor', FeatureExtractor(transforms)),
                     ('MissingFeaturesImputer', MissingFeaturesImputer()),
                     ('Oversampler', Oversampler()),
                     ('SelectKBest', SelectKBest(model)),
                     ('Model', model)
                     ])

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    params['MissingFeaturesImputer__n_estimators'] = [75, 150]
    params['MissingFeaturesImputer__max_depth'] = [10, 15]
    params['SelectKBest__n_features_to_select'] = [0.5, 0.75, 1]

    grid = GridSearchCV(pipe=pipe,
                        params=params,
                        cv=5,
                        scoring=scorers,
                        refit='roc_auc',
                        verbose=1,
                        n_jobs=n_jobs)
    grid.fit(X, y)

    return grid.cv_results_, grid.best_params_
