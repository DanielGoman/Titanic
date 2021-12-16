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
        model_names = kargs.pop('model_names')
        if os.path.exists(kargs['out_path']):
            os.remove(kargs['out_path'])

        for model_name in model_names:
            print(f'Running {model_name}:')
            func(model_name=model_name, **kargs)

    return run_loop


def runtime_counter(func):
    def timer(**kargs):
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
def run(df: pd.DataFrame, model_name, scorers, transforms: list, random_state: int):
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

    grid = GridSearchCV(pipe, params, cv=5, scoring=scorers, refit='roc_auc', verbose=1, n_jobs=-1)
    grid.fit(X, y)

    return grid.cv_results_, grid.best_params_
