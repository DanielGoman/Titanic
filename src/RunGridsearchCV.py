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
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import time


def runtime_counter(func):
    def timer(**kargs):
        start_time = time.time()
        func(**kargs)
        end_time = time.time()

        time_diff = end_time - start_time
        minutes = int(time_diff // 60)
        seconds = int(time_diff % 60)
        print(f'Total run time: {minutes} minutes and {seconds} seconds')

    return timer


def get_model_config(model_name):
    params = dict()
    if model_name == 'GradientBoost':
        params['Model__min_samples_split'] = [2, 3, 4, 5]
        params['Model__max_depth'] = [3, 5, 7]
        params['Model__learning_rate'] = [0.05, 0.1, 0.2, 0.3]
        params['Model__n_estimators'] = [100, 200, 300]
        return GradientBoostingClassifier(random_state=42), params, False
    elif model_name == 'XGBoost':
        params['Model__eta'] = [1]  # [1, 2, 3]
        params['Model__gamma'] = [1, 2, 3]
        params['Model__max_depth'] = [6]    # [4, 6, 8]
        return XGBClassifier(use_label_encoder=False, random_state=42), params, True
    elif model_name == 'SVM':
        params['Model__kernel'] = ['rbf', 'poly', 'sigmoid']
        params['Model__degree'] = [2, 3, 4]
        return SVC(random_state=42), params, False


@runtime_counter
def run(df: pd.DataFrame, model_name, transforms: list):
    model, params, get_as_numpy = get_model_config('GradientBoost')

    pipe = Pipeline([('FeatureExtractor', FeatureExtractor(transforms)),
                     ('SelectKBest', SelectKBest()),
                     ('MissingFeaturesImputer', MissingFeaturesImputer()),
                     ('Oversampler', Oversampler(get_as_numpy)),
                     ('Model', model)
                     ])

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    params['MissingFeaturesImputer__n_estimators'] = [100]
    params['MissingFeaturesImputer__max_depth'] = [15]
    params['SelectKBest__numerical_features_num'] = [5]
    params['SelectKBest__categorial_features_num'] = [5]

    scorers = ['roc_auc', 'f1']
    grid = GridSearchCV(pipe, params, cv=5, scoring=scorers, refit='roc_auc')
    grid.fit(X, y)

    for scorer in scorers:
        print(f'{scorer} score: {round(grid.cv_results_[f"mean_test_{scorer}"][0], 4)}')

    print('Best params:')
    for param_name, param_value in grid.best_params_.items():
        print(f'\t{param_name} = {param_value}')
