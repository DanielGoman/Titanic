from preprocess import Preprocessor, Imputer, SelectKBest
from model import Model

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def main():
    df = pd.read_csv('data/train.csv')

    pipe = Pipeline([('Preprocessor', Preprocessor()),
                     ('Imputer', Imputer()),
                     ('Model', Model())
                     ])

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    params = {}
    params['Imputer__n_estimators'] = [100, 200, 300]
    params['Imputer__max_depth'] = [10, 15, 20, 25]

    grid = GridSearchCV(pipe, params, cv=5, scoring='roc_auc')
    grid.fit(X, y)

    print(f'Best score: {grid.best_score_}')
    print(f'Best params: {grid.best_params_}')


if __name__ == '__main__':
    main()
