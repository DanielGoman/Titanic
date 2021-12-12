from src.model import Model
from src.FeatureExtractor import FeatureExtractor
from src.MissingFeaturesImputer import MissingFeaturesImputer
from src.SelectKBest import SelectKBest

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def main():
    df = pd.read_csv('data/train.csv')

    transforms = [FeatureExtractor.cabin_transform,
                  FeatureExtractor.grouping_type_transform,
                  FeatureExtractor.boolean_sibsp_parch,
                  FeatureExtractor.multiple_cabins_transform,
                  FeatureExtractor.age_groups_transform,
                  FeatureExtractor.drop_columns_transform]

    pipe = Pipeline([('FeatureExtractor', FeatureExtractor(transforms)),
                     ('MissingFeaturesImputer', MissingFeaturesImputer()),
                     ('Model', Model())
                     ])

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    params = {}
    params['MissingFeaturesImputer__n_estimators'] = [300]
    params['MissingFeaturesImputer__max_depth'] = [25]

    grid = GridSearchCV(pipe, params, cv=5, scoring='roc_auc')
    grid.fit(X, y)

    print(f'Best score: {grid.best_score_}')
    print(f'Best params: {grid.best_params_}')


if __name__ == '__main__':
    main()
