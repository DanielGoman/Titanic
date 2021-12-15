from __future__ import annotations
import pandas as pd


# TODO: choose a method for feature selection and implement it
class SelectKBest:
    numerical_features = ['age_group', 'Sex', 'Pclass', 'cabin_type', 'Embarked', 'multiple_cabins', 'grouping_type']
    categorial_features = ['cabin_indexes', 'Fare', 'Age', 'Parch', 'SibSp']

    def __init__(self):
        self.numerical_features_num = None
        self.categorial_features_num = None

    def set_params(self, **params: dict) -> SelectKBest:
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SelectKBest:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        selected_numerical_features = SelectKBest.numerical_features[:getattr(self, 'numerical_features_num')]
        selected_categorial_features = SelectKBest.categorial_features[:getattr(self, 'categorial_features_num')]

        selected_features = selected_numerical_features + selected_categorial_features
        return X[selected_features]
