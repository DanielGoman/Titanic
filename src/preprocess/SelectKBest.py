from __future__ import annotations
import pandas as pd
from sklearn.feature_selection import RFE


class SelectKBest:
    def __init__(self, estimator, get_as_numpy):
        self.estimator = estimator
        self.get_as_numpy = get_as_numpy
        self.rfe = None

    def set_params(self, **params: dict) -> SelectKBest:
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SelectKBest:
        input_X = X.copy()
        if self.get_as_numpy:
            input_X = X.to_numpy()
        self.estimator = self.estimator.fit(input_X, y)
        self.rfe = RFE(self.estimator, n_features_to_select=getattr(self, 'n_features_to_select'))
        self.rfe.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = self.rfe.transform(X)
        return X_transformed
