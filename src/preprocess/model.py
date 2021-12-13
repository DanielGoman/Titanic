from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# TODO: implement model to try:
#          try: SVM, Gradient Boosted Trees, XGBoost

class Model(BaseEstimator, ClassifierMixin):
    def __init__(self):
        super().__init__()
        self.model = GradientBoostingClassifier()
        self.classes_ = [0, 1]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Model:
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.decision_function(X)
