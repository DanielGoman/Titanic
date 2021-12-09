import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import numpy as np


#TODO: implement the model to use:
#          try: SVM, SVM with kernels, Gradient Boosted Trees, XGBoost

class Model:
    def __init__(self):
        self.model = GradientBoostingClassifier()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
