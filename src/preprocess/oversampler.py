from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTENC


class Oversampler:
    categorial_features = ['Embarked', 'Pclass', 'Sex', 'age_group', 'cabin_type',
                           'binary_SinSp', 'binary_parch', 'grouping_type', 'multiple_cabins']
    target_feature = 'Survived'
    missing_features = ['age', 'Cabin']

    def __init__(self, get_as_numpy):
        self.get_as_numpy = get_as_numpy

    def fit_resample(self, X: pd.DataFrame, y: pd.Series):
        categorial_columns_indexes = []
        for i, col in enumerate(X.columns):
            for categorial_feature in Oversampler.categorial_features:
                if col.startswith(categorial_feature):
                    categorial_columns_indexes.append(i)

        oversampler = SMOTENC(categorical_features=categorial_columns_indexes, random_state=42)

        oversampled_X, oversampled_y = oversampler.fit_resample(X.to_numpy(), y)
        if not self.get_as_numpy:
            oversampled_X = pd.DataFrame(oversampled_X, columns=X.columns)
        return oversampled_X, oversampled_y
