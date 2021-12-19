from __future__ import annotations

import itertools

import pandas as pd
from imblearn.over_sampling import SMOTENC


class Oversampler:
    """Handles oversampling

    This class performs SMOTE oversampling over the transformed
    and imputed input dataframe

    Attributes:
        categorial_features: list
          list of categorial features in the transformed dataset
        target_feature: str
          The name of the feature we're trying to predict
    """
    categorial_features = ['Embarked', 'Pclass', 'Sex', 'age_group', 'cabin_type',
                           'binary_SinSp', 'binary_parch', 'grouping_type', 'multiple_cabins']
    target_feature = 'Survived'

    def __init__(self):
        pass

    @staticmethod
    def fit_resample(X: pd.DataFrame, y: pd.Series):
        """Oversamples the dataset

        Uses SMOTE to oversample dataset to balance the ratio
        among classes of the target feature

        Args:
            X: pd.DataFrame
              the transformed and imputed input dataframe
            y: pd.Series
              the corresponding labels for X

        Returns:
            oversampled_X: np.ndarray
              array of the oversampled independent features
            oversampled_Y: np.ndarray
              vector of the oversampled target feature, corresponding to oversampled_X
        """

        # maps
        categorial_columns_indexes = list({index for categorial_feature, (index, col_name) in
                                          itertools.product(Oversampler.categorial_features, enumerate(X.columns))
                                          if col_name.startswith(categorial_feature)})

        oversampler = SMOTENC(categorical_features=categorial_columns_indexes, random_state=42)

        oversampled_X, oversampled_y = oversampler.fit_resample(X.to_numpy(), y)
        oversampled_X = pd.DataFrame(oversampled_X, columns=X.columns)
        return oversampled_X, oversampled_y
