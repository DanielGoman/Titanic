from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from itertools import chain


class MissingFeaturesImputer:
    """Performs missing data imputations

    This class imputes all missing data that there is in the given dataset

    Attributes:
        missing_index_value: int
          a value to impute for the feature of 'cabin_indexes'
        absence_label: str
          a label to impute for the feature of 'cabin_type'
        age_groups: list
          list of available labels of the feature 'age_group'
    """
    missing_indexes_value = -50
    absence_label = 'Missing'
    age_groups = ['senior', 'junior']

    def __init__(self):
        self.regressors_dict = {}
        self.trained_features = {}
        self.n_estimators = None
        self.max_depth = None

    def set_params(self, **params: dict) -> MissingFeaturesImputer:
        for key, val in params.items():
            setattr(self, key, val)
        return self

    # TODO: try MICE imputation strategy
    def fit(self, X: pd.DataFrame, y: pd.Series) -> MissingFeaturesImputer:
        """Fits the imputation model

        Trains a model to predict the 'age' feature, one per each 'age_group' category

        Args:
            X: pd.DataFrame
              the dataframe of independent features to be imputed
            y: pd.Series
              a series of the labels, corresponding to X

        Returns:
            self: MissingFeatureImputer
              returns itself, as the sklearn.Pipeline API requires
        """
        df = X.copy()

        # imputing missing values of the 'age' feature
        for age_group in self.age_groups:
            age_group_df = df[df['age_group'] == age_group].copy()
            age_group_df = age_group_df.drop(['cabin_type', 'cabin_indexes'], axis=1)
            regressor = RandomForestRegressor(n_estimators=getattr(self, 'n_estimators'),
                                              max_depth=getattr(self, 'max_depth'),
                                              random_state=42)

            dummies_df = pd.get_dummies(age_group_df)

            X = dummies_df.drop('Age', axis=1)
            y = dummies_df['Age']
            X_train, y_train = X[y.isna() == False], y[y.isna() == False]
            self.trained_features[age_group] = X.columns  # saving the features used to train the imputation method

            regressor.fit(X_train, y_train)

            self.regressors_dict[age_group] = regressor

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Imputes the given dataframe

        Performs all the imputations required.
        For the 'age' feature, for each label of 'age_group' uses the respective model
        trained in the fit stage to predict the missing values of 'age'

        Args:
            X: pd.DataFrame
              the dataframe to be imputed

        Returns:
            adjusted_features_df: pd.DataFrame
              the imputed dataframe
        """
        df = X.copy()
        imputed_df = df.copy()

        for age_group in self.age_groups:
            regressor = self.regressors_dict[age_group]
            X_missing = df[(df['age_group'] == age_group) &
                           (df['Age'].isna() == True)].copy()

            X_missing = X_missing.drop('age_group', axis=1)
            X_missing = pd.get_dummies(X_missing)

            # dropping features which the imputation model wasn't trained on
            X_missing = self.adjust_features(X_missing, self.trained_features[age_group])

            y_pred = np.ceil(regressor.predict(X_missing))
            imputed_df.loc[X_missing.index, 'Age'] = y_pred

        imputed_df = self.cabin_indexes_impute(imputed_df)
        imputed_df = self.cabin_type_impute(imputed_df)

        dummies_df = pd.get_dummies(imputed_df)
        trained_features = list(chain.from_iterable(self.trained_features.values()))
        adjusted_features_df = self.adjust_features(dummies_df, trained_features)
        return adjusted_features_df

    def cabin_indexes_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Imputes the cabin_indexes feature

        Imputes the cabin_indexes feature, first by averaging over the
        values of the list (for each entry that is not missing)
        and then filling the missing entries with a value stored in the
        class attribute missing_indexes_value

        Args:
            df: pd.DataFrame
             the dataframe to be imputed

        Returns:
            avg_cabin_index_df: pd.DataFrame
              the imputed dataframe
        """
        def avg_cabin_index(item):
            if not item:
                return self.missing_indexes_value
            return np.mean(item)

        avg_cabin_index_df = df.copy()
        avg_cabin_index_df['cabin_indexes'] = avg_cabin_index_df['cabin_indexes'].fillna(self.missing_indexes_value)
        avg_cabin_index_df['cabin_indexes'] = avg_cabin_index_df['cabin_indexes'].apply(avg_cabin_index)

        return avg_cabin_index_df

    @staticmethod
    def cabin_type_impute(df: pd.DataFrame) -> pd.DataFrame:
        """Imputes the cabin_type feature

        Imputes the cabin_type feature by filling the missing entries with
        the label stored in the class attribute absence_label

        Args:
            df: pd.DataFrame
             the dataframe to be imputed

        Returns:
            avg_cabin_index_df: pd.DataFrame
              the imputed dataframe
        """
        cabin_type_missing_label_df = df.copy()
        missing_idx = cabin_type_missing_label_df['cabin_type'][cabin_type_missing_label_df['cabin_type'].isna()].index

        cabin_type_missing_label_df.loc[missing_idx, 'cabin_type'] = MissingFeaturesImputer.absence_label

        return cabin_type_missing_label_df

    @staticmethod
    def adjust_features(df: pd.DataFrame, trained_features: list) -> pd.DataFrame:
        """Removes redundant features, adds missing features

        Adjusts the columns of the df for the DataFrame of val/test sets
        This is needed as we're not guaranteed to get the exact same subset
        of features when using pd.get_dummies on the val/test set, since not all
        values that appear in one df necessarily appear in the other df.

        Args:
            df: pd.DataFrame
              the dataframe who's set of features needs to be adjusted
            trained_features: list
              the set of feature present during the training of the regressor models
        Returns:
            X_missing: pd.DataFrame
              the input df, after his features were adjusted
        """
        X_missing = df.copy()
        # dropping features which the imputation model wasn't trained on
        features_to_drop = list(set(X_missing.columns) - set(trained_features))
        X_missing = X_missing.drop(features_to_drop, axis=1)

        # adding features which the imputation model was trained on and are missing in the test set
        features_to_add = list(set(trained_features) - set(X_missing.columns))
        X_missing[features_to_add] = 0

        # reordering features to match the input order into the imputation model
        X_missing = X_missing[trained_features]

        return X_missing
