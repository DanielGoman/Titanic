from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class Preprocessor:
    def __init__(self):
        super(Preprocessor, self).__init__()
        self.title_mapper = {'junior': ['Miss.', 'Miss', 'Master.', 'Master'],
                             'senior': ['Ms.', 'Mrs.', 'Ms', 'Mrs', 'Mr.', 'Mr', 'Dr.'],
                             'other': []}

        self.other_title = 'other'

        self.cols_to_drop = ['Name', 'Ticket', 'Cabin']

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Preprocessor:
        print('Preprocessor fit')
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        print('Preprocessor transform')
        df = X.copy()

        transforms = [self.cabin_transform,
                      self.grouping_type_transform,
                      self.boolean_sibsp_parch,
                      self.multiple_cabins_transform,
                      self.age_groups_transform,
                      self.drop_columns_transform]

        out_df = df.copy()
        for func in transforms:
            out_df = func(out_df)

        return out_df

    # split the 'Cabin' feature into 2 features - 'cabin_type' and 'cabin_indexes'
    # 'cabin_type' - the first letter of the 'Cabin' feature
    # 'cabin_indexes' - the room numbers that follow the first letter of the 'Cabin' feature
    def cabin_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cabins = df['Cabin'].dropna().map(lambda x: x.split(' '))

        cabin_types = []
        cabin_indexes = []
        for idx, passenger_cabins in zip(cabins.index, cabins.tolist()):
            cabin_type = passenger_cabins[0][0]
            cabin_idx = []
            for item in passenger_cabins:
                if len(item) > 1:
                    cabin_idx.append(int(item[1:]))

            if len(cabin_idx) > 0:
                cabin_indexes.append(np.ceil(np.mean(cabin_idx)))
                cabin_types.append(cabin_type)
            else:
                cabin_indexes.append(np.nan)
                cabin_types.append(np.nan)

        cabin_df = df.copy()
        cabin_df.loc[cabins.index, 'cabin_type'] = cabin_types
        cabin_df.loc[cabins.index, 'cabin_indexes'] = cabin_indexes

        return cabin_df

    # added 'grouping_type' feature with 4 categories:
    #   parents_with_kids - passenger which is a parent with kids on board
    #   kids_with_parents - passenger which is a kid with parents on board
    #   alone_or_friends - passenger which is either alone or is traveling with friends on board
    #   other - any other unhandled case
    def grouping_type_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parents_idx = df[((df['Age'] > 20) & (df['SibSp'] == 1) & (df['Parch'] > 0)) |
                         ((df['Age'] > 25) & (df['Parch'] > 0))
                         ].index

        kids_idx = df[((df['Age'] <= 20) & (df['Parch'] == 2)) |
                      (df['Age'] <= 18)
                      ].index

        alone_or_friends_idx = df[(df['SibSp'] == 0) & (df['Parch'] == 0)].index

        other_idx = set(df.index) - set(parents_idx) - set(kids_idx) - set(alone_or_friends_idx)

        parents_with_kids = 'parents_with_kids'
        kids_with_parents = 'kids_with_parents'
        alone_or_friends = 'alone_or_friends'
        other = 'other'

        grouping_type_df = df.copy()
        grouping_type_df['grouping_type'] = ''
        grouping_type_df.loc[parents_idx, 'grouping_type'] = parents_with_kids
        grouping_type_df.loc[kids_idx, 'grouping_type'] = kids_with_parents
        grouping_type_df.loc[alone_or_friends_idx, 'grouping_type'] = alone_or_friends
        grouping_type_df.loc[other_idx, 'grouping_type'] = other

        return grouping_type_df

    # added binary 'SibSp' and 'Parch' features
    # i.e. 1 if val > 0, else 0
    def boolean_sibsp_parch(self, df: pd.DataFrame) -> pd.DataFrame:
        boolean_df = df.copy()
        boolean_df['binary_SibSp'] = 0
        boolean_df.loc[boolean_df['SibSp'] > 0, 'binary_SibSp'] = 1

        boolean_df['binary_parch'] = 0
        boolean_df.loc[boolean_df['Parch'] > 0, 'binary_parch'] = 1

        return boolean_df

    # added 'multiple_cabins' feature
    # 1 - if the passenger has more than one cabin
    # 0 - if they have only 1 cabin
    def multiple_cabins_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cabin_count = df['Cabin'].value_counts()
        count_index = cabin_count[cabin_count > 1].index
        more_than_one_in_cabin = df.loc[df['Cabin'].isin(count_index)]

        in_cabin_count_df = df.copy()
        in_cabin_count_df['multiple_cabins'] = 0
        in_cabin_count_df.loc[more_than_one_in_cabin.index, 'multiple_cabins'] = 1

        return in_cabin_count_df

    # added 'age_group' feature according to the title of the passengers
    # mapping can be found in self.title_mapper
    def age_groups_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        age_grp_df = df.copy()

        def title_classifier(name):
            for title in self.title_mapper['junior']:
                if title in name:
                    return 'junior'

            for title in self.title_mapper['senior']:
                if title in name:
                    return 'senior'

            return self.other_title

        age_grp_df['age_group'] = age_grp_df['Name'].apply(lambda x: title_classifier(x))
        return age_grp_df

    def drop_columns_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dropped_df = df.copy()
        dropped_df = dropped_df.drop(self.cols_to_drop, axis=1)

        return dropped_df


class Imputer:
    def __init__(self):
        super(Imputer, self).__init__()
        self.reg_dict = {}
        self.missing_indexes_value = -50
        self.age_groups = ['senior', 'junior']
        self.trained_features = {}

    # TODO: try MICE imputation strategy
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Imputer:
        print('Imputer fit')
        df = X.copy()

        # imputing missing values of the 'age' feature
        for age_group in self.age_groups:
            temp_df = df[df['age_group'] == age_group].copy()
            temp_df = temp_df.drop(['cabin_type', 'cabin_indexes'], axis=1)
            reg = RandomForestRegressor(n_estimators=300,
                                        max_depth=25,
                                        random_state=42)

            temp_df = pd.get_dummies(temp_df)

            X = temp_df.drop('Age', axis=1)
            y = temp_df['Age']
            X_train, y_train = X[y.isna() == False], y[y.isna() == False]
            self.trained_features[age_group] = X.columns    # saving the features used to train the imputation method

            reg.fit(X_train, y_train)

            self.reg_dict[age_group] = reg

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        print('Imputer transform')
        imputed_df = df.copy()

        for age_group in self.age_groups:
            regressor = self.reg_dict[age_group]
            X_missing = df[(df['age_group'] == age_group) &
                           (df['Age'].isna() == True)].copy()

            X_missing = X_missing.drop('age_group', axis=1)
            X_missing = pd.get_dummies(X_missing)

            # dropping features which the imputation model wasn't trained on
            features_to_drop = set(X_missing.columns) - set(self.trained_features[age_group])
            features_to_drop = [item for item in features_to_drop]
            X_missing = X_missing.drop(features_to_drop, axis=1)

            # adding features which the imputation model was trained on and are missing in the test set
            features_to_add = set(self.trained_features[age_group]) - set(X_missing.columns)
            features_to_add = [item for item in features_to_add]
            X_missing[features_to_add] = 0

            # reordering features to match the input order into the imputation model
            X_missing = X_missing[self.trained_features[age_group]]

            y_pred = np.ceil(regressor.predict(X_missing))
            imputed_df.loc[X_missing.index, 'Age'] = y_pred

        imputed_df = self.cabin_indexes_impute(imputed_df)
        imputed_df = self.cabin_type_impute(imputed_df)

        return pd.get_dummies(imputed_df)

    # averaged over the 'cabin_indexes' feature to have a single numeric value
    # missing cells were imputed with a constant value
    def cabin_indexes_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        def avg_cabin_index(item):
            if type(item) == list:
                if len(item) == 0:
                    return self.missing_indexes_value
            return np.mean(item)

        avg_cabin_index_df = df.copy()
        avg_cabin_index_df['cabin_indexes'] = avg_cabin_index_df['cabin_indexes'].fillna(self.missing_indexes_value)
        avg_cabin_index_df['cabin_indexes'] = avg_cabin_index_df['cabin_indexes'].apply(avg_cabin_index)

        return avg_cabin_index_df

    # encoded the missingness of 'cabin_type' using an additional label - 'Missing'
    def cabin_type_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        missingness_label = 'Missing'
        cabin_type_missing_label_df = df.copy()
        missing_idx = cabin_type_missing_label_df['cabin_type'][cabin_type_missing_label_df['cabin_type'].isna()].index

        cabin_type_missing_label_df.loc[missing_idx, 'cabin_type'] = missingness_label

        return cabin_type_missing_label_df


# TODO: choose a method for feature selection and implement
class SelectKBest:
    def __init__(self):
        super(SelectKBest, self).__init__()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        pass

    def transform(self, X: pd.DataFrame):
        pass
