import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class Preprocessor:
    def __init__(self):
        super(Preprocessor, self).__init__()
        self.df = None

        self.title_mapper = {'junior': ['Miss.', 'Miss', 'Master.', 'Master'],
                             'senior': ['Ms.', 'Mrs.', 'Ms', 'Mrs', 'Mr.', 'Mr', 'Dr.'],
                             'other': []}

        self.other_title = 'other'

        self.cols_to_drop = ['Name', 'Ticket', 'Cabin']

    def fit(self, df: pd.Dataframe):
        self.df = df
        return self

    def transform(self):
        transforms = [self.cabin_transform,
                      self.grouping_type_transform,
                      self.boolean_sibsp_parch,
                      self.multiple_cabins_transform,
                      self.grouping_type_transform,
                      self.drop_columns_transform]

        out_df = self.df.copy()
        for func in transforms:
            out_df = func(out_df)

        return out_df

    # split the 'Cabin' feature into 2 features - 'cabin_type' and 'cabin_indexes'
    # 'cabin_type' - the first letter of the 'Cabin' feature
    # 'cabin_indexes' - the room numbers that follow the first letter of the 'Cabin' feature
    def cabin_transform(self, df: pd.DataFrame):
        cabins = self.df['Cabin'].dropna().map(lambda x: x.split(' '))

        cabin_types = []
        cabin_indexes = []
        for passenger_cabins in cabins.tolist():
            cabin_type = passenger_cabins[0][0]
            cabin_idx = []
            for item in passenger_cabins:
                if len(item) > 1:
                    cabin_idx.append(int(item[1:]))

            cabin_indexes.append(cabin_idx)
            cabin_types.append(cabin_type)

        cabin_df = self.df.copy()
        cabin_df['cabin_type'] = np.nan
        cabin_df.loc[cabins.index, 'cabin_type'] = cabin_types

        cabin_df['cabin_indexes'] = np.nan
        cabin_df.loc[cabins.index, 'cabin_indexes'] = cabin_indexes

        return cabin_df

    # added 'grouping_type' feature with 4 categories:
    #   parents_with_kids - passenger which is a parent with kids on board
    #   kids_with_parents - passenger which is a kid with parents on board
    #   alone_or_friends - passenger which is either alone or is traveling with friends on board
    #   other - any other unhandled case
    def grouping_type_transform(self, df: pd.DataFrame):
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
        grouping_type_df['grouping_type'] = ""
        grouping_type_df.loc[parents_idx, 'grouping_type'] = parents_with_kids
        grouping_type_df.loc[kids_idx, 'grouping_type'] = kids_with_parents
        grouping_type_df.loc[alone_or_friends_idx, 'grouping_type'] = alone_or_friends
        grouping_type_df.loc[other_idx, 'grouping_type'] = other

        return grouping_type_df

    # added binary 'SibSp' and 'Parch' features
    # i.e. 1 if val > 0, else 0
    def boolean_sibsp_parch(self, df: pd.DataFrame):
        boolean_df = df.copy()
        boolean_df['binary_SibSp'] = 0
        boolean_df.loc[boolean_df['SibSp'] > 0, 'binary_SibSp'] = 1

        boolean_df['binary_parch'] = 0
        boolean_df.loc[boolean_df['Parch'] > 0, 'binary_parch'] = 1

        return boolean_df

    # added 'multiple_cabins' feature
    # 1 - if the passenger has more than one cabin
    # 0 - if they have only 1 cabin
    def multiple_cabins_transform(self, df: pd.DataFrame):
        cabin_count = df['Cabin'].value_counts()
        count_index = cabin_count[cabin_count > 1].index
        more_than_one_in_cabin = df.loc[df['Cabin'].isin(count_index)]

        in_cabin_count_df = df.copy()
        in_cabin_count_df['multiple_cabins'] = pd.Series(data=np.zeros(in_cabin_count_df.shape[0]),
                                                         index=in_cabin_count_df.index, dtype=int)
        in_cabin_count_df.loc[more_than_one_in_cabin.index, 'multiple_cabins'] = 1

        return in_cabin_count_df

    # added 'age_group' feature according to the title of the passengers
    # mapping can be found in self.title_mapper
    def age_groups_transform(self, df: pd.DataFrame):
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

    def drop_columns_transform(self, df: pd.DataFrame):
        dropped_df = df.copy()
        dropped_df = dropped_df.drop(self.cols_to_drop, axis=1)

        return dropped_df


class Imputer:
    def __init__(self, df: pd.DataFrame):
        super(Imputer, self).__init__()
        self.df = df
        self.rf_dict = {}
        self.missing_indexes_value = -50

    def fit(self, df: pd.Dataframe):
        # imputing missing values of the 'age' feature
        for age_group in ['senior', 'junior']:
            temp_df = df[df['age_group'] == age_group].copy()
            temp_df = temp_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
            reg = RandomForestRegressor(n_estimators=300,
                                        max_depth=25,
                                        random_state=42)
            temp_df = pd.get_dummies(temp_df)

            X = temp_df.drop('Age', axis=1)
            y = temp_df['Age']

            X_train, y_train = X[y.isna() == False], y[y.isna() == False]
            X_test = X[y.isna() == True]

            reg.fit(X_train, y_train)
            self.rf_dict[age_group] = (reg, X_test)

        return self

    def transform(self):
        imputed_df = self.df.copy()
        for (reg, X_test) in self.rf_dict.values():
            y_pred = np.ceil(reg.predict(X_test))
            imputed_df.loc[X_test.index, 'Age'] = y_pred

        return imputed_df

    # averaged over the 'cabin_indexes' feature to have a single numeric value
    # missing cells were imputed with a constant value
    def cabin_indexes_impute(self, df: pd.DataFrame):
        def avg_cabin_index(item):
            if item is np.nan or len(item) == 0:
                return self.missing_indexes_value
            return np.mean(item)

        avg_cabin_index_df = df.copy()
        avg_cabin_index_df['cabin_indexes'] = avg_cabin_index_df['cabin_indexes'].apply(avg_cabin_index).astype(int)

        return avg_cabin_index_df

    # encoded the missingness of 'cabin_type' using an additional label - 'Missing'
    def cabin_type_impute(self, df: pd.DataFrame):
        missingness_label = 'Missing'
        cabin_type_missing_label_df = df.copy()
        missing_idx = cabin_type_missing_label_df['cabin_type'][cabin_type_missing_label_df['cabin_type'].isna()].index

        cabin_type_missing_label_df.loc[missing_idx, 'cabin_type'] = missingness_label

        return cabin_type_missing_label_df


class SelectKBest:
    def __init__(self):
        super(SelectKBest, self).__init__()

    def fit(self):
        pass

    def transform(self):
        pass
