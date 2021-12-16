from __future__ import annotations

import pandas as pd
import numpy as np


class FeatureExtractor:
    """This class extracts and transforms features

    This class applies various functions, each either transforming
    an existent feature into a usable feature which can be fed into the models
    or extracting new features from the existent ones

    Attributes:
        title_mapper: dict
          a dictionary that specifies which title (which can be found in the
          feature 'name') belongs to each 'age_group'
        other_title: str
          a label for the remaining titles that aren't included in title_mapper
        cols_to_drop: list
          a list of column names which should be dropped once all the transformations
          are done
    """
    title_mapper = {'junior': ['Miss.', 'Miss', 'Master.', 'Master'],
                    'senior': ['Ms.', 'Mrs.', 'Ms', 'Mrs', 'Mr.', 'Mr', 'Dr.']
                    }

    other_title = 'other'

    cols_to_drop = ['Name', 'Ticket', 'Cabin']

    def __init__(self, transforms=None):
        self.transforms = transforms

    def fit(self, X: pd.DataFrame, y: pd.Series) -> FeatureExtractor:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input dataframe

        Iteratively applies all the transformations that the class
        was given when initialized through 'self.transforms'

        Args:
            X: pd.DataFrame
              the dataframe to be transformed

        Returns:
            out_df: pd.DataFrame
              the transformed dataframe
        """
        df = X.copy()

        out_df = df.copy()
        for func in self.transforms:
            out_df = func(out_df)

        return out_df

    @staticmethod
    def cabin_transform(df: pd.DataFrame) -> pd.DataFrame:
        """Processes the 'Cabin' feature

        Creates 2 new features using the 'Cabin' feature:
            'cabin_type' - the first letter of the 'Cabin' feature
            'cabin_indexes' - the room numbers that follows the first letter of the 'Cabin' feature

        Args:
            df: pd.DataFrame
              the dataframe to be transformed

        Returns:
            cabin_df: pd.DataFrame
             the transformed dataframe
        """
        cabins = df['Cabin'].dropna()
        cabins = cabins.map(lambda x: x.split(' '))

        cabin_types = []
        cabin_indexes = []
        for idx, passenger_cabins in cabins.iteritems():
            cabin_type = passenger_cabins[0][0]
            cabin_idx = [int(item[1:]) for item in passenger_cabins if len(item) > 1]

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

    @staticmethod
    def grouping_type_transform(df: pd.DataFrame) -> pd.DataFrame:
        """Creates the feature 'grouping type'

        Heuristically creates a new feature 'grouping type' with the following labels:
            parents_with_kids - passenger which is a parent with kids on board
            kids_with_parents - passenger which is a kid with parents on board
            alone_or_friends - passenger which is either alone or is traveling with friends on board
            other - any other unhandled case

        Args:
            df: pd.DataFrame
              the dataframe to be transformed

        Returns:
            trouping_type_df: pd.DataFrame
              the transformed dataframe
        """
        parents_idx = df[((df['Age'] > 20) & (df['SibSp'] == 1) & (df['Parch'] > 0)) |
                         ((df['Age'] > 25) & (df['Parch'] > 0))
                         ].index

        kids_idx = df[((df['Age'] <= 20) & (df['Parch'] == 2)) |
                      (df['Age'] <= 18)
                      ].index

        alone_or_friends_idx = df[(df['SibSp'] == 0) & (df['Parch'] == 0)].index

        other_idx = set(df.index) - set(parents_idx) - set(kids_idx) - set(alone_or_friends_idx)

        parents_with_kids_label = 'parents_with_kids'
        kids_with_parents_label = 'kids_with_parents'
        alone_or_friends_label = 'alone_or_friends'
        other_label = 'other'

        grouping_type_df = df.copy()
        grouping_type_df['grouping_type'] = ''
        grouping_type_df.loc[parents_idx, 'grouping_type'] = parents_with_kids_label
        grouping_type_df.loc[kids_idx, 'grouping_type'] = kids_with_parents_label
        grouping_type_df.loc[alone_or_friends_idx, 'grouping_type'] = alone_or_friends_label
        grouping_type_df.loc[other_idx, 'grouping_type'] = other_label

        return grouping_type_df

    @staticmethod
    def boolean_sibsp_parch(df: pd.DataFrame) -> pd.DataFrame:
        """Creates boolean features out of numerics features

        Creates binary features of 'SibSp' and 'Parch'.
        'binary_SibSp' and 'binary_parch' respectively.

        Args:
            df: pd.DataFrame
              the dataframe to be transformed

        Returns:
            boolean_df: pd.DataFrame
              the transformed dataframe
        """
        boolean_df = df.copy()
        boolean_df['binary_SibSp'] = 0
        boolean_df.loc[boolean_df['SibSp'] > 0, 'binary_SibSp'] = 1

        boolean_df['binary_parch'] = 0
        boolean_df.loc[boolean_df['Parch'] > 0, 'binary_parch'] = 1

        return boolean_df

    @staticmethod
    def multiple_cabins_transform(df: pd.DataFrame) -> pd.DataFrame:
        """Creates the 'multiple_cabins' feature

        Creates the 'multiple_cabins' feature, which specifies if in the dataset
        more than one person had been assigned to the same set of cabins

        Args:
            df: pd.DataFrame
              the dataframe to be transformed

        Returns:
            in_cabin_count_df: pd.DataFrame
              the transformed dataframe
        """
        in_cabin_count_df = df.copy()

        cabin_count = df['Cabin'].value_counts()
        count_index = cabin_count[cabin_count > 1].index
        more_than_one_in_cabin = df.loc[df['Cabin'].isin(count_index)]

        in_cabin_count_df['multiple_cabins'] = 0
        in_cabin_count_df.loc[more_than_one_in_cabin.index, 'multiple_cabins'] = 1

        return in_cabin_count_df

    @staticmethod
    def age_groups_transform(df: pd.DataFrame) -> pd.DataFrame:
        """Creates the 'age_group' feature

        Credit to Moshe Nahs for the idea ;)

        Uses the 'title_mapper' dict to map each person in the dataset
        to a certain age group according to their title, found in the 'name' feature

        Args:
            df: pd.DataFrame
              the dataframe to be transformed

        Returns:
            age_grp_df: pd.DataFrame
              the transformed dataframe
        """
        age_grp_df = df.copy()

        def title_classifier(name):
            for title in FeatureExtractor.title_mapper['junior']:
                if title in name:
                    return 'junior'

            for title in FeatureExtractor.title_mapper['senior']:
                if title in name:
                    return 'senior'

            return FeatureExtractor.other_title

        age_grp_df['age_group'] = age_grp_df['Name'].apply(lambda x: title_classifier(x))
        return age_grp_df

    @staticmethod
    def drop_columns_transform(df: pd.DataFrame) -> pd.DataFrame:
        dropped_df = df.copy()
        dropped_df = dropped_df.drop(FeatureExtractor.cols_to_drop, axis=1)

        return dropped_df

