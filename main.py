from src.preprocess.FeatureExtractor import FeatureExtractor
import pandas as pd

from src.RunGridsearchCV import run

# TODO: try more metrics
#       add readme


def main():
    df = pd.read_csv('data/train.csv')

    transforms = [FeatureExtractor.cabin_transform,
                  FeatureExtractor.grouping_type_transform,
                  FeatureExtractor.boolean_sibsp_parch,
                  FeatureExtractor.multiple_cabins_transform,
                  FeatureExtractor.age_groups_transform,
                  FeatureExtractor.drop_columns_transform]

    run(df=df,
        model_name='GradientBoost',
        transforms=transforms)


if __name__ == '__main__':
    main()
