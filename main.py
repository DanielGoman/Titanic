from src.preprocess.FeatureExtractor import FeatureExtractor
import pandas as pd

from src.RunGridsearchCV import run


def main():
    dataset_path = 'data/train.csv'
    out_path = 'output.txt'

    df = pd.read_csv(dataset_path)

    transforms = [FeatureExtractor.cabin_transform,
                  FeatureExtractor.grouping_type_transform,
                  FeatureExtractor.boolean_sibsp_parch,
                  FeatureExtractor.multiple_cabins_transform,
                  FeatureExtractor.age_groups_transform,
                  FeatureExtractor.drop_columns_transform]

    models = ['AdaBoost', 'GradientBoost', 'XGBoost']   # 'RandomForest', 'SVM'

    scorers = ['roc_auc', 'f1', 'balanced_accuracy']

    run(df=df,
        model_names=models,
        scorers=scorers,
        transforms=transforms,
        out_path=out_path,
        random_state=42)


if __name__ == '__main__':
    main()
