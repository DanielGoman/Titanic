from src.preprocess.FeatureExtractor import FeatureExtractor
import pandas as pd

from src.RunParameterSearchCV import run


def main():
    dataset_path = 'data/train.csv'
    out_path = 'output/output.txt'

    df = pd.read_csv(dataset_path)

    transforms = [FeatureExtractor.cabin_transform,
                  FeatureExtractor.grouping_type_transform,
                  FeatureExtractor.boolean_sibsp_parch,
                  FeatureExtractor.multiple_cabins_transform,
                  FeatureExtractor.age_groups_transform,
                  FeatureExtractor.drop_columns_transform]

    models = ['AdaBoost', 'GradientBoost', 'XGBoost']   # 'RandomForest', 'SVM'

    scorers = ['roc_auc', 'f1', 'balanced_accuracy']

    # either 'grid' or 'randomized'
    search_type = 'randomized'

    run(df=df,
        model_names=models,
        search_type=search_type,
        scorers=scorers,
        transforms=transforms,
        out_path=out_path,
        n_jobs=-1,
        random_state=42)


if __name__ == '__main__':
    main()
