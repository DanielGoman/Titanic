from preprocess import Preprocessor, Imputer, SelectKBest
from model import Model

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *

test_size = 0.3

# TODO: move project to a new branch
#       issue a pull request to have Moshe do CR on the code


def main():
    df = pd.read_csv('data/train.csv')
    pipe = Pipeline([('Preprocessor', Preprocessor()),
                     ('Imputer', Imputer()),
                     ('Model', Model())
                     ])
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_val)

    # TODO: read about metrics and choose which to use
    score = roc_auc_score(y_val, predictions)
    print(f'Score: {round(score, 4)}')

    # TODO: write a function to run GridSearchCV obtain optimal parameters/model


if __name__ == '__main__':
    main()
