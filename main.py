from preprocess import Preprocessor, Imputer, SelectKBest
from model import Model

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def main():
    df = pd.read_csv('data/train.csv')
    pipe = Pipeline([('Preprocessor', Preprocessor()),
                     ('Imputer', Imputer()),
                     ('Model', Model())
                     ])
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    pipe.fit(X, y)
    out = pipe.predict(X.iloc[:100])
    print(out)


if __name__ == '__main__':
    main()
