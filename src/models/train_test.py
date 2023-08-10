from os.path import abspath, dirname, join

import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_splitter(final_df, target_df):

    # test_df = df.loc[(df['date'] > '2022-01-01')]
    # train_df = df.drop(test_df.index)
    # train_df = train_df.loc[(train_df['date'] > '2014-01-01')]
    
    X_train, X_test, y_train, y_test = train_test_split(final_df, target_df, test_size=0.2)

    return X_train, X_test, y_train, y_test
