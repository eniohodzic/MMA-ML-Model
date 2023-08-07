from os.path import abspath, dirname, join

import pandas as pd


def train_test_split(df):

    test_df = df.loc[(df['date'] > '2020-01-01')]
    train_df = df.drop(test_df.index)
    train_df = train_df.loc[(train_df['date'] > '2012-01-01')]

    return train_df, test_df

def X_y_split(train_df, test_df, target, drop_cols):

    drop_cols = drop_cols.insert(0, target)

    X_train = train_df.drop(columns=drop_cols, axis=1)
    y_train = train_df[target]

    X_test = test_df.drop(columns=drop_cols, axis=1)
    y_test = test_df[target]

    return X_train, y_train, X_test, y_test

