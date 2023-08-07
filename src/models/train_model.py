import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .train_test import X_y_split, train_test_split


def train_model(model, df):

    df['date'] = pd.to_datetime(df['date']) # make sure logic can handle datetime for time series data 
    df.drop(df.loc[df['result'] == 'D'].index, inplace=True) # Drop ties first 
    drop_cols = df.loc[:, :'referee'].columns # columns that will be dropped as features (strings, formats, etc.)

    drop_cols = drop_cols.insert(0, df.columns[df.isna().all()].to_list()) # drop columns that contain all nans

    post_comp_cols = df.loc[:, ~df.columns.str.contains('precomp_([a-zA-Z_]+)_vs_opp', regex=True)].columns.to_list() # All columns that do not have the precomp_ id or vs_opp id

    drop_cols = drop_cols.insert(0, post_comp_cols)

    train_df, test_df = train_test_split(df)
    X_train, y_train, X_test, y_test = X_y_split(train_df.copy(), test_df.copy(), 'result', drop_cols)

    y_train[y_train == 'W'] = 1
    y_train[y_train != 1] = 0

    y_test[y_test == 'W'] = 1
    y_test[y_test != 1] = 0

    y_train = y_train.to_numpy(dtype=int)
    y_test = y_test.to_numpy(dtype=int)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_score = accuracy_score(y_train, y_pred_train)
    test_score = accuracy_score(y_test, y_pred_test)

    print(f'Model Training Accuracy: {train_score}')
    print(f'Model Test Accuracy: {test_score}')

    return model, X_train.columns