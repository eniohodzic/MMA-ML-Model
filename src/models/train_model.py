import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .feature_selection import get_corr_features
from .train_test import train_test_splitter


def train_model(model, df):

    df['date'] = pd.to_datetime(df['date']) # make sure logic can handle datetime for time series data 
    df.drop(df.loc[df['result'] == 'D'].index, inplace=True) # Drop ties first 
    drop_cols = df.loc[:, :'referee'].columns # columns that will be dropped as features (strings, formats, etc.)

    drop_cols = drop_cols.insert(0, df.columns[df.isna().mean() > 0.9].to_list()) # drop columns that contain more than % nans

    post_comp_cols = df.loc[:, ~df.columns.str.contains('precomp_([a-zA-Z_]+)_vs_opp', regex=True)].columns.to_list() # All columns that do not have the precomp_ id or vs_opp id
    drop_cols = drop_cols.insert(0, post_comp_cols)

    features = df.drop(columns=drop_cols, axis=1)
    drop_features_cols = get_corr_features(features, thresh=1)
    drop_cols = drop_cols.insert(0, drop_features_cols)

    final_df = df.drop(columns=drop_cols, axis=1)

    indexes = final_df.loc[final_df.isna().mean(axis=1) < 0.5].index # Keep rows that have less than % nans by index
    
    final_df = final_df.loc[indexes, :]
    target_df = df.loc[indexes, 'result']

    X_train, X_test, y_train, y_test = train_test_splitter(final_df, target_df)

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

    return model, [X_train, y_train, X_test, y_test]