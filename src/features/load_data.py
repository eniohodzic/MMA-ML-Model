from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .feature_selection import get_corr_features


def load_data(as_numpy=False):
    df = pd.read_csv(abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'processed/', 'extracted_stats.csv')))

    df['date'] = pd.to_datetime(df['date']) # make sure logic can handle datetime for time series data 
    df.drop(df.loc[df['result'] == 'D'].index, inplace=True) # Drop ties first 
    drop_cols = df.loc[:, :'referee'].columns # columns that will be dropped as features (strings, formats, etc.)

    drop_cols = drop_cols.insert(0, df.columns[df.isna().mean() > 0.9].to_list()) # drop columns that contain more than % nans

    post_comp_cols = df.loc[:, ~df.columns.str.contains('precomp_([a-zA-Z_]+)_vs_opp', regex=True)].columns.to_list() # All columns that do not have the precomp_ id or vs_opp id
    drop_cols = drop_cols.insert(0, post_comp_cols)

    features = df.drop(columns=drop_cols, axis=1)
    drop_features_cols = get_corr_features(features, thresh=0.95)
    drop_cols = drop_cols.insert(0, drop_features_cols)

    final_df = df.drop(columns=drop_cols, axis=1)

    indexes = final_df.loc[final_df.isna().mean(axis=1) < 0.5].index # Keep rows that have less than % nans by index
    
    X = final_df.loc[indexes, :]
    y = df.loc[indexes, 'result']

    y[y == 'W'] = 1
    y[y != 1] = 0

    if as_numpy:
        X = X.to_numpy(dtype=np.float32)
        y = np.reshape(y.to_numpy(dtype=np.float32), (-1,1))
        return X, y
    
    return X, y