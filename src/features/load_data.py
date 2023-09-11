import pickle
from os.path import abspath, dirname, exists, join

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .feature_selection import get_corr_features


def load_data(split='train', return_drop_cols=False):
    """
    Loading dataset for model development. Select train, val, test, or all for chosen subset. 
    Sorts by time automatically. Outputs a dataset as a 3D numpy array with fighters, fights, features as axes.

    Can also return the columns dropped 
    """

    # Checking if pre-split data is present and loading without computations
    path = abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'final/', 'final_3D_' + split + '_.npz'))
    if exists(path):
        npz = np.load(path)
        return npz['X'], npz['y'], npz['odds']

    # Reading Processed Dataset
    df = pd.read_csv(abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'processed/', 'odds.csv')))
    df.drop(df.loc[df['result'] == 'D'].index, inplace=True) # Drop ties
    
    # Splitting dataframes into 
    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[df['date'] > '2012-01-01'] 
    val_df = df.loc[df['date'] > '2021-01-01'] 
    test_df = df.loc[df['date'] > '2022-06-01'] 
    train_df = df.drop(val_df.index)
    val_df = val_df.drop(test_df.index)  

    # Dropping non-precomp features and only keeping vs_opp for keeping symmetry in prediction 
    drop_cols = df.loc[:, :'referee'].columns 
    drop_cols = drop_cols.insert(0, df.columns[df.isna().mean() > 0.9].to_list()) # drop columns that contain more than % nans
    post_comp_cols = df.loc[:, ~df.columns.str.contains('precomp_([a-zA-Z_]+)_vs_opp', regex=True)].columns.to_list() 
    drop_cols = drop_cols.insert(0, post_comp_cols)

    # Dropping correlated features 
    features = df.drop(columns=drop_cols, axis=1)
    drop_features_cols = get_corr_features(features, thresh=0.95)
    drop_cols = drop_cols.insert(0, drop_features_cols)

    if return_drop_cols:
        return drop_cols
    
    # Selecting dataframe for processing based on subset
    if split == 'train':
        df = train_df
    elif split == 'val':
        df = val_df
    elif split == 'test':
        df = test_df
    elif split == 'full':
        pass
    else:
        raise ValueError('split was not properly defined')

    # Dropping columns
    multi_df = df.set_index(['fighter_url', df.groupby('fighter_url').cumcount(ascending=False)])
    drop_cols = drop_cols.drop_duplicates().drop('fighter_url')
    odds_df = multi_df['odds']
    final_df = multi_df.drop(columns=drop_cols, axis=1)

    # Removing fighters with many missing values
    mask = multi_df.isna().mean(axis=1).groupby('fighter_url').mean() < 0.5 
    
    X = final_df.loc[(mask[mask].index, slice(None)), :]
    y = multi_df.loc[(mask[mask].index, slice(None)), ['result']]
    odds = odds_df.loc[(mask[mask].index, slice(None))]

    X = X.to_xarray().to_array().to_numpy() # Feats x Fighter x Fights 
    y = y.to_xarray().to_array().to_numpy() # 1 x Fighter x Fights
    odds = np.expand_dims(odds.to_xarray().to_numpy(), 0) # 1 x Fighter x Fights

    X = np.transpose(X, (1, 2, 0)) # Fighter x Fights x Feats
    y = np.transpose(y, (1, 2, 0)) # Fighters x Fights x 1
    odds = np.transpose(odds, (1, 2, 0)) # Fighters x Fights x 1

    y[y == 'W'] = 1
    y[y == 'L'] = 0

    X = X.astype(np.float32)
    y = y.astype(np.float32)
    odds = odds.astype(np.float32)

    np.savez(path, X=X, y=y, odds=odds)

    return X, y, odds
