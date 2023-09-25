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
    if exists(path) and not return_drop_cols:
        npz = np.load(path)
        return npz['X'], npz['y'], npz['odds']

    # Reading Processed Dataset and organize into multiindex 
    df = pd.read_csv(abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'processed/', 'PROCESSED_stats_plus_odds.csv')))
    multi_df = df.set_index(['fighter_url', df.groupby('fighter_url').cumcount(ascending=False)])
    multi_df.index.names = ['fighter_url', 'fight_num']

    # Remove all Women fights from dataframe
    multi_df = multi_df[~multi_df['division'].str.contains('Women')]

    # Select rows where fighters have at least X fights (2)
    num = 3
    multi_df = multi_df.query(f'fight_num >= {num}')
    multi_df.drop(multi_df.loc[multi_df['result'] == 'D'].index, inplace=True) # Drop ties

    # Splitting dataframes into test, val, train based on fight date 
    multi_df['date'] = pd.to_datetime(multi_df['date'])
    multi_df = multi_df.loc[multi_df['date'] > '2012-01-01'] 
    val_df = multi_df.loc[multi_df['date'] > '2021-01-01'] 
    test_df = multi_df.loc[multi_df['date'] > '2022-06-01'] 
    train_df = multi_df.drop(val_df.index)
    val_df = val_df.drop(test_df.index)  

    # Dropping non-precomp features and only keeping vs_opp for keeping symmetry in prediction 
    drop_cols = multi_df.loc[:, :'referee'].columns 
    drop_cols = drop_cols.insert(0, multi_df.columns[multi_df.isna().mean() > 0.5].to_list()) # drop columns that contain more than % nans
    post_comp_cols = multi_df.loc[:, ~multi_df.columns.str.contains('precomp_(?:[a-zA-Z_]+)_vs_opp', regex=True)].columns.to_list() 
    drop_cols = drop_cols.insert(0, post_comp_cols)

    # Dropping correlated features 
    # features = multi_df.drop(columns=drop_cols, axis=1)
    # drop_features_cols = get_corr_features(features, thresh=0.6)
    # drop_cols = drop_cols.insert(0, drop_features_cols)

    if return_drop_cols:
        return drop_cols
    
    # Selecting dataframe for processing based on subset
    if split == 'train':
        multi_df = train_df
    elif split == 'val':
        multi_df = val_df
    elif split == 'test':
        multi_df = test_df
    elif split == 'full':
        pass
    else:
        raise ValueError('split was not properly defined')

    # Dropping columns
    drop_cols = drop_cols.drop_duplicates()
    odds_df = multi_df[['odds']]
    final_df = multi_df.drop(columns=drop_cols, axis=1)

    # Removing fighters with many missing values
    mask = multi_df.isna().mean(axis=1).groupby('fighter_url').mean() < 0.5 
    
    X = final_df.loc[(mask[mask].index, slice(None)), :]
    y = multi_df.loc[(mask[mask].index, slice(None)), ['result']]
    odds = odds_df.loc[(mask[mask].index, slice(None))]

    # Converting to float tensor and saving 
    X = X.to_xarray().to_array().to_numpy() # Feats x Fighter x Fights 
    y = y.to_xarray().to_array().to_numpy() # 1 x Fighter x Fights
    odds = odds.to_xarray().to_array().to_numpy() # 1 x Fighter x Fights

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
