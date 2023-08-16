import pickle
from os.path import abspath, dirname, exists, join

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .feature_selection import get_corr_features


def load_data(as_3D=False):
    path_2d = abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'final/', 'final_2D.npz'))
    path_3d = abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'final/', 'final_3D.npz'))
    
    if exists(path_2d) and not as_3D:
        npz = np.load(path_2d)
        return npz['X'], npz['y']
    if exists(path_3d) and as_3D:
        npz = np.load(path_3d)
        return npz['X'], npz['y']

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

    if as_3D:
        multi_df = df.set_index(['fighter_url', df.groupby('fighter_url').cumcount(ascending=False)])
        drop_cols = drop_cols.drop_duplicates().drop('fighter_url')
        final_df = multi_df.drop(columns=drop_cols, axis=1)
        
        mask = multi_df.isna().mean(axis=1).groupby('fighter_url').mean() < 0.5           # Dropping Slices that are below some percent
        
        X = final_df.loc[(mask[mask].index, slice(None)), :]
        y = multi_df.loc[(mask[mask].index, slice(None)), ['result']]

        X = X.to_xarray().to_array().to_numpy() # Feats x Fighter x Fights 
        y = y.to_xarray().to_array().to_numpy() # 1 x Fighter x Fights

        X = np.transpose(X, (1, 2, 0)) # Fighter x Fights x Feats
        y = np.transpose(y, (1, 2, 0)) # Fighters x Fights x 1

        y[y == 'W'] = 1
        y[y == 'L'] = 0

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        np.savez(path_3d, X=X, y=y)

        return X, y

    else:
        final_df = df.drop(columns=drop_cols, axis=1)

        indexes = final_df.loc[final_df.isna().mean(axis=1) < 0.5].index # Keep rows that have less than % nans by index
        
        X = final_df.loc[indexes, :]
        y = df.loc[indexes, 'result']

        y[y == 'W'] = 1
        y[y != 1] = 0

        X = X.to_numpy(dtype=np.float32)
        y = np.reshape(y.to_numpy(dtype=np.float32), (-1,1))

        np.savez(path_3d, X=X, y=y)

        return X, y
            