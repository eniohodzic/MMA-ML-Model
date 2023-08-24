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
        return npz['X_train'], npz['y_train'], npz['X_test'], npz['y_test']
    if exists(path_3d) and as_3D:
        npz = np.load(path_3d)
        return npz['X_train'], npz['y_train'], npz['X_test'], npz['y_test']

    df = pd.read_csv(abspath(join(dirname(dirname(dirname(__file__))), 'data/', 'processed/', 'extracted_stats.csv')))

    df.drop(df.loc[df['result'] == 'D'].index, inplace=True) # Drop ties first 
    
    # Splitting training and test set based on date of bout to prevent data leakage 
    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[df['date'] > '2014-01-01'] 
    test_df = df.loc[df['date'] > '2022-01-01'] 
    train_df = df.drop(test_df.index)

    # Dropping non-precomp features 
    drop_cols = df.loc[:, :'referee'].columns 
    drop_cols = drop_cols.insert(0, df.columns[df.isna().mean() > 0.9].to_list()) # drop columns that contain more than % nans
    post_comp_cols = df.loc[:, ~df.columns.str.contains('precomp_([a-zA-Z_]+)_vs_opp', regex=True)].columns.to_list() 
    drop_cols = drop_cols.insert(0, post_comp_cols)

    # Dropping correlated features 
    features = df.drop(columns=drop_cols, axis=1)
    drop_features_cols = get_corr_features(features, thresh=0.95)
    drop_cols = drop_cols.insert(0, drop_features_cols)

    if as_3D:
        multi_df_train = train_df.set_index(['fighter_url', train_df.groupby('fighter_url').cumcount(ascending=False)])
        drop_cols = drop_cols.drop_duplicates().drop('fighter_url')
        final_df_train = multi_df_train.drop(columns=drop_cols, axis=1)

        multi_df_test = test_df.set_index(['fighter_url', test_df.groupby('fighter_url').cumcount(ascending=False)])
        final_df_test = multi_df_test.drop(columns=drop_cols, axis=1)
        
        mask_train = multi_df_train.isna().mean(axis=1).groupby('fighter_url').mean() < 0.5 
        mask_test = multi_df_test.isna().mean(axis=1).groupby('fighter_url').mean() < 0.5          
        
        X_train = final_df_train.loc[(mask_train[mask_train].index, slice(None)), :]
        y_train = multi_df_train.loc[(mask_train[mask_train].index, slice(None)), ['result']]

        X_test = final_df_test.loc[(mask_test[mask_test].index, slice(None)), :]
        y_test = multi_df_test.loc[(mask_test[mask_test].index, slice(None)), ['result']]

        X_train = X_train.to_xarray().to_array().to_numpy() # Feats x Fighter x Fights 
        y_train = y_train.to_xarray().to_array().to_numpy() # 1 x Fighter x Fights

        X_test = X_test.to_xarray().to_array().to_numpy() # Feats x Fighter x Fights 
        y_test = y_test.to_xarray().to_array().to_numpy() # 1 x Fighter x Fights

        X_train = np.transpose(X_train, (1, 2, 0)) # Fighter x Fights x Feats
        y_train = np.transpose(y_train, (1, 2, 0)) # Fighters x Fights x 1

        X_test = np.transpose(X_test, (1, 2, 0)) # Fighter x Fights x Feats
        y_test = np.transpose(y_test, (1, 2, 0)) # Fighters x Fights x 1

        y_train[y_train == 'W'] = 1
        y_train[y_train == 'L'] = 0

        y_test[y_test == 'W'] = 1
        y_test[y_test == 'L'] = 0

        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.float32)

        np.savez(path_3d, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        return X_train, y_train, X_test, y_test

    else:
        final_df = df.drop(columns=drop_cols, axis=1)

        indexes = final_df.loc[final_df.isna().mean(axis=1) < 0.5].index # Keep rows that have less than % nans by index
        
        X = final_df.loc[indexes, :]
        y = df.loc[indexes, 'result']

        y[y == 'W'] = 1
        y[y != 1] = 0

        X = X.to_numpy(dtype=np.float32)
        y = np.reshape(y.to_numpy(dtype=np.float32), (-1,1))

        np.savez(path_2d, X=X, y=y)

        return X, y
            