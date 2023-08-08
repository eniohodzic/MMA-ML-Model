import pandas as pd


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr

def get_corr_features(df, thresh=0.95):
    x = get_top_abs_correlations(df)
    drop_features = []
    for idx, _ in x[x > thresh].items():
        drop_features.append(idx[1])
    drop_features = list(set(drop_features))

    return drop_features