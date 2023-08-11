import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from ..features.load_data import load_data


def train_model(model):

    X, y = load_data()

    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    
    train_score_list = []
    test_score_list = []

    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]


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

        train_score_list.append(train_score)
        test_score_list.append(test_score)

    train_avg = sum(train_score_list) / len(train_score_list)
    test_avg = sum(test_score_list) / len(test_score_list)

    print(f'Model Training Accuracy: {train_avg}')
    print(f'Model Test Accuracy: {test_avg}')

    return model, [X, y]