import xgboost as xgb


def make_model():

    return xgb.XGBClassifier(reg_lambda = 0.21262699261144707, 
                            alpha = 9.603744709436778, 
                            tree_method = 'gpu_hist',
                            objective = 'binary:logistic',
                            verbosity = 0,
                            n_jobs = -1, 
                            learning_rate = 0.01830371431723197,
                            min_child_weight = 12, 
                            max_depth = 6, 
                            max_delta_step = 5, 
                            subsample = 0.12516270393991097,
                            colsample_bytree = 0.39799515236683536,
                            gamma = 0.225275077908943, 
                            n_estimators = 315,
                            eta = 0.11452245768637671)