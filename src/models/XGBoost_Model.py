import xgboost as xgb


def make_model():

    return xgb.XGBClassifier(reg_lambda = 0.21262699261144707, 
                            alpha = 0.002190735457731732, 
                            tree_method = 'gpu_hist',
                            objective = 'binary:logistic',
                            verbosity = 0,
                            n_jobs = -1, 
                            learning_rate = 0.01830371431723197,
                            min_child_weight = 12, 
                            max_depth = 6, 
                            max_delta_step = 5, 
                            subsample =  0.9711264167071539,
                            colsample_bytree =  0.10004257091322603,
                            gamma = 0.04562985736835494, 
                            n_estimators = 500,
                            eta = 0.11452245768637671)