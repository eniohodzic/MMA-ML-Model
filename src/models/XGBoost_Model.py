import xgboost as xgb


def make_model():

    return xgb.XGBClassifier(reg_lambda = 0.5, 
                            alpha = 1, 
                            tree_method = 'gpu_hist',
                            objective = 'binary:logistic',
                            verbosity = 0,
                            n_jobs = -1, 
                            learning_rate = 0.02,
                            min_child_weight = 16, 
                            max_depth = 3, 
                            max_delta_step = 0, 
                            subsample =  0.75,
                            colsample_bytree =  0.1,
                            gamma = 0.05, 
                            n_estimators = 1000)