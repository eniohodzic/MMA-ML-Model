import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


class OptunaTuning:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def objective(self, trial):
        # search_space = {
        #     'lambda': trial.suggest_float('lambda', 1e-8, 1, log=True),
        #     'alpha': trial.suggest_float('alpha', 1e-8, 1, log=True),
        #     'booster': trial.suggest_categorical('booster', ['gbtree']),
        #     'tree_method': trial.suggest_categorical('tree_method', ['gpu_hist']),
        #     "objective": trial.suggest_categorical("objective", ["binary:logistic"]),
        #     "verbosity": trial.suggest_categorical("verbosity", [0]),
        #     "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
        #     "min_child_weight": trial.suggest_int("min_child_weight", 0, 20),
        #     "max_depth": trial.suggest_int("max_depth", 3, 6),
        #     "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        #     "subsample": trial.suggest_float("subsample", 0.1, 1.0, log=False),
        #     "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, log=False),
        #     "gamma": trial.suggest_float("gamma", .01, 0.4, log=True),
        #     "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        #     "eta": trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
        # }

        search_space = {
            'lambda': trial.suggest_float('lambda', 1e-8, 1, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1, log=True),
            'booster': trial.suggest_categorical('booster', ['dart']),
            'tree_method': trial.suggest_categorical('tree_method', ['gpu_hist']),
            "objective": trial.suggest_categorical("objective", ["binary:logistic"]),
            "verbosity": trial.suggest_categorical("verbosity", [0]),
            "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
            "min_child_weight": trial.suggest_int("min_child_weight", 0, 20),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0, log=False),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, log=False),
            "gamma": trial.suggest_float("gamma", .01, 0.4, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "eta": trial.suggest_float("eta", 0.005, 0.1, log=True),
            'rate_drop' : trial.suggest_float("rate_drop", 0.01, 0.5, log=True),
            'skip_drop' : trial.suggest_float("skip_drop", 0.01, 0.5, log=True)
        }

        model = xgb.XGBClassifier(**search_space)
        kf = KFold(n_splits=5)
        kf.get_n_splits(self.X)

        acc = []
        for train_index, test_index in kf.split(self.X):

            X_train, X_test = self.X.iloc[train_index,:], self.X.iloc[test_index,:]
            y_train, y_test = self.y[train_index], self.y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test, iteration_range=(0,model.get_params()['n_estimators']))
            acc.append(accuracy_score(y_test, y_pred))

        return sum(acc) / len(acc)

    def print_status(self, study, trial):
        print(f"Best value: {study.best_value}")
        print(f"Best params: {study.best_trial.params}")

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=100, callbacks=[self.print_status])
        print("Number of finished trials: ", len(study.trials))
        print("Best trial: ", study.best_trial.params)
        print("Best value: ", study.best_value)