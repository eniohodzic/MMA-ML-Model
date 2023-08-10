import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score


class OptunaTuning:
    def __init__(self, X_test, y_test, X_train, y_train):
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train

    def objective(self, trial):
        search_space = {
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'tree_method': trial.suggest_categorical('tree_method', ['gpu_hist']),
            "objective": trial.suggest_categorical("objective", ["binary:logistic"]),
            "verbosity": trial.suggest_categorical("verbosity", [0]),
            "n_jobs": trial.suggest_categorical("n_jobs", [-1]),
            "min_child_weight": trial.suggest_int("min_child_weight", 0, 20),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
            "subsample": trial.suggest_float("subsample", 0.1, 1.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0, log=True),
            "gamma": trial.suggest_float("gamma", .01, 0.4, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        }
        model = xgb.XGBClassifier(**search_space)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def print_status(self, study, trial):
        print(f"Best value: {study.best_value}")
        print(f"Best params: {study.best_trial.params}")

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=2000, callbacks=[self.print_status])
        print("Number of finished trials: ", len(study.trials))
        print("Best trial: ", study.best_trial.params)
        print("Best value: ", study.best_value)