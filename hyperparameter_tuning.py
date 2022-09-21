import warnings
import numpy as np
import torch, optuna
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping
from lightgbm import log_evaluation

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from utils.isolation_forest_wrapper import IsolationForestWrapper, isolationforestsplit

optuna.logging.set_verbosity(optuna.logging.WARNING)


class TabnetObjective:
    def __init__(self, X_train, y_train, categorical_cols_idx, categorical_cols_dims):
        (
            self.X_train,
            self.y_train,
            self.categorical_cols_idx,
            self.categorical_cols_dims,
        ) = (
            X_train.to_numpy(),
            y_train.to_numpy(),
            categorical_cols_idx,
            categorical_cols_dims,
        )

    def __call__(self, trial):
        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_da = trial.suggest_int("n_da", 56, 64, step=4)
        n_steps = trial.suggest_int("n_steps", 1, 3, step=1)
        gamma = trial.suggest_float("gamma", 1.0, 1.4, step=0.2)
        n_shared = trial.suggest_int("n_shared", 1, 3)
        lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)

        tabnet_params = dict(
            cat_idxs=self.categorical_cols_idx,
            cat_dims=self.categorical_cols_dims,
            n_d=n_da,
            n_a=n_da,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            mask_type=mask_type,
            n_shared=n_shared,
            scheduler_params=dict(
                mode="min",
                patience=trial.suggest_int(
                    "patienceScheduler", low=3, high=10
                ),  # changing scheduler patience to be lower than early stopping patience
                min_lr=1e-5,
                factor=0.5,
            ),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            verbose=0,
        )  # early stopping
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        CV_score_array = []
        for train_index, test_index in kf.split(X_train, y_train):
            X_train_tn, X_valid_tn = (
                self.X_train[train_index],
                self.X_train[test_index],
            )
            y_train_tn, y_valid_tn = (
                self.y_train[train_index],
                self.y_train[test_index],
            )
            clf = TabNetClassifier(**tabnet_params)
            clf.fit(
                X_train=X_train_tn,
                y_train=y_train_tn,
                eval_set=[(X_valid_tn, y_valid_tn)],
                patience=trial.suggest_int("patience", low=15, high=30),
                max_epochs=trial.suggest_int("epochs", 1, 100),
                eval_metric=["auc"],
            )
            CV_score_array.append(clf.best_cost)
        avg = np.mean(CV_score_array)
        return avg


def tune_tabnet(X_train, y_train, categorical_cols_idx, categorical_cols_dims):
    obj = TabnetObjective(X_train, y_train, categorical_cols_idx, categorical_cols_dims)
    study = optuna.create_study(direction="maximize", study_name="TabNet optimization")
    study.optimize(obj, n_trials=1, n_jobs=-1, timeout=60 * 60)
    tabnet_params = dict(
        cat_idxs=categorical_cols_idx,
        cat_dims=categorical_cols_dims,
        n_a=study.best_params["n_da"],
        n_d=study.best_params["n_da"],
        n_steps=study.best_params["n_steps"],
        gamma=study.best_params["gamma"],
        lambda_sparse=study.best_params["lambda_sparse"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        mask_type=study.best_params["mask_type"],
        n_shared=study.best_params["n_shared"],
        scheduler_params=dict(
            mode="min",
            patience=study.best_params["patienceScheduler"],
            min_lr=1e-5,
            factor=0.5,
        ),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        verbose=0,
    )

    print("BEST PARAMETERS")
    print(study.best_params)

    return tabnet_params, study.best_params["epochs"], study.best_params["patience"]


class XgboostObjective:
    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train

    def __call__(self, trial):
        param = {
            "XGB__verbosity": 0,
            "XGB__objective": "binary:logistic",
            "XGB__enable_categorical": True,
            # use exact for small dataset.
            "XGB__tree_method": trial.suggest_categorical(
                "tree_method", ["approx", "hist"]
            ),
            # defines booster, gblinear for linear functions.
            "XGB__booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            # L2 regularization weight.
            "XGB__lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "XGB__alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "XGB__subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "XGB__colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            # weighting for imbalance
            "XGB__scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
            "IMB__sampling_strategy": trial.suggest_float(
                "IMB_sampling_strategy", 0.1, 0.5
            ),
        }

        if param["XGB__booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["XGB__max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
            # minimum child weight, larger the term more conservative the tree.
            param["XGB__min_child_weight"] = trial.suggest_int(
                "min_child_weight", 2, 10
            )
            param["XGB__eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["XGB__gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["XGB__grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            )

        if param["XGB__booster"] == "dart":
            param["XGB__sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"]
            )
            param["XGB__normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]
            )
            param["XGB__rate_drop"] = trial.suggest_float(
                "rate_drop", 1e-8, 1.0, log=True
            )
            param["XGB__skip_drop"] = trial.suggest_float(
                "skip_drop", 1e-8, 1.0, log=True
            )

        model = ImbPipeline(
            steps=[("IMB", RandomUnderSampler()), ("XGB", XGBClassifier()),]
        ).set_params(**param)
        cv = cross_validate(model, X_train, y_train, cv=5, scoring="roc_auc",)
        return cv["test_score"].mean()


def tune_xgboost(X_train, y_train, n_trials=100, timeout=60 * 60, n_jobs=-1):
    obj = XgboostObjective(X_train, y_train)
    study = optuna.create_study(direction="maximize", study_name="XGBoost optimization")
    study.optimize(obj, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

    print("BEST PARAMETERS")
    print(study.best_params)

    return study.best_params


def tune_lgbm(X_train, y_train, timeout=60 * 60):
    dtrain = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "binary",
        "metrics": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "is_unbalance": True,
        "verbose": -1,
    }

    tuner = lgb.LightGBMTunerCV(
        params,
        dtrain,
        folds=StratifiedKFold(n_splits=3),
        callbacks=[
            early_stopping(100, verbose=False),
            log_evaluation(100, show_stdv=False),
        ],
        time_budget=timeout,
        show_progress_bar=False,
    )

    tuner.run()

    return tuner.best_params


class RandomForestObjective:
    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = X_train.to_numpy(), y_train

    def __call__(self, trial):
        param = {}
        param["RF__n_estimators"] = trial.suggest_categorical(
            "RF__n_estimators", [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
        )
        param["RF__max_features"] = trial.suggest_float("RF__max_features", 0.15, 1.0)
        param["RF__min_samples_split"] = trial.suggest_int(
            "RF__min_samples_split", 2, 14
        )
        param["RF__min_samples_leaf"] = trial.suggest_int("RF__min_samples_leaf", 1, 14)
        param["RF__max_samples"] = trial.suggest_float("RF__max_samples", 0.6, 0.99)
        param["RF__class_weight"] = trial.suggest_categorical(
            "RF__class_weight", [None, "balanced", "balanced_subsample"]
        )
        param["IMB__sampling_strategy"] = trial.suggest_float(
            "IMB__sampling_strategy", 0.1, 0.5
        )

        model = ImbPipeline(
            steps=[("IMB", RandomUnderSampler()), ("RF", RandomForestClassifier())]
        ).set_params(**param)

        cv = cross_validate(model, self.X_train, self.y_train, cv=5, scoring="roc_auc")
        return cv["test_score"].mean()


def tune_randomforest(X_train, y_train, n_trials=100, timeout=60 * 60, n_jobs=-1):
    obj = RandomForestObjective(X_train, y_train)
    study = optuna.create_study(
        direction="maximize", study_name="Random Forest optimization"
    )
    study.optimize(obj, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

    print("BEST PARAMETERS")
    print(study.best_params)

    return study.best_params


class IsolationForestObjective:
    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = X_train.to_numpy(), y_train.to_numpy()

    def __call__(self, trial):
        param = {}
        param["n_estimators"] = trial.suggest_int("n_estimators", 1, 200)
        param["max_samples"] = trial.suggest_float("max_samples", 0.0, 1.0)
        param["contamination"] = trial.suggest_float("contamination", 1e-6, 1e-1)
        param["max_features"] = trial.suggest_float("max_features", 0.0, 1.0)
        param["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])

        model = IsolationForestWrapper(**param)
        cv = cross_validate(
            model,
            self.X_train,
            self.y_train,
            cv=isolationforestsplit(
                self.X_train,
                self.y_train,
                StratifiedKFold(n_splits=5, random_state=42, shuffle=True),
            ),
            scoring='roc_auc'
        )
        return cv["test_score"].mean()


def tune_isolationforest(X_train, y_train, n_trials=100, timeout=60 * 60, n_jobs=-1):
    obj = IsolationForestObjective(X_train, y_train)
    study = optuna.create_study(
        direction="maximize", study_name="Isolation Forest optimization"
    )
    study.optimize(obj, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

    print("BEST PARAMETERS")
    print(study.best_params)

    return study.best_params


class LogisticObjective:
    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = \
            X_train, y_train

    def __call__(self, trial):
        param = {}
        param['LR__penalty'] = trial.suggest_categorical('LR__penalty', ['l2', 'l1'])
        if param['LR__penalty'] == 'l1':
            param['LR__solver'] = 'saga'
        else:
            param['LR__solver'] = 'lbfgs'
        param['LR__C'] = trial.suggest_float('LR__C', 0.01, 10)
        param['LR__class_weight'] = trial.suggest_categorical('LR__class_weight', [None, 'balanced'])
        param["IMB__sampling_strategy"] = trial.suggest_float(
            "IMB__sampling_strategy", 0.1, 0.5
        )

        model = ImbPipeline(
            steps=[("IMB", RandomUnderSampler()), ("LR", LogisticRegression(max_iter=10000))]
        ).set_params(**param)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            cv = cross_validate(model, self.X_train, self.y_train, cv=5, scoring="roc_auc")
        return cv["test_score"].mean()

def tune_logisticregression(X_train, y_train, n_trials=100, timeout=60 * 60, n_jobs=-1):
    obj = LogisticObjective(X_train, y_train)
    study = optuna.create_study(
        direction="maximize", study_name="Logistic Regression optimization"
    )
    study.optimize(obj, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

    print("BEST PARAMETERS")
    print(study.best_params)

    return study.best_params