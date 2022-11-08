import warnings, pickle, os, itertools, argparse
from joblib import Parallel, delayed, parallel_backend

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 300)

from IPython.display import display
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={"figure.figsize": (10, 10)})

import shap
import optuna, sqlalchemy

optuna.logging.set_verbosity(optuna.logging.WARNING)

from utils.evaluation import get_metrics, get_threshold_fpr
from sklearn.model_selection import train_test_split

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import cross_validate
from sklearn.calibration import CalibratedClassifierCV

from models import *

from typing import Dict, Any, Iterable, Optional, Tuple


def get_studies(sci_train, study_grid=None, cli_model_arg=None):
    estimators = dict(
        cpu=[
            Estimator_IsolationForest,
            Estimator_LightGBM,
            Estimator_LogisticRegression,
            Estimator_RandomForest,
            Estimator_XGBoost,
        ],
        gpu=[Estimator_TabNet],
    )
    estimators["all"] = estimators["cpu"] + estimators["gpu"]
    estimators.update({_._name: [_] for _ in estimators["all"]})

    if study_grid is None:
        study_grid = dict(
            estimator=estimators[cli_model_arg],
            resampler=[No_Resampling, Resampler_RandomUnderSampler, Resampler_SMOTE],
            features=sci_train.feature_group_combinations,
        )

    k, v = zip(*study_grid.items())
    return [dict(zip(k, _)) for _ in itertools.product(*v)]


class PipelineFactory:
    def __init__(
        self,
        estimator: Estimator,
        X_train,
        y_train,
        resampler: Optional[Estimator] = None,
    ):
        (self._estimator, self._resampler, self._X_train, self._y_train,) = (
            estimator,
            resampler,
            X_train,
            y_train,
        )

    def __call__(self, **kwargs):
        steps = [
            (self._estimator._name, self._estimator.factory(),),
        ]
        if self._resampler is not None:
            steps = [(self._resampler._name, self._resampler.factory(),),] + steps

        return ImbPipeline(steps=steps).set_params(**kwargs)


class Objective:
    def __init__(
        self,
        estimator: Estimator,
        resampler: Estimator,
        X_train,
        y_train,
        cv=5,
        scoring="average_precision",
        cv_jobs=1,
        n_trials=100,
        stop_callback=None,
    ):
        (
            self._estimator,
            self._resampler,
            self._X_train,
            self._y_train,
            self._cv,
            self._scoring,
            self._cv_jobs,
            self._n_trials,
            self._stop_callback,
        ) = (
            estimator,
            resampler,
            X_train,
            y_train,
            cv,
            scoring,
            cv_jobs,
            n_trials,
            stop_callback,
        )

        self._pipeline_factory = PipelineFactory(
            estimator=self._estimator,
            resampler=self._resampler,
            X_train=self._X_train,
            y_train=self._y_train,
        )

        self._fit_params = self._estimator.fit_params(self._X_train, self._y_train)

    def __call__(self, trial):
        trial_params = {
            **(self._resampler.suggest_parameters(trial) if self._resampler else {}),
            **self._estimator.suggest_parameters(trial),
        }
        model = self._pipeline_factory(**trial_params)

        score = cross_validate(
            model,
            self._X_train,
            self._y_train,
            cv=self._cv,
            scoring=self._scoring,
            n_jobs=self._cv_jobs,
            fit_params=self._fit_params,
        )["test_score"].mean()

        if trial.number >= self._n_trials:
            self._stop_callback()

        return score


def construct_study(
    estimator: Estimator,
    sci_train: SCIData,
    sci_test: SCIData,
    features: Tuple[str, Iterable[str]] = ("All", []),
    resampler: Estimator = None,
    cv=5,
    scoring="average_precision",
    storage=None,
    model_persistence_path=None,
    cv_jobs=1,
    n_trials=100,
    **kwargs,
):
    X_train, y_train, X_test, y_test = estimator.get_xy(
        sci_train, sci_test, features[1]
    )

    name = f"{estimator._name}_{resampler._name if resampler else 'None'}_{features[0]}"
    study = optuna.create_study(
        direction="maximize", study_name=name, storage=storage, load_if_exists=True
    )
    objective = Objective(
        estimator=estimator(SCIData(sci_train[features[1]])),
        resampler=resampler(SCIData(sci_train[features[1]])) if resampler else None,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        scoring=scoring,
        cv_jobs=cv_jobs,
        n_trials=n_trials,
        stop_callback=study.stop,
    )

    def handle_study_result(model_persistence_path=None, n_resamples=99, **kwargs):
        model = CalibratedClassifierCV(
            objective._pipeline_factory(**study.best_params),
            cv=cv,
            method="isotonic",
            n_jobs=cv_jobs,
        ).fit(X_train, y_train)
        explanations = estimator.explain_calibrated(model, X_test)

        if model_persistence_path is not None:
            with open(f"{model_persistence_path}/{name}.bin", "wb") as file:
                pickle.dump((model, explanations), file)

        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_pred_proba = model.decision_function(X_test)

        y_pred = np.where(
            y_pred_proba > get_threshold_fpr(y_test, y_pred_proba, target=0.05), 1, 0
        )

        metrics = {
            **dict(
                name=name,
                estimator=estimator._name,
                resampler=resampler._name if resampler else "None",
                features=features[0],
            ),
            **get_metrics(y_test, y_pred, y_pred_proba, n_resamples),
        }

        return metrics, (name, y_pred_proba)

    def call(model_persistence_path=None, n_resamples=99, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            while True:
                try:
                    study.optimize(objective, **kwargs)
                    return handle_study_result(model_persistence_path, n_resamples)
                except (
                    optuna.exceptions.StorageInternalError,
                    sqlalchemy.exc.OperationalError,
                ):
                    print("################# CAUGHT DB ERROR #################")
                    pass

    return call


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models", help="Can be 'all', 'cpu', or 'gpu'", type=str)
parser.add_argument(
    "-j", "--njobs", help="Number of CPUs to use. Default=1", type=int, default=1
)
parser.add_argument(
    "-c", "--cv_jobs", help="Number of CV jobs. Default=5", type=int, default=5
)
parser.add_argument(
    "-p",
    "--persist",
    help="Filepath to save the models. If unset, wont save them",
    type=str,
    default=None,
)
parser.add_argument(
    "-d",
    "--debug",
    help="Whether to only use a small subset of data for debugging",
    action="store_true",
)
parser.add_argument(
    "-t", "--trials", help="Number of trials. Default=1000", type=int, default=1000
)
parser.add_argument(
    "-hr", "--hours", help="Trial timeout in hours", type=int, default=2
)
parser.add_argument(
    "-s",
    "--storage",
    help="Trial storage for optuna",
    default=None,  # "sqlite:///models/studies.db",
)
parser.add_argument(
    "-o", "--output", help="Output path for final results", default="results.h5"
)
parser.add_argument(
    "--n_resamples", help="Number of resamples for bootstrapping metrics", default=999
)
parser.add_argument("-v", "--verbose", help="Optuna verbosity", action="store_true")


def run(args):
    args = vars(args)
    scii = (
        SCIData(
            SCIData.quickload("data/sci_processed.h5").sort_values("AdmissionDateTime")
        )
        .mandate(SCICols.news_data_raw)
        .derive_critical_event(within=1, return_subcols=True)
        .augment_shmi(onehot=True)
        .omit_redundant()
        .derive_ae_diagnosis_stems(onehot=False)
        .categorize()
    )

    sci_train, sci_test, _, y_test_mortality, _, y_test_criticalcare = train_test_split(
        scii,
        scii.DiedWithinThreshold,
        scii.CriticalCare,
        test_size=0.33,
        random_state=42,
        shuffle=False,
    )
    sci_train, sci_test = SCIData(sci_train), SCIData(sci_test)

    if args["verbose"]:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    if args["persist"] is not None:
        try:
            os.makedirs(args["persist"])
        except FileExistsError:
            pass

    sci_train_ = sci_train
    if args["debug"]:
        sci_train_ = SCIData(sci_train.sample(1000))

    if args["storage"] is not None:
        args["storage"] = optuna.storages.RDBStorage(
            url=args["storage"], engine_kwargs={"connect_args": {"timeout": 100}}
        )

    n_trials = args["trials"] if not args["debug"] else 2

    studies = [
        construct_study(
            **_, **args, sci_train=sci_train_, sci_test=sci_test, n_trials=n_trials
        )
        for _ in get_studies(sci_train, cli_model_arg=args["models"])
    ]

    study_args = dict(
        model_persistence_path=args["persist"],
        n_resamples=args["n_resamples"],
        n_trials=n_trials,
        timeout=args["hours"] * 60 * 60,
    )

    if args["njobs"] > 1:
        print("Starting execution (parallel)")
        with parallel_backend("loky", inner_max_num_threads=args["cv_jobs"]):
            results = Parallel(n_jobs=args["njobs"])(
                delayed(_)(**study_args) for _ in studies
            )
    else:
        print("Starting execution (linear)")
        results = [_(**study_args) for _ in studies]

    metrics, y_preds = list(zip(*results))
    pd.DataFrame(metrics).to_hdf(args["output"], "metrics")
    pd.DataFrame(dict(y_preds)).to_hdf(args["output"], "y_preds")


if __name__ == "__main__":
    run(parser.parse_args())

