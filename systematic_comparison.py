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
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV

from models import *

from dataset import SCICategoriser

from typing import Dict, Any, Iterable, Optional, Tuple, Callable


def oneclasssplit(X_train, y_train, kf):
    for train_index, test_index in kf.split(X_train, y_train):
        if type(y_train) == pd.Series:
            y_train = y_train.to_numpy()
        trainy, validy = (
            y_train[train_index],
            y_train[test_index],
        )

        test_index = np.concatenate((train_index[trainy], test_index))
        train_index = train_index[~trainy]
        yield train_index, test_index


def get_studies(sci_train, study_grid=None, cli_model_arg=None):
    estimators = dict(
        cpu=[
            Estimator_IsolationForest,
            Estimator_LightGBM,
            Estimator_LogisticRegression,
            Estimator_RandomForest,
            Estimator_XGBoost,
            Estimator_LinearSVM,
            Estimator_OneClassSVM,
            Estimator_L1Regression,
            Estimator_L2Regression,
        ],
        gpu=[Estimator_TabNet],
    )
    estimators["all"] = estimators["cpu"] + estimators["gpu"]
    estimators.update({_._name: [_] for _ in estimators["all"]})

    if study_grid is None:
        study_grid = dict(
            estimator=estimators[cli_model_arg],
            resampler=[None, Resampler_RandomUnderSampler, Resampler_SMOTE],
            features=sci_train.feature_group_combinations,
        )

    k, v = zip(*study_grid.items())

    r = [dict(zip(k, _)) for _ in itertools.product(*v)]
    return [
        _
        for _ in r
        if not (_["estimator"]._requirements["oneclass"] and _["resampler"] is not None)
    ]


@dataclass
class PipelineFactory:
    estimator: Estimator
    resampler: Estimator
    X_train: SCIData
    y_train: pd.Series

    def __post_init__(self):
        self._scaler = self.estimator._requirements["scaling"]

    def __call__(self, **kwargs):
        steps = []
        if self.resampler is not None:
            steps.append((self.resampler._name, self.resampler.factory(),))

        if self._scaler:
            steps.append(
                (
                    "Scaling",
                    ColumnTransformer(
                        [("Scaler", StandardScaler(), self.X_train.numeric_columns)],
                        remainder="passthrough",
                    ),
                )
            )

        steps.append((self.estimator._name, self.estimator.factory()))
        return ImbPipeline(steps=steps).set_params(**kwargs)


@dataclass
class Objective:
    estimator: Estimator
    resampler: Estimator
    pipeline_factory: PipelineFactory
    X_train: SCIData
    y_train: pd.Series
    cv: int
    cv_jobs: int
    scoring: str
    n_trials: int
    stop_callback: Callable

    def __post_init__(self):
        self._fit_params = self.estimator.fit_params(self.X_train, self.y_train)

    def __call__(self, trial):
        trial_params = {
            **(self.resampler.suggest_parameters(trial) if self.resampler else {}),
            **self.estimator.suggest_parameters(trial),
        }
        model = self.pipeline_factory(**trial_params)

        cv = self.cv
        if self.estimator._requirements["oneclass"]:
            cv = oneclasssplit(
                self.X_train,
                self.y_train,
                StratifiedKFold(n_splits=5, random_state=42, shuffle=True),
            )

        score = cross_validate(
            model,
            self.X_train,
            self.y_train,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.cv_jobs,
            fit_params=self._fit_params,
        )["test_score"].mean()

        if trial.number >= self.n_trials:
            self.stop_callback()

        return score


def restructure_parameters(params, estimator_name, resampler_name):
    return {k: v for k, v in params.items() if k.startswith(estimator_name)} | {
        f"{resampler_name}__kw_args": {
            k.split("__")[1]: v
            for k, v in params.items()
            if k.startswith(resampler_name)
        }
    }


def evaluate_model(model, X_test, y_test, n_resamples):
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_pred_proba = model.decision_function(X_test)

    y_pred = np.where(
        y_pred_proba > get_threshold_fpr(y_test, y_pred_proba, target=0.05), 1, 0
    )

    metrics = get_metrics(y_test, y_pred, y_pred_proba, n_resamples)

    return metrics, y_pred_proba


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

    pipeline_factory = PipelineFactory(
        estimator=estimator, resampler=resampler, X_train=X_train, y_train=y_train,
    )

    objective = Objective(
        estimator=estimator,
        resampler=resampler,
        pipeline_factory=pipeline_factory,
        X_train=X_train,
        y_train=y_train,
        cv=cv,
        scoring=scoring,
        cv_jobs=cv_jobs,
        n_trials=n_trials,
        stop_callback=study.stop,
    )

    def handle_study_result(model_persistence_path=None, n_resamples=99, **kwargs):
        params = study.best_params
        if resampler:
            params = restructure_parameters(params, estimator._name, resampler._name)

        X, y = X_train, y_train
        if estimator._requirements["oneclass"]:
            X = SCIData(X_train[y_train.eq(0)])
            y = y_train[y_train.eq(0)]

        explanations = []
        if estimator._requirements["calibration"]:
            model = CalibratedClassifierCV(
                pipeline_factory(**params), cv=cv, method="isotonic", n_jobs=cv_jobs,
            ).fit(X, y)
            if estimator._requirements["explanation"]:
                explanations = estimator.explain_calibrated(model, X, X_test)
        else:
            model = pipeline_factory(**params).fit(X, y)
            if estimator._requirements["explanation"]:
                explanations = estimator.explain(model[estimator._name], X, X_test)

        if model_persistence_path is not None:
            with open(f"{model_persistence_path}/{name}.bin", "wb") as file:
                pickle.dump((model, explanations), file)

        metrics, y_pred_proba = evaluate_model(model, X_test, y_test, n_resamples)

        metrics = (
            dict(
                name=name,
                estimator=estimator._name,
                resampler=resampler._name if resampler else "None",
                features=features[0],
            )
            | metrics
        )

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
                    AssertionError,
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

