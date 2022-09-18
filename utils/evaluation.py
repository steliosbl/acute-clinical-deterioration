from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats as st
import math

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.base import clone as clone_estimator
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.dummy import DummyClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE, SMOTENC

from pytorch_tabnet.metrics import Metric as TabNetMetric


f2_score = make_scorer(fbeta_score, beta=2)


class F2TabNet(TabNetMetric):
    def __init__(self):
        self._name = "F2"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        return fbeta_score(y_true, y_pred, beta=2)


METRICS = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "F1 Score": "f1",
    "F2 Score": f2_score,
    "AUC": "roc_auc",
}


def roc_auc_ci(y_true, y_score):
    """ Computes AUROC with 95% confidence intervals
    Uses the formula from 
    :param y_true: True labels or binary label indicators
    :param y_score: Target scores
    """

    # See https://stackoverflow.com/a/20864883/7662085
    # zscore inside of which 95% of data lies
    za2 = st.norm.ppf(0.975)

    # From: https://gist.github.com/doraneko94/e24643136cfb8baf03ef8a314ab9615c
    AUC = roc_auc_score(y_true, y_score)
    N1, N2 = sum(y_true == True), sum(y_true != True)
    Q1, Q2 = AUC / (2 - AUC), 2 * AUC ** 2 / (1 + AUC)
    SE_AUC = math.sqrt(
        (AUC * (1 - AUC) + (N1 - 1) * (Q1 - AUC ** 2) + (N2 - 1) * (Q2 - AUC ** 2))
        / (N1 * N2)
    )
    lower, upper = max(AUC - za2 * SE_AUC, 0), min(AUC + za2 * SE_AUC, 1)
    return lower, AUC, upper


def joint_plot(
    subplots,
    filename=None,
    ax=None,
    title=None,
    style="white",
    legend_location="lower right",
    baseline_key="Baseline (NEWS)",
    plot_baseline=True,
):
    sns.set_style(style)
    plt.rc("axes", titlesize=16)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for key, plot in subplots.items():
        if key != baseline_key:
            plot.plot(ax=ax, name=key)

    ax.set_title(title)
    ax.legend(loc=legend_location)

    if filename:
        plt.savefig(f"{filename}_no_baseline.png", bbox_inches="tight")

    if baseline_key in subplots.keys() and plot_baseline:
        subplots[baseline_key].plot(
            ax=ax, linestyle="--", color="dimgray", name=baseline_key
        )
    ax.legend(loc=legend_location)

    if filename:
        plt.savefig(f"{filename}.png", bbox_inches="tight")

    plt.rc("axes", titlesize=12)  # Revert to default


def spotCheckDatasets(
    models,
    datasets,
    cv=3,
    set_params={},
    order=["dataset", "model"],
    fit_params={},
    scoring=METRICS,
):
    """ Run stratified k-fold cross-validation on the given models over the given dataset variants
        :param models: Single estimator or dict of the format {'Model Name': model}
        :param datasets: Single tuple (X, y) or dict of the format {'Dataset name': (X, y)}
        :param set_params: Hook to set model parameters based on each dataset. Callable of type function(X, y) -> Dict
    """

    models = {"Model": models} if type(models) != dict else models
    datasets = {"Data": datasets} if type(datasets) != dict else datasets

    param_callables = defaultdict(lambda: lambda X, y: dict())
    param_callables.update(set_params)

    fit_param_dict = defaultdict(lambda: dict())
    fit_param_dict.update(fit_params)

    tests = [
        (
            datakey,
            (X, y),
            modelkey,
            clone_estimator(model).set_params(**(param_callables[modelkey](X, y))),
            fit_param_dict[modelkey],
        )
        for datakey, (X, y) in datasets.items()
        for modelkey, model in models.items()
    ]

    return (
        pd.concat(
            (
                pd.DataFrame.from_dict(
                    cross_validate(
                        model,
                        X=X,
                        y=y,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=-1,
                        fit_params=fitparams,
                    )
                ).assign(dataset=datakey, model=modelkey)
                for datakey, (X, y), modelkey, model, fitparams in tests
            )
        )
        .groupby(["dataset", "model"])
        .mean()
    )


def spotCheckCV(models, X, y, cv=3, fit_params={}, scoring=METRICS):
    """ Run stratified k-fold cross-validation on the given models
    """
    return spotCheckDatasets(
        models, (X, y), fit_params=fit_params, scoring=scoring
    ).loc["Data"]


def spotCheckParams(model, X, y, cv=3, fit_params={}, scoring=METRICS):
    """ For more specialised models, run stratified k-fold cross-validation with fit-params
    """

    return (
        pd.DataFrame.from_dict(
            cross_validate(
                model,
                X=X,
                y=y,
                cv=cv,
                scoring=METRICS,
                n_jobs=-1,
                fit_params=fit_params,
            )
        )
        .assign(model="Model")
        .groupby("model")
        .mean()
    )


def train_test_split_oneclass(*frames, label, test_size=0.25, random_state=42):
    """ Split DataFrames or Series into random train and test subsets, such that the train subset has no positive instances
    """
    X = frames[0]
    label = label.reset_index(drop=True)
    train_idx, test_idx = train_test_split(
        label[~label].index, test_size=test_size, random_state=random_state
    )
    test_idx = np.concatenate([test_idx, label[label].index])

    return tuple(__ for _ in frames for __ in (_.iloc[train_idx], _.iloc[test_idx]))


def train_test_split_notna(*frames, test_size=0.25, random_state=42):
    """ Split DataFrames or Series into random train and test subsets, such that the test subset contains no NaN values.
    """
    X = frames[0]
    mask = ~X.reset_index(drop=True).isna().any(axis=1)
    sub_size = test_size * X.shape[0] / mask.sum()

    rem_idx, test_idx = train_test_split(
        mask[mask].index, test_size=sub_size, random_state=random_state
    )
    rem_idx = np.concatenate([rem_idx, mask[~mask].index])

    return tuple(__ for _ in frames for __ in (_.iloc[rem_idx], _.iloc[test_idx]))


def with_sampling_strategies(clf, clf_name="Classifier", categorical_cols_idx=[]):
    smote = (
        SMOTENC(categorical_features=categorical_cols_idx)
        if categorical_cols_idx
        else SMOTE()
    )

    return {
        f"{clf_name}_Undersampling": ImbPipeline(
            steps=[
                ("undersampling", RandomUnderSampler(sampling_strategy=0.1)),
                (clf_name, clone_estimator(clf)),
            ]
        ),
        f"{clf_name}_SMOTE": ImbPipeline(
            steps=[("smote", clone_estimator(smote)), (clf_name, clone_estimator(clf))]
        ),
        f"{clf_name}_OverUnder": ImbPipeline(
            steps=[
                ("smote", clone_estimator(smote).set_params(sampling_strategy=0.1)),
                ("undersampling", RandomUnderSampler(sampling_strategy=0.5)),
                (clf_name, clone_estimator(clf)),
            ]
        ),
        f"{clf_name}_SMOTE-Tomek": ImbPipeline(
            steps=[
                ("smote", clone_estimator(smote).set_params(sampling_strategy="auto")),
                ("tomek", TomekLinks(sampling_strategy="all")),
                (clf_name, clone_estimator(clf)),
            ]
        ),
    }


def evaluate_from_pred(
    y_true,
    y_pred,
    y_pred_proba,
    plot_title=None,
    pos_label=1,
    save=None,
    style="darkgrid",
):
    lower, auc, upper = roc_auc_ci(y_true, y_pred_proba)
    metric_df = pd.DataFrame(
        {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, pos_label=pos_label),
            "Recall": recall_score(y_true, y_pred, pos_label=pos_label),
            "F1 Score": f1_score(y_true, y_pred, pos_label=pos_label),
            "F2 Score": fbeta_score(y_true, y_pred, beta=2, pos_label=pos_label),
            "AUC": roc_auc_score(y_true, y_pred_proba),
            "AUC_CI": f"{auc:.3f} ({lower:.3f}-{upper:.3f})",
        },
        index=["Model"],
    )

    display(metric_df)

    sns.set_style(style)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[2].grid(False)
    roc_fig = RocCurveDisplay.from_predictions(
        y_true, y_pred_proba, ax=ax[0],  # pos_label=pos_label
    )
    pr_fig = PrecisionRecallDisplay.from_predictions(
        y_true, y_pred_proba, ax=ax[1], pos_label=pos_label
    )

    if (-1) in np.array(y_true):
        get = {1: True, -1: False}.get
        y_true, y_pred = (list(map(get, y_true)), list(map(get, y_pred)))
    cm_fig = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax[2], normalize="true", values_format=".2%"
    )

    plt.suptitle(plot_title)

    if save:
        plt.savefig(save, bbox_inches="tight")

    return metric_df, roc_fig, pr_fig, cm_fig


def evaluate(model, X, y, plot_title=None, save=None, style="darkgrid"):
    y_pred = model.predict(X)
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
    except AttributeError:
        try:
            y_pred_proba = model.decision_function(X)
        except AttributeError:
            y_pred_proba = model.score_samples(X)

    return evaluate_from_pred(
        y, y_pred, y_pred_proba, plot_title=plot_title, save=save, style=style
    )


def ideal_pr_curve(y, **kwargs):
    return PrecisionRecallDisplay.from_predictions(y, y, **kwargs)


def naive_pr_curve(y, **kwargs):
    return PrecisionRecallDisplay.from_predictions(y, np.zeros_like(y), **kwargs)


def naive_roc_curve(y, **kwargs):
    return RocCurveDisplay.from_predictions(y, np.zeros_like(y), **kwargs)
