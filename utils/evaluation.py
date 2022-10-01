from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats as st
import math

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.base import clone as clone_estimator
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedShuffleSplit,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    make_scorer,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.dummy import DummyClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE, SMOTENC

from pytorch_tabnet.metrics import Metric as TabNetMetric

from shapely.geometry import LineString


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


def alert_rate_curve(y_true, y_score, n_days, sample=None):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    alert_rate = (
        np.array(
            [np.where(y_score > threshold, 1, 0).sum() for threshold in thresholds]
        )
        / n_days
    )

    if sample is not None:
        return recall[::sample], alert_rate[::sample]
    else:
        return recall[:-1], alert_rate


def find_earliest_intersection(x1, y1, x2, y2, after=0.7):
    intersection = LineString(np.column_stack((x1, y1))).intersection(
        LineString(np.column_stack((x2, y2)))
    )

    if type(intersection) != LineString:
        intersection = LineString(intersection.geoms)

    if not intersection.xy[0]:
        return None

    return next(
        _ for _ in sorted(zip(*intersection.xy), key=lambda xy: xy[0]) if _[0] > 0.7
    )


def plot_alert_rate(y_pred_probas, y_test, n_days, intercept=None, ax=None, save=None):
    sns.set_style("white")
    plt.rc("axes", titlesize=16)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    x_intercept, y_intercept = alert_rate_curve(
        y_test, y_pred_probas[intercept], n_days, sample=None
    )
    for idx, (model, y_pred_proba) in enumerate(y_pred_probas.items()):
        if model == intercept:
            continue

        x, y = alert_rate_curve(y_test, y_pred_proba, n_days, sample=100)
        intersection = find_earliest_intersection(x_intercept, y_intercept, x, y)
        sns.lineplot(
            x=x, y=y, label=model, linewidth=2, ax=ax, color=sns.color_palette()[idx]
        )
        if intersection:
            ax.plot(*intersection, marker="x", color="black")
            ax.annotate(
                text=round(intersection[0], 3),
                xy=intersection,
                xytext=(1 - 0.03, intersection[1] - 0.4),
            )

    sns.lineplot(
        x=x_intercept,
        y=y_intercept,
        label=intercept,
        linestyle="--",
        linewidth=2,
        ax=ax,
        color="tomato",
    )

    ax.set_title("Sensitivity vs. Alert Rate")
    if save:
        plt.savefig(save, bbox_inches="tight")
    plt.rc("axes", titlesize=12)


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


def roc_auc_ci_bootstrap(y_true, y_score, n_resamples=9999):
    """ Computes AUROC with 95% confidence intervals by boostrapping """
    res = st.bootstrap(
        data=(y_true.to_numpy(), y_score),
        statistic=roc_auc_score,
        confidence_level=0.95,
        method="percentile",
        n_resamples=n_resamples,
        vectorized=False,
        paired=True,
        random_state=42,
    )

    return res.confidence_interval.low, res.confidence_interval.high


def joint_plot(
    subplots,
    filename=None,
    ax=None,
    title=None,
    style="white",
    legend_location="lower right",
    baseline_key="Baseline (NEWS)",
    plot_baseline=True,
    linewidth=1,
):
    sns.set_style(style)
    plt.rc("axes", titlesize=16)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for key, plot in subplots.items():
        if key != baseline_key:
            plot.plot(ax=ax, name=key, linewidth=linewidth)

    ax.set_title(title)
    ax.legend(loc=legend_location)

    if filename:
        plt.savefig(f"{filename}_no_baseline.png", bbox_inches="tight")

    if baseline_key in subplots.keys() and plot_baseline:
        subplots[baseline_key].plot(
            ax=ax,
            linestyle="--",
            color="dimgray",
            name=baseline_key,
            linewidth=linewidth,
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


def plot_confusion_matrix(y_true, y_pred, ax=None, save=None, plot_title=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.grid(False)
    cm_fig = ConfusionMatrixDisplay(
        np.rot90(np.flipud(confusion_matrix(y_true, y_pred, normalize="true"))),
        display_labels=[1, 0],
    ).plot(values_format=".2%", ax=ax)

    ax.set_xlabel("True Class")
    ax.set_ylabel("Predicted Class")
    ax.set_title(plot_title)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=200)

    return cm_fig


def evaluate_multiple(
    y_true, y_preds, n_resamples=99, news_modelkey=None, linewidth=2, save=None,
):
    sns.set_style("white")
    sns.set_palette("tab10")
    plt.rc("axes", titlesize=16)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    metrics = []
    for idx, (modelkey, (y_pred, y_pred_proba)) in enumerate(y_preds.items()):
        linestyle = "--" if modelkey == news_modelkey else "-"
        color = "tomato" if modelkey == news_modelkey else sns.color_palette()[idx]
        lower, upper = roc_auc_ci_bootstrap(y_true, y_pred_proba, n_resamples)
        auc = roc_auc_score(y_true, y_pred_proba)
        metrics.append(
            {
                "Model": modelkey,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred),
                "Recall": recall_score(y_true, y_pred),
                "F1 Score": f1_score(y_true, y_pred),
                "F2 Score": fbeta_score(y_true, y_pred, beta=2),
                "AUC": auc,
                "AUC_CI": f"{auc:.3f} ({lower:.3f}-{upper:.3f})",
            }
        )
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)

        sns.lineplot(
            x=fpr,
            y=tpr,
            label=f"{modelkey} (AUC = {auc:.2f})",
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            ax=ax[0],
        )
        sns.lineplot(
            x=recall,
            y=precision,
            label=f"{modelkey} (AP = {ap:.2f})",
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            ax=ax[1],
        )
        # roc_fig = RocCurveDisplay.from_predictions(
        #     y_true,
        #     y_pred_proba,
        #     ax=ax[0],
        #     linewidth=linewidth,
        #     name=modelkey,
        #     linestyle=linestyle,
        #     color=color,
        # )

        # pr_fig = PrecisionRecallDisplay.from_predictions(
        #     y_true,
        #     y_pred_proba,
        #     name=modelkey,
        #     linestyle=linestyle,
        #     ax=ax[1],
        #     linewidth=linewidth,
        #     color=color,
        # )

    ax[0].set_title("Receiver Operating Characteristic (ROC)")
    ax[1].set_title("Precision-Recall")
    ax[1].legend(loc="upper right")

    metrics = pd.DataFrame(metrics).set_index("Model")
    display(metrics)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=100)

    plt.rc("axes", titlesize=12)


def evaluate_all_outcomes(
    y_true,
    y_true_mortality,
    y_true_criticalcare,
    y_pred,
    y_pred_proba,
    modelkey,
    n_resamples=9999,
    news_prcurve_fix=False,
    linewidth=2,
    save=None,
):
    sns.set_style("darkgrid")
    sns.set_palette("tab10")
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    metrics = []
    for ylabel, y in [
        ("Critical event", y_true),
        ("Mortality only", y_true_mortality),
        ("Critical care", y_true_criticalcare),
    ]:
        lower, upper = roc_auc_ci_bootstrap(y, y_pred_proba, n_resamples)
        metrics.append(
            {
                modelkey: ylabel,
                "Accuracy": accuracy_score(y, y_pred),
                "Precision": precision_score(y, y_pred),
                "Recall": recall_score(y, y_pred),
                "F1 Score": f1_score(y, y_pred),
                "F2 Score": fbeta_score(y, y_pred, beta=2),
                "AUC": roc_auc_score(y, y_pred_proba),
                "AUC_CI": f"{roc_auc_score(y, y_pred_proba):.3f} ({lower:.3f}-{upper:.3f})",
            }
        )
        roc_fig = RocCurveDisplay.from_predictions(
            y, y_pred_proba, ax=ax[0], linewidth=linewidth, name=ylabel
        )

        p, r, _ = precision_recall_curve(y, y_pred_proba)
        if news_prcurve_fix:
            p, r = np.delete(p, -2), np.delete(r, -2)

        pr_fig = PrecisionRecallDisplay(p, r, estimator_name=ylabel)
        pr_fig.plot(ax=ax[1], linewidth=linewidth)

    metrics = pd.DataFrame(metrics).set_index(modelkey)
    display(metrics)

    cm_fig = plot_confusion_matrix(y_true, y_pred, ax[2])
    ax[1].legend(loc="upper right")

    plt.suptitle(modelkey)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=100)


def evaluate_from_pred(
    y_true,
    y_pred,
    y_pred_proba,
    plot_title=None,
    pos_label=1,
    save=None,
    style="darkgrid",
    n_resamples=9999,
):
    lower, upper = roc_auc_ci_bootstrap(y_true, y_pred_proba, n_resamples)
    metric_df = pd.DataFrame(
        {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, pos_label=pos_label),
            "Recall": recall_score(y_true, y_pred, pos_label=pos_label),
            "F1 Score": f1_score(y_true, y_pred, pos_label=pos_label),
            "F2 Score": fbeta_score(y_true, y_pred, beta=2, pos_label=pos_label),
            "AUC": roc_auc_score(y_true, y_pred_proba),
            "AUC_CI": f"{roc_auc_score(y_true, y_pred_proba):.3f} ({lower:.3f}-{upper:.3f})",
        },
        index=["Model"],
    )

    display(metric_df)

    display(confusion_matrix(y_true, y_pred))

    sns.set_style(style)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    roc_fig = RocCurveDisplay.from_predictions(
        y_true, y_pred_proba, ax=ax[0],  # pos_label=pos_label
    )
    pr_fig = PrecisionRecallDisplay.from_predictions(
        y_true, y_pred_proba, ax=ax[1], pos_label=pos_label
    )

    if (-1) in np.array(y_true):
        get = {1: True, -1: False}.get
        y_true, y_pred = (list(map(get, y_true)), list(map(get, y_pred)))

    # cm_fig = ConfusionMatrixDisplay.from_predictions(
    #     y_true, y_pred, ax=ax[1], normalize="true", values_format=".2%"
    # )
    cm_fig = plot_confusion_matrix(y_true, y_pred, ax[2])

    plt.suptitle(plot_title)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=100)

    return metric_df, roc_fig, pr_fig, cm_fig


def evaluate(
    model, X, y, plot_title=None, save=None, style="darkgrid", n_resamples=9999,
):
    y_pred = model.predict(X)
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
    except AttributeError:
        try:
            y_pred_proba = model.decision_function(X)
        except AttributeError:
            y_pred_proba = model.score_samples(X)

    return evaluate_from_pred(
        y,
        y_pred,
        y_pred_proba,
        plot_title=plot_title,
        save=save,
        style=style,
        n_resamples=n_resamples,
    )


def ideal_pr_curve(y, **kwargs):
    return PrecisionRecallDisplay.from_predictions(y, y, **kwargs)


def naive_pr_curve(y, **kwargs):
    return PrecisionRecallDisplay.from_predictions(y, np.zeros_like(y), **kwargs)


def naive_roc_curve(y, **kwargs):
    return RocCurveDisplay.from_predictions(y, np.zeros_like(y), **kwargs)
