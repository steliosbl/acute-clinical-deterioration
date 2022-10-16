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
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE, SMOTENC

from pytorch_tabnet.metrics import Metric as TabNetMetric

from shapely.geometry import LineString, Point

import shap


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


def biggest_alert_rate_diff(y_true, y_score_x, y_score_y, n_days):
    r_x, a_x = alert_rate_curve(y_true, y_score_x, n_days)
    r_y, a_y = alert_rate_curve(y_true, y_score_y, n_days)

    diffs = np.array(
        [a_x[idx] - a_y[np.argmin(np.abs(r_y - _))] for idx, _ in enumerate(r_x)]
    )

    biggest_diff = diffs.argmax()
    recall_at_biggest_diff = r_x[diffs.argmax()]
    closest_recall_idx_in_y = np.argmin(np.abs(r_y - recall_at_biggest_diff))
    return recall_at_biggest_diff, a_x[diffs.argmax()], a_y[closest_recall_idx_in_y]


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

    if type(intersection) not in [LineString, Point]:
        intersection = LineString(intersection.geoms)

    if not intersection.xy[0]:
        return None

    return next(
        _ for _ in sorted(zip(*intersection.xy), key=lambda xy: xy[0]) if _[0] > 0.7
    )


def plot_alert_rate(
    y_true, y_preds, n_days, baseline_key=None, ax=None, save=None, save_format="png"
):
    no_ax = ax is None
    if no_ax:
        sns.set_style("white")
        plt.rc("axes", titlesize=14)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if type(list(y_preds.values())[0]) == tuple:
        y_preds = {key: value[1] for key, value in y_preds.items()}

    x_intercept, y_intercept = alert_rate_curve(
        y_true, y_preds[baseline_key], n_days, sample=None
    )

    for idx, (model, y_pred_proba) in enumerate(y_preds.items()):
        if model == baseline_key:
            continue

        x, y = alert_rate_curve(y_true, y_pred_proba, n_days, sample=100)
        intersection = find_earliest_intersection(x_intercept, y_intercept, x, y)
        sns.lineplot(
            x=x,
            y=y,
            label=model.replace(" (tuned)", ""),
            linewidth=2,
            ax=ax,
            color=sns.color_palette()[idx],
        )
        if intersection:
            ax.plot(*intersection, marker="x", color="black")
            ax.annotate(
                text=round(intersection[0], 3),
                xy=intersection,
                xytext=(min(1 - 0.015, intersection[0] + 0.045), intersection[1] - 0.4),
            )

    sns.lineplot(
        x=x_intercept,
        y=y_intercept,
        label=baseline_key.replace(" (tuned)", ""),
        linestyle="--",
        linewidth=2,
        ax=ax,
        color="tomato",
    )

    ax.set_title("Sensitivity vs. Alert Rate")
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Mean alerts per day")
    ax.set_xlim(0, 1.08)
    if save:
        plt.savefig(
            save,
            bbox_inches="tight",
            dpi=200 if save_format != "svg" else None,
            format=save_format,
        )

    if no_ax:
        plt.rc("axes", titlesize=12)


def plot_calibrated_regression_coefficients(
    model, columns, topn=60, figsize=(8, 12), save=None
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    df = pd.DataFrame(
        zip(
            np.array(
                [_.base_estimator.coef_[0] for _ in model.calibrated_classifiers_]
            ).mean(axis=0),
            columns,
        ),
        columns=["Coefficient", "Feature"],
    )
    df = df.loc[
        df.Coefficient.apply(abs).sort_values(ascending=False).head(topn).index
    ].sort_values("Coefficient", ascending=False)
    sns.barplot(
        data=df,
        x="Coefficient",
        y="Feature",
        palette=(df.Coefficient > 0).map({True: "r", False: "b"}),
        ax=ax,
    )

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=200)


def plot_roc_curves(
    y_true,
    y_preds,
    baseline_key=None,
    linewidth=2,
    save=None,
    ax=None,
    smoothing=True,
    save_format="png",
):
    no_ax = ax is None
    if no_ax:
        sns.set_style("white")
        plt.rc("axes", titlesize=14)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if type(list(y_preds.values())[0]) == tuple:
        y_preds = {key: value[1] for key, value in y_preds.items()}

    for idx, (modelkey, y_pred_proba) in enumerate(y_preds.items()):
        linestyle = "--" if modelkey == baseline_key else "-"
        color = "tomato" if modelkey == baseline_key else sns.color_palette()[idx]
        if smoothing:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc = roc_auc_score(y_true, y_pred_proba)
            sns.lineplot(
                x=fpr,
                y=tpr,
                label=f"{modelkey.replace(' (tuned)', '')} (AUC = {auc:.2f})",
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
                ax=ax,
            )
        else:
            RocCurveDisplay.from_predictions(
                y_true,
                y_pred_proba,
                ax=ax,
                linewidth=linewidth,
                name=modelkey.replace(" (tuned)", ""),
                linestyle=linestyle,
                color=color,
            )

    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.set_xlabel("1-Specificity")
    ax.set_ylabel("Sensitivity")
    if save:
        plt.savefig(
            save,
            bbox_inches="tight",
            dpi=200 if save_format != "svg" else None,
            format=save_format,
        )

    if no_ax:
        plt.rc("axes", titlesize=12)


def plot_pr_curves(
    y_true,
    y_preds,
    baseline_key=None,
    linewidth=2,
    save=None,
    ax=None,
    smoothing=True,
    save_format="png",
):
    no_ax = ax is None
    if no_ax:
        sns.set_style("white")
        plt.rc("axes", titlesize=14)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if type(list(y_preds.values())[0]) == tuple:
        y_preds = {key: value[1] for key, value in y_preds.items()}

    for idx, (modelkey, y_pred_proba) in enumerate(y_preds.items()):
        linestyle = "--" if modelkey == baseline_key else "-"
        color = "tomato" if modelkey == baseline_key else sns.color_palette()[idx]
        if smoothing:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            ap = average_precision_score(y_true, y_pred_proba)
            sns.lineplot(
                x=recall,
                y=precision,
                label=f'{modelkey.replace(" (tuned)", "")} (AP = {ap:.2f})',
                linewidth=linewidth,
                linestyle=linestyle,
                color=color,
                ax=ax,
            )
        else:
            pr_fig = PrecisionRecallDisplay.from_predictions(
                y_true,
                y_pred_proba,
                name=modelkey.replace(" (tuned)", ""),
                linestyle=linestyle,
                ax=ax,
                linewidth=linewidth,
                color=color,
            )

    ax.legend(loc="upper right")
    ax.set_title("Precision-Recall")
    ax.set_xlabel("Sensitivity (a.k.a. Recall)")
    ax.set_ylabel("Positive predictive value (a.k.a. Precision)")
    if save:
        plt.savefig(
            save,
            bbox_inches="tight",
            dpi=200 if save_format != "svg" else None,
            format=save_format,
        )

    if no_ax:
        plt.rc("axes", titlesize=12)


def plot_calibration_curves(
    y_true, y_preds, linewidth=2, save=None, ax=None, save_format="png"
):
    no_ax = ax is None
    if no_ax:
        sns.set_style("white")
        plt.rc("axes", titlesize=14)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if type(list(y_preds.values())[0]) == tuple:
        y_preds = {key: value[1] for key, value in y_preds.items()}

    for modelkey, y_pred_proba in y_preds.items():
        try:
            CalibrationDisplay.from_predictions(
                y_true,
                y_pred_proba,
                ax=ax,
                linewidth=linewidth,
                name=modelkey.replace(" (tuned)", ""),
            )
        except ValueError:
            pass

    ax.set_title("Calibration")
    if save:
        plt.savefig(
            save,
            bbox_inches="tight",
            dpi=200 if save_format != "svg" else None,
            format=save_format,
        )

    if no_ax:
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


def plot_shap_features_joint(
    shap_values,
    modelkey,
    max_display=20,
    figsize=(16, 8),
    bar_aspect=0.045,
    wspace=-0.3,
    topadjust=0.93,
    save=None,
    save_format="png",
):
    sns.set_style("darkgrid")
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(122, aspect="auto")
    shap.summary_plot(
        shap_values,
        max_display=max_display,
        show=False,
        plot_size=None,
        cmap=plt.get_cmap("coolwarm"),
    )
    ax1.set_yticklabels([])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.tick_params(axis="both", which="minor", labelsize=14)
    ax1.set_xlabel("SHAP value (impact on model output)", fontsize=16)

    ax2 = fig.add_subplot(121, aspect=bar_aspect)
    shap.summary_plot(
        shap_values,
        plot_type="bar",
        plot_size=None,
        max_display=max_display,
        show=False,
        color="purple",
    )
    ax2.set_xlabel("Mean magnitude of SHAP value", fontsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="minor", labelsize=14)
    plt.tight_layout()
    plt.subplots_adjust(wspace=wspace)
    plt.suptitle(modelkey, fontsize=20)
    plt.subplots_adjust(top=topadjust)
    if save:
        plt.savefig(
            save,
            bbox_inches="tight",
            dpi=200 if save_format != "svg" else None,
            format=save_format,
        )


def confusion_matrix_multiplot(y_true, y_preds, save=None, plot_title=None):
    sns.set_style("darkgrid")
    sns.set(font_scale=1.3)
    fig, ax = plt.subplots(1, len(y_preds), figsize=(4 * len(y_preds), 4))

    if type(list(y_preds.values())[0]) == tuple:
        y_preds = {key: value[0] for key, value in y_preds.items()}

    for idx, (modelkey, y_pred) in enumerate(y_preds.items()):
        matrix = plot_confusion_matrix(y_true, y_pred, ax=ax[idx], plot_title=modelkey)
        if idx > 0:
            ax[idx].set_ylabel(None)
        if idx < len(y_preds) - 1:
            matrix.figure_.axes[-1].set_visible(False)

    plt.subplots_adjust(wspace=0, hspace=0)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=400)
    sns.set(font_scale=1)
    # return fig


def plot_confusion_matrix(y_true, y_pred, ax=None, save=None, plot_title=None):
    no_ax = ax is None
    if no_ax:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        plt.rc("axes", titlesize=12)

    ax.grid(False)
    cm_fig = ConfusionMatrixDisplay(
        np.rot90(np.flipud(confusion_matrix(y_true, y_pred, normalize="true"))),
        display_labels=[1, 0],
    ).plot(values_format=".2%", ax=ax, cmap="Purples")

    ax.set_xlabel("True Class")
    ax.set_ylabel("Predicted Class")
    ax.set_title(plot_title)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=200)

    if no_ax:
        plt.rc("axes", titlesize=12)

    return cm_fig


def get_metrics_table(y_true, y_preds, n_resamples=99):
    metrics = []
    for modelkey, (y_pred, y_pred_proba) in y_preds.items():
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

    return pd.DataFrame(metrics).set_index("Model")


def evaluate_multiple(
    y_true,
    y_preds,
    alert_rate_n_days,
    n_resamples=99,
    news_modelkey="Baseline (NEWS)",
    linewidth=2,
    save=None,
):
    metrics = get_metrics_table(y_true, y_preds, n_resamples=n_resamples)

    sns.set_style("white")
    sns.set_palette("tab10")
    plt.rc("axes", titlesize=16)
    fig, ax = plt.subplots(2, 2, figsize=(14, 14))

    plot_roc_curves(
        y_true,
        y_preds,
        baseline_key=news_modelkey,
        ax=ax[0][0],
        linewidth=linewidth,
        smoothing=True,
    )
    plot_pr_curves(
        y_true,
        y_preds,
        baseline_key=news_modelkey,
        ax=ax[0][1],
        linewidth=linewidth,
        smoothing=True,
    )
    plot_calibration_curves(y_true, y_preds, ax=ax[1][0], linewidth=linewidth)
    plot_alert_rate(
        y_true, y_preds, alert_rate_n_days, ax=ax[1][1], baseline_key=news_modelkey
    )

    display(metrics)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=100)

    plt.rc("axes", titlesize=12)
    sns.set_style("darkgrid")


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

        try:
            cal_fig = CalibrationDisplay.from_predictions(
                y, y_pred_proba, ax=ax[2], linewidth=linewidth, name=ylabel
            )
        except ValueError:
            pass

    metrics = pd.DataFrame(metrics).set_index(modelkey)
    display(metrics)

    # cm_fig = plot_confusion_matrix(y_true, y_pred, ax[2])
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper left")

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
