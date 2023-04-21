import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from functools import partial
import itertools

from scipy import stats as st
import shap

from sklearn.metrics import (
    precision_recall_curve,
    PrecisionRecallDisplay,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    roc_curve,
)
from sklearn.calibration import CalibrationDisplay

from sklearn.model_selection import train_test_split

from shapely.geometry import LineString, Point
import shap

from aif360.sklearn.metrics import (
    generalized_entropy_error,
    between_group_generalized_entropy_error,
    intersection,
)


def bootstrap_metric(metric, y_true, y_score, n_resamples=9999):
    """Computes AUROC with 95% confidence intervals by boostrapping"""
    res = st.bootstrap(
        data=(y_true.to_numpy(), y_score),
        statistic=metric,
        confidence_level=0.95,
        method="percentile",
        n_resamples=n_resamples,
        vectorized=False,
        paired=True,
        random_state=42,
    )

    return res.confidence_interval.low, res.confidence_interval.high


def roc_auc_ci_bootstrap(y_true, y_score, n_resamples=9999):
    """Computes AUROC with 95% confidence intervals by boostrapping"""
    return bootstrap_metric(roc_auc_score, y_true, y_score, n_resamples)


def average_precision_ci_bootstrap(y_true, y_score, n_resamples):
    return bootstrap_metric(average_precision_score, y_true, y_score, n_resamples)


def get_metrics(y_true, y_pred_proba, y_pred_threshold=0.5, n_resamples=99):
    auc_lower, auc_upper = roc_auc_ci_bootstrap(y_true, y_pred_proba, n_resamples)
    ap_lower, ap_upper = average_precision_ci_bootstrap(
        y_true, y_pred_proba, n_resamples
    )

    y_pred = np.where(y_pred_proba > y_pred_threshold, 1, 0)

    return dict(
        AUC=roc_auc_score(y_true, y_pred_proba),
        AUC_Upper=auc_upper,
        AUC_Lower=auc_lower,
        AP=average_precision_score(y_true, y_pred_proba),
        AP_Upper=ap_upper,
        AP_Lower=ap_lower,
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred),
        Recall=recall_score(y_true, y_pred),
        F2=fbeta_score(y_true, y_pred, beta=2),
    )


def _single_confidence_interval_string(score, y_true, y_pred):
    mid = score(y_true, y_pred)
    lower, upper = bootstrap_metric(score, y_true, y_pred, n_resamples=99)
    return f"{mid:.3f} ({lower:.3f}-{upper:.3f})"


def get_decision_metrics(y_true, y_pred, confidence_intervals=False):
    nne = lambda *args, **kwargs: 1 / precision_score(*args, **kwargs)

    scores = dict(
        Sensitivity=recall_score,
        Specificity=partial(recall_score, pos_label=0),
        PPV=precision_score,
        NPV=partial(precision_score, pos_label=0),
        Accuracy=accuracy_score,
        F2=partial(fbeta_score, beta=2),
        NNE=nne,
    )

    if confidence_intervals:
        compute_scores = lambda y_t, y_p: {
            score_name: _single_confidence_interval_string(score, y_t, y_p)
            for score_name, score in scores.items()
        }
    else:
        compute_scores = lambda y_t, y_p: {
            score_name: score(y_t, y_p) for score_name, score in scores.items()
        }

    return compute_scores(y_true, y_pred)


def get_discriminative_metrics(y_true, y_pred_proba, n_resamples=99):
    auc_lower, auc_upper = roc_auc_ci_bootstrap(y_true, y_pred_proba, n_resamples)
    ap_lower, ap_upper = average_precision_ci_bootstrap(
        y_true, y_pred_proba, n_resamples
    )

    return dict(
        AUROC=roc_auc_score(y_true, y_pred_proba),
        AUROC_Upper=auc_upper,
        AUROC_Lower=auc_lower,
        AP=average_precision_score(y_true, y_pred_proba),
        AP_Upper=ap_upper,
        AP_Lower=ap_lower,
    )


def get_decision_threshold(y_true, y_pred_proba, target_recall=0.85):
    """Given prediction probabilities, sets the prediction threshold to approach the given target recall"""

    # Get candidate thresholds from the model, and find the one that gives the best fbeta score
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    closest = thresholds[np.abs(recall - target_recall).argmin()]

    return closest


def get_train_test_indexes(sal, test_size=0.33):
    train_idx, test_idx = train_test_split(
        sal.index, shuffle=False, test_size=test_size
    )
    sal_train, sal_test = sal.loc[train_idx], sal.loc[test_idx]

    is_unseen = ~sal_test.PatientNumber.isin(sal_train.PatientNumber.unique())
    unseen_idx = sal_test[is_unseen].index

    is_precovid = sal_test.AdmissionDateTime < "2020-03-01"
    pre_covid_idx = sal_test[is_precovid].index

    return train_idx, test_idx, unseen_idx, is_unseen, pre_covid_idx, is_precovid


def get_calibrated_regression_coefficients(model, pipeline_key=None):
    models = [_.base_estimator for _ in model.calibrated_classifiers_]
    if pipeline_key:
        models = [_[pipeline_key] for _ in models]

    return pd.DataFrame(
        zip(
            np.array([_.coef_[0] for _ in models]).mean(axis=0), model.feature_names_in_
        ),
        columns=["Coefficient", "Feature"],
    )


def plot_calibration_curves(
    y_true,
    y_preds,
    linewidth=2,
    palette=sns.color_palette("deep"),
    title="Calibration",
    save=None,
    ax=None,
    save_format="png",
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
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
        except ValueError:
            pass

    ax.set_title(title)
    if save:
        plt.savefig(
            save,
            bbox_inches="tight",
            dpi=200 if save_format != "svg" else None,
            format=save_format,
        )

    if no_ax:
        plt.rc("axes", titlesize=12)


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
    alert_rate = np.array(
        [np.where(y_score > threshold, 1, 0).sum() for threshold in thresholds]
    ) / (n_days)

    if sample is not None:
        return (
            recall[np.round(np.linspace(0, len(recall) - 1, sample)).astype(int)],
            alert_rate[
                np.round(np.linspace(0, len(alert_rate) - 1, sample)).astype(int)
            ],
        )
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
    y_true,
    y_preds,
    n_days,
    baseline_key=None,
    intercepts=True,
    ax=None,
    save=None,
    save_format="png",
    title="Sensitivity vs. Alert Rate",
    xlim=(0, 1.08),
    ylim=None,
):
    no_ax = ax is None
    if no_ax:
        sns.set_style("white")
        plt.rc("axes", titlesize=14)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if type(list(y_preds.values())[0]) == tuple:
        y_preds = {key: value[1] for key, value in y_preds.items()}

    if baseline_key:
        x_intercept, y_intercept = alert_rate_curve(
            y_true, y_preds[baseline_key], n_days, sample=None
        )

    for idx, (model, y_pred_proba) in enumerate(y_preds.items()):
        if model == baseline_key:
            continue

        x, y = alert_rate_curve(y_true, y_pred_proba, n_days, sample=200)
        intersection = None
        if baseline_key and intercepts:
            try:
                intersection = find_earliest_intersection(
                    x_intercept, y_intercept, x, y
                )
            except StopIteration:
                pass
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

    if baseline_key:
        sns.lineplot(
            x=x_intercept,
            y=y_intercept,
            label=baseline_key.replace(" (tuned)", ""),
            linestyle="--",
            linewidth=2,
            ax=ax,
            color="tomato",
        )

    ax.set_title(title)
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("Mean alerts per day")
    ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if save:
        plt.savefig(
            save,
            bbox_inches="tight",
            dpi=200 if save_format != "svg" else None,
            format=save_format,
        )

    if no_ax:
        plt.rc("axes", titlesize=12)


# return fig, ax


def plot_pr_curves(
    y_true,
    y_preds,
    baseline_key=None,
    linewidth=2,
    save=None,
    ax=None,
    smoothing=True,
    save_format="png",
    palette=sns.color_palette("deep"),
    title="Precision-Recall",
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
        color = "tomato" if modelkey == baseline_key else palette[idx]
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
    # sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title)
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel("PPV (Precision)")
    if save:
        plt.savefig(
            save,
            bbox_inches="tight",
            dpi=200 if save_format != "svg" else None,
            format=save_format,
        )

    if no_ax:
        plt.rc("axes", titlesize=12)


def plot_shap_features_joint(
    shap_values,
    title=None,
    max_display=20,
    figsize=(16, 8),
    bar_aspect=0.045,
    wspace=-0.3,
    topadjust=0.93,
    save=None,
    save_format="png",
):
    sns.set_style("whitegrid")
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
    ax1.set_xlabel("(b) Episode-individual SHAP value", fontsize=16)
    ax1.set_xlim((-2.5, 3))

    ax2 = fig.add_subplot(121, aspect=bar_aspect)
    shap.summary_plot(
        shap_values,
        plot_type="bar",
        plot_size=None,
        max_display=max_display,
        show=False,
        color="purple",
    )
    ax2.set_xlabel("(a) Mean absolute SHAP value", fontsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="minor", labelsize=14)
    plt.tight_layout()
    plt.subplots_adjust(wspace=wspace)
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(top=topadjust)
    if save:
        plt.savefig(
            save,
            bbox_inches="tight",
            dpi=200 if save_format != "svg" else None,
            format=save_format,
        )


def generalized_entropy_curve(
    y_true, y_pred_proba, prot_attr=None, function=generalized_entropy_error
):
    _, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    if function == generalized_entropy_error:
        r = [function(y_true, np.where(y_pred_proba > _, 1, 0)) for _ in thresholds]
    elif function == between_group_generalized_entropy_error:
        r = [
            between_group_generalized_entropy_error(
                y_true, np.where(y_pred_proba > _, 1, 0), prot_attr=prot_attr
            )
            for _ in thresholds
        ]
    return recall[:-1], r


def plot_entropy_curves(
    y_true,
    y_preds,
    prot_attr=None,
    baseline_key=None,
    ylabel="Generalised Entropy Index",
    ax=None,
    palette=sns.color_palette("deep"),
    title="Equalised Odds",
    function=generalized_entropy_error,
):
    no_ax = ax is None
    if no_ax:
        sns.set_style("white")
        plt.rc("axes", titlesize=14)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    for idx, (modelkey, y_pred_proba) in enumerate(y_preds.items()):
        linestyle = "--" if modelkey == baseline_key else "-"
        color = "tomato" if modelkey == baseline_key else palette[idx]
        x, y = generalized_entropy_curve(
            y_true, y_pred_proba, prot_attr, function=function
        )
        sns.lineplot(
            x=x,
            y=y,
            label=modelkey,
            linewidth=2,
            linestyle=linestyle,
            color=color,
            ax=ax,
        )

    ax.legend(loc="upper right")
    # sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(title)
    ax.set_xlabel("Sensitivity")
    ax.set_ylabel(ylabel)


import matplotlib.cm as cm


def plot_clustered_stacked(
    dfall,
    labels=None,
    legend_titles=[None, None],
    title="multiple stacked bar plot",
    H=["", "//", "XX", ".."],
    **kwargs,
):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
    labels is a list of the names of the dataframe, used for the legend
    title is a string for the title of the plot
    H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    sns.set_style("whitegrid")

    axe = plt.subplot(111)

    for df in dfall:  # for each data frame
        axe = df.plot(
            kind="bar",
            linewidth=0,
            stacked=True,
            ax=axe,
            legend=False,
            grid=False,
            **kwargs,
        )  # make bar plots

    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i : i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H[int(i / n_col)])  # edited part
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.0)
    axe.set_xticklabels(df.index, rotation=0)
    axe.set_title(title)

    # Add invisible data to add another legend
    n = []
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H[i]))
    plt.grid(axis="y")
    l1 = axe.legend(
        h[:n_col], l[:n_col], loc=[1.01, 0.5], frameon=False, title=legend_titles[0]
    )
    if labels is not None:
        l2 = plt.legend(
            n, labels, loc=[1.01, 0.1], frameon=False, title=legend_titles[1]
        )
    axe.add_artist(l1)
    return axe


def soft_base_rate(
    y_true, y_pred=None, *, concentration=1.0, pos_label=1, sample_weight=None
):
    return (np.sum(y_true) + concentration) / (y_true.shape[0] + concentration)


def soft_selection_rate(
    y_true, y_pred, *, concentration=1.0, pos_label=1, sample_weight=None
):
    return soft_base_rate(
        y_pred,
        concentration=concentration,
        pos_label=pos_label,
        sample_weight=sample_weight,
    )


def soft_edf(
    y_true,
    y_pred=None,
    *,
    prot_attr=None,
    pos_label=1,
    concentration=1.0,
    sample_weight=None,
):
    rate = soft_base_rate if y_pred is None else soft_selection_rate
    sbr = intersection(rate, y_true, y_pred, prot_attr=prot_attr)
    logsbr = np.log(sbr)
    pos_ratio = max(abs(i - j) for i, j in itertools.permutations(logsbr, 2))
    lognegsbr = np.log(1 - np.array(sbr))
    neg_ratio = max(abs(i - j) for i, j in itertools.permutations(lognegsbr, 2))
    return max(pos_ratio, neg_ratio)


def soft_df_bias_amplification(
    y_true, y_pred, prot_attr, pos_label=1, concentration=1.0, sample_weight=None
):
    eps_true = soft_edf(
        y_true,
        prot_attr=prot_attr,
        pos_label=pos_label,
        concentration=concentration,
        sample_weight=sample_weight,
    )
    eps_pred = soft_edf(
        y_true,
        y_pred,
        prot_attr=prot_attr,
        pos_label=pos_label,
        concentration=concentration,
        sample_weight=sample_weight,
    )
    return eps_pred - eps_true


def bootstrap_bias_amplification(y_true, y_score, prot_attr, n_resamples=99):
    center = soft_df_bias_amplification(y_true, y_score, prot_attr)
    res = st.bootstrap(
        data=(y_true.to_numpy(), y_score, prot_attr.to_numpy()),
        statistic=soft_df_bias_amplification,
        confidence_level=0.95,
        method="percentile",
        n_resamples=n_resamples,
        vectorized=False,
        paired=True,
        random_state=42,
    )
    return res.confidence_interval.low, center, res.confidence_interval.high
