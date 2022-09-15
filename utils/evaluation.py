import pandas as pd
import numpy as np

from IPython.display import display
import matplotlib.pyplot as plt

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

f2_score = make_scorer(fbeta_score, beta=2)

METRICS = {
    "Accuracy": "accuracy",
    "Precision": "precision",
    "Recall": "recall",
    "AUC": "roc_auc",
    "F1 Score": "f1",
    "F2 Score": f2_score,
}

ISOLATION_METRICS = {
    "Accuracy": "accuracy",
    "Precision": make_scorer(precision_score, pos_label=-1),
    "Recall": make_scorer(recall_score, pos_label=-1),
    "AUC": "roc_auc",
    "F1 Score": make_scorer(fbeta_score, pos_label=-1, beta=1),
    "F2 Score": make_scorer(fbeta_score, pos_label=-1, beta=2),
}


def evaluate_from_pred(y_true, y_pred, y_pred_proba, plot_title=None, pos_label=1):
    display(
        pd.DataFrame(
            {
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision": precision_score(y_true, y_pred, pos_label=pos_label),
                "Recall": recall_score(y_true, y_pred, pos_label=pos_label),
                "AUC": roc_auc_score(y_true, y_pred_proba),
                "F1 Score": f1_score(y_true, y_pred, pos_label=pos_label),
                "F2 Score": fbeta_score(y_true, y_pred, beta=2, pos_label=pos_label),
            },
            index=["Model"],
        )
    )

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    ax[2].grid(False)
    RocCurveDisplay.from_predictions(
        y_true, y_pred_proba, ax=ax[0],  # pos_label=pos_label
    )
    PrecisionRecallDisplay.from_predictions(
        y_true, y_pred_proba, ax=ax[1], pos_label=pos_label
    )

    if (-1) in np.array(y_true):
        get = {1: True, -1: False}.get
        y_true, y_pred = (list(map(get, y_true)), list(map(get, y_pred)))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax[2], normalize='true', values_format='.2%')

    plt.suptitle(plot_title)


def evaluate(model, X, y, plot_title=None):
    y_pred = model.predict(X)
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
    except AttributeError:
        try:
            y_pred_proba = model.decision_function(X)
        except AttributeError:
            y_pred_proba = model.score_samples(X)

    evaluate_from_pred(y, y_pred, y_pred_proba, plot_title)
