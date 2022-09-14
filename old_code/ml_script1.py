# %%
import warnings, math, itertools, json, logging
from hashlib import md5

import numpy as np
import pandas as pd

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


# %%
pd.set_option("display.max_columns", None)
sns.set_theme(style="darkgrid", palette="colorblind")
sns.set(rc={"figure.figsize": (11.5, 8.5), "figure.dpi": 100})

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

from dataset import SCIData, SCICols

# %%

# sci.clean_all().filter_vague_diagnoses().derive_readmission().omit_redundant().omit_vbg().omit_ae().save()
X, y = (
    SCIData.load("data/sci_processed.h5")
    .augment_hsmr(onehot=True)
    .impute_blood()
    .raw_news()
    .mandate_news()
    .mandate_diagnoses()
    .xy(outcome="DiedDuringStay")
)

# %%
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

numeric_pipeline = Pipeline(steps=[("scale", MinMaxScaler()),])

categorical_pipeline = Pipeline(
    steps=[("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False)),]
)

ct = make_column_transformer(
    (numeric_pipeline, make_column_selector(dtype_include=np.number)),
    (categorical_pipeline, make_column_selector(dtype_include=object)),
)

full_processor = Pipeline(steps=[("columns", ct)])


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    full_processor.fit_transform(X), y, test_size=0.25
)

# %%
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    make_scorer,
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

# %%
from sklearn.model_selection import cross_validate


def spotCheckCV(model, X, y, cv=3, pretty=True):
    scores = cross_validate(model, X, y, scoring=METRICS, cv=cv)
    if pretty:
        display(
            pd.DataFrame(
                [
                    (name.split("_")[1], sc)
                    for name, score in scores.items()
                    if name.startswith("test")
                    for sc in score
                ],
                columns=["Metric", "Score"],
            )
            .groupby("Metric")
            .agg(
                Mean=pd.NamedAgg(column="Score", aggfunc=np.mean),
                Std=pd.NamedAgg(column="Score", aggfunc=np.std),
            )
        )
    else:
        return scores


# %%
from cross_validation import cross_validate_parallel_file


# %%
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

MODELS = {
    "Logistic Regression": LogisticRegression(random_state=0, C=1e2, max_iter=1000),
    "Gaussian NB": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
}

IMBALANCED_MODELS = {
    "Balanced Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
    "Balanced SVM": SVC(gamma="scale", class_weight="balanced"),
    "Balanced Random Forest": BalancedRandomForestClassifier(
        n_estimators=10, class_weight="balanced_subsample"
    ),
    "Balanced XGBoost": XGBClassifier(
        use_label_encoder=False, scale_pos_weight=21, eval_metric="logloss"
    ),
}

RESAMPLERS = {
    "SMOTE": SMOTE(),
    "Undersampling": RandomUnderSampler(sampling_strategy="majority"),
    "SMOTE-Tomek": SMOTETomek(tomek=TomekLinks(sampling_strategy="majority")),
}

resampled_models = {
    f"{classifier[0]} with {sampler[0]}": ImbPipeline(steps=[sampler, classifier])
    for sampler in RESAMPLERS.items()
    for classifier in MODELS.items()
}

logging.info("############ Starting now ##########################")
# %%
cross_validate_parallel_file(
    filename="test3.csv",
    models={**MODELS, **resampled_models, **IMBALANCED_MODELS},
    X=X_train,
    y=y_train,
    scoring=METRICS,
    cv=5,
)

logging.info("###################### Finished ##############################")
