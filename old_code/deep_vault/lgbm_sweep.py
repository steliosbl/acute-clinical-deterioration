import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pickle
import numpy as np
import pandas as pd
from evaluation import f2_score
import random

from dataset import SCIData, SCICols

# SCIData.load('data/sci.h5').clean_all().filter_vague_diagnoses().derive_readmission().omit_vbg()
sci = (
    SCIData.load("data/sci_processed_2.h5")
    .fix_readmissionband()
    .derive_critical_event(within=2)
)

X, y = (
    sci.omit_redundant()
    .drop(["ReadmissionBand", "AgeBand"], axis=1)
    .omit_ae()
    .raw_news()
    .mandate_news()
    .mandate_blood()
    .augment_hsmr()
    .encode_ccs_onehot()
    .xy(
        outcome="CriticalEvent", dropna=True
    )  # Use ordinal encoding because of bug in XGB that prevents use of SHAP when pd.categorical is involved
)
categorical_cols_idx = X.describe_categories()[0]
X = X.ordinal_encode_categories()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.33, random_state=42
)

print(X_train.shape)


model = ImbPipeline(
    steps=[
        ("smote", SMOTENC(categorical_features=categorical_cols_idx)),
        ("LGBM", LGBMClassifier(metric=["auc"], n_jobs=5)),
    ]
)

param_grid = {
    "LGBM__learning_rate": [0.01, 0.025, 0.05, 0.085, 0.1, 0.15, 0.2, 0.25, 0.3],
    "LGBM__boosting_type": ["gbdt", "dart", "goss"],
    # "LGBM__sub_feature": np.arange(0, 1, 0.05),
    "LGBM__num_leaves": np.arange(20, 300, 20),
    "LGBM__min_child_samples": np.arange(10, 100, 10),
    "LGBM__max_depth": np.arange(5, 200, 20),
    "LGBM__scale_pos_weight": np.arange(1, 60, 2),
    "LGBM__subsample": np.arange(0.3, 1.0, 0.05),
    "smote__sampling_strategy": [0.1, 0.25, 0.5, 0.75, 1]
}

clf = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    scoring=f2_score,
    n_iter=3000,
    verbose=10,
    n_jobs=5,
)

clf.fit(X_train, y_train)

fname = random.randint(10000, 100000)

with open(f"lgbm_smote_{fname}.log", "w") as file:
    file.write(str(clf.best_params_) + "\n")
    file.write(f"Best score: {clf.best_score_}")

with open(f"lgbm_smote_{fname}.out", "wb") as file:
    pickle.dump(clf, file)

print("DOne")
