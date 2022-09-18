import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
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
    .xy(outcome="CriticalEvent", ordinal_encoding=True, dropna=True)
)

categorical_cols_idx, categorical_cols_dims = X.describe_categories()

X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(), y.to_numpy(), stratify=y, test_size=0.25, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, stratify=y_train, test_size=0.33, random_state=42
)

tabnet_params = dict(
    cat_idxs=categorical_cols_idx,
    cat_dims=categorical_cols_dims,
    cat_emb_dim=1,
    optimizer_fn=torch.optim.Adam,
    # optimizer_params=dict(lr=2e-2),
    #   scheduler_params=dict(step_size=50, gamma=0.9),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type="entmax",
    verbose=0,
)

tabnet_model = TabNetClassifier(**tabnet_params)


param_grid = {
    #'batch_size': [300],
    "n_a": np.arange(4, 100, 4),
    "n_d": np.arange(4, 100, 4),
    #'cat_emb_dim': np.arange(1,10,1),
    "optimizer_params": [dict(lr=_) for _ in [2e-2, 1e-2, 3e-2, 5e-2, 1e-1, 8e-3]],
    "scheduler_params": [
        dict(step_size=50, gamma=0.9),
        dict(step_size=50, gamma=0.7),
        dict(step_size=30, gamma=0.9),
        dict(step_size=50, gamma=0.5),
        dict(step_size=80, gamma=0.5),
    ],
    #'virtual_batch_size':[128]
}

clf = RandomizedSearchCV(
    tabnet_model,
    param_distributions=param_grid,
    scoring=f2_score,
    n_iter=300,
    verbose=10,
    n_jobs=-1,
)

clf.fit(X_train, y_train)

fname = random.randint(10000, 100000)

with open(f"tabnet_{fname}.log", "w") as file:
    file.write(str(clf.best_params_) + "\n")
    file.write(f"Best score: {clf.best_score_}")

with open(f"tabnet_{fname}.out", "wb") as file:
    pickle.dump(clf, file)

print("DOne")
