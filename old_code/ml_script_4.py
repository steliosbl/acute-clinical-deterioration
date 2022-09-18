import numpy as np
import pandas as pd

from evaluation import f2_score, METRICS

from collections import defaultdict
from sklearn.model_selection import cross_validate
from sklearn.base import clone as clone_estimator
from multiprocessing import Pool

from evaluation import METRICS


def cvf(datakey, xy, modelkey, model, fitparams):
    return pd.DataFrame.from_dict(
        cross_validate(
            model,
            X=xy[0],
            y=xy[1],
            cv=5,
            n_jobs=1,
            fit_params=fitparams,
            scoring=METRICS,
        )
    ).assign(dataset=datakey, model=modelkey)


def spotCheckDatasets(
    models,
    datasets,
    cv=5,
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

    print("Spinning up 5 processes")
    with Pool(24) as p:
        results = p.starmap(cvf, tests)

    return pd.concat(results).groupby(["dataset", "model"]).mean()


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


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.base import clone as clone_estimator


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


from dataset import SCIData, SCICols

# SCIData.load('data/sci.h5').clean_all().filter_vague_diagnoses().derive_readmission().omit_vbg()
sci = (
    SCIData.load("data/sci_processed_2.h5")
    .fix_readmissionband()
    .derive_critical_event(within=2)
)

scii = (
    sci.omit_redundant()
    .drop(["ReadmissionBand", "AgeBand"], axis=1)
    .omit_ae()
    .raw_news()
)

datasets_rf = {
    "Mandated vitals, One-hot diagnoses": (
        scii.mandate_news()
        .mandate_blood()
        .augment_hsmr()
        .encode_ccs_onehot()
        .xy(outcome="CriticalEvent", ordinal_encoding=True, dropna=True)
    ),
    "Mandated vitals, Categorical diagnoses (main only)": (
        scii.mandate_news()
        .mandate_blood()
        .augment_hsmr()
        .drop(SCICols.diagnoses[1:], axis=1)
        .xy(outcome="CriticalEvent", ordinal_encoding=True, dropna=True)
    ),
    "Mandated vitals, Categorical diagnoses (with missing)": (
        scii.mandate_news()
        .mandate_blood()
        .augment_hsmr()
        .drop(SCICols.diagnoses[1:], axis=1)
        .xy(outcome="CriticalEvent", ordinal_encoding=True, fillna=True)
    ),
    "Imputed vitals": (
        scii.impute_news()
        .impute_blood()
        .augment_hsmr()
        .encode_ccs_onehot()
        .xy(outcome="CriticalEvent", ordinal_encoding=True, dropna=True)
    ),
    "Missing NEWS, imputed blood": (
        scii.augment_hsmr()
        .impute_blood()
        .encode_ccs_onehot()
        .mandate_diagnoses()
        .xy(outcome="CriticalEvent", ordinal_encoding=True, fillna=True)
    ),
    "Missing vitals": (
        scii.augment_hsmr()
        .impute_blood()
        .encode_ccs_onehot()
        .mandate_diagnoses()
        .xy(outcome="CriticalEvent", ordinal_encoding=True, fillna=True)
    ),
}

from sklearn.ensemble import IsolationForest


class IsolationForestWrapper(IsolationForest):
    def __init__(
        self,
        *,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
    ):
        super().__init__(
            bootstrap=bootstrap,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            contamination=contamination,
        )

    def predict(self, X):
        return np.fromiter(map({-1: 0, 1: 1}.get, super().predict(X)), dtype=int)

    def decision_function(self, X):
        return -super().decision_function(X)


from sklearn.metrics import fbeta_score
from pytorch_tabnet.metrics import Metric


class F2TabNet(Metric):
    def __init__(self):
        self._name = "F2"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.argmax(y_score, axis=1)
        return fbeta_score(y_true, y_pred, beta=2)


import torch
from pytorch_tabnet.tab_model import TabNetClassifier

tabnet_params = dict(
    n_a=32,
    n_d=32,
    # cat_idxs=categorical_cols_idx,
    # cat_dims=categorical_cols_dims,
    cat_emb_dim=1,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params=dict(step_size=50, gamma=0.9),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type="entmax",
    verbose=False,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

dataset = datasets_rf["Mandated vitals, One-hot diagnoses"]
categorical_cols_idx = dataset[0].describe_categories()[0]
contamination = dataset[1].sum() / dataset[1].shape[0]
models = {
    "Approx XGB": XGBClassifier(tree_method="approx", scale_pos_weight=1, missing=-1),
    "Approx XGB Balanced": XGBClassifier(
        tree_method="approx", missing=-1, scale_pos_weight=30
    ),
    "Hist XGB": XGBClassifier(tree_method="hist", scale_pos_weight=1, missing=-1),
    "Hist XGB Balanced": XGBClassifier(
        tree_method="hist", missing=-1, scale_pos_weight=30
    ),
    **with_sampling_strategies(
        XGBClassifier(tree_method="hist", scale_pos_weight=30, missing=-1),
        "XGB",
        categorical_cols_idx,
    ),
    "Random Forest": RandomForestClassifier(),
    "Random Forest (balanced)": RandomForestClassifier(
        class_weight="balanced_subsample"
    ),
    **with_sampling_strategies(
        RandomForestClassifier(), "Random Forest", categorical_cols_idx
    ),
    "Isolation Forest": IsolationForestWrapper(),
    "Isolation Forest (contamination)": IsolationForestWrapper(
        contamination=contamination
    ),
    "LightGBM": LGBMClassifier(metric=["l2", "auc"]),
    "LightGBM Balanced": LGBMClassifier(metric=["l2", "auc"], is_unbalance=True),
    "LightGBM Weighted": LGBMClassifier(metric=["l2", "auc"], scale_pos_weight=30),
    "TabNet": TabNetClassifier(**tabnet_params),
    **with_sampling_strategies(
        TabNetClassifier(**tabnet_params), "TabNet", categorical_cols_idx
    ),
}

fit_params = {
    "TabNet": dict(
        max_epochs=100,
        # patience=50,
        batch_size=512,
        virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False,
    )
}

datasets = {
    key: (X.to_numpy(), y.to_numpy()) for key, (X, y) in list(datasets_rf.items())[0:1]
}

result = spotCheckDatasets(models, datasets, fit_params=fit_params)

result.to_csv("dataset_results.csv")

