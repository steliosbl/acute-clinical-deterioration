from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pickle
from evaluation import f2_score

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
        outcome="CriticalEvent", dropna=False, ordinal_encoding=True
    )  # Use ordinal encoding because of bug in XGB that prevents use of SHAP when pd.categorical is involved
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.33, random_state=42
)


model = ImbPipeline(
    steps=[
        ("undersampling", RandomUnderSampler(sampling_strategy=0.1)),
        (
            "XGB",
            XGBClassifier(
                tree_method="approx",
                enable_categorical=True,
                scale_pos_weight=(y_train.shape[0] / (2 * y_train.sum())),
            ),
        ),
    ]
)

param_grid = {
    "XGB__max_depth": [3, 5, 6, 10, 15, 20],
    "XGB__learning_rate": [0.01, 0.1, 0.2, 0.3],
    "XGB__subsample": np.arange(0.5, 1.0, 0.1),
    "XGB__colsample_bytree": np.arange(0.4, 1.0, 0.1),
    "XGB__colsample_bylevel": np.arange(0.4, 1.0, 0.1),
    "XGB__n_estimators": [100, 500, 1000],
}

clf = RandomizedSearchCV(
    model, param_distributions=param_grid, scoring=f2_score, n_iter=25, verbose=1
)

clf.fit(X_train, y_train)

with open("xgb_tuned.log", "w") as file:
    file.write(str(clf.best_params_) + "\n")
    file.write(f"Best score: {clf.best_score_}")

pickle.dump(clf, "xgb_tuned.out")
