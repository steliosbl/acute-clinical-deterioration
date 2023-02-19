import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def isolationforestsplit(X_train, y_train, kf):
    for train_index, test_index in kf.split(X_train, y_train):
        if type(y_train) == pd.Series:
            y_train = y_train.to_numpy()
        trainy, validy = (
            y_train[train_index],
            y_train[test_index],
        )

        test_index = np.concatenate((train_index[trainy], test_index))
        train_index = train_index[~trainy]
        yield train_index, test_index


class IsolationForestWrapper(IsolationForest):
    """ Wraps the scikit-learn Isolation Forest model to adapt it to our test harness. This is because the original gives non-standard outputs when predicting """

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
