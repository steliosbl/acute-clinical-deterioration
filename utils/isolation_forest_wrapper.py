import numpy as np
from sklearn.ensemble import IsolationForest


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
