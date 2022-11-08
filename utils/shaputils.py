import numpy as np
import pandas as pd
import shap


def get_joined_categorical_string(X, categorical_cols, separator="__"):
    return (
        X[categorical_cols]
        .eq(1)
        .dot(pd.Index([_.split(separator)[1] for _ in categorical_cols]) + ", ")
        .str[:-2]
    )


def group_explanations_by_categorical(explanations, X, categorical_groups):
    idxs_to_exclude = []
    summed_shap_values = []
    if len(explanations.shape) > 2:
        explanations = explanations[:, :, 1]
    for group, col_names in categorical_groups.items():
        idxs = [X.columns.get_loc(_) for _ in col_names]
        idxs_to_exclude += idxs
        summed_shap_values.append(explanations.values[:, idxs].sum(axis=1))

    joined_categorical_data = [
        get_joined_categorical_string(X, _).values[:, np.newaxis]
        for _ in categorical_groups.values()
    ]

    idxs_to_include = list(set(range(X.shape[1])) - set(idxs_to_exclude))

    r = shap.Explanation(
        data=np.concatenate(
            [explanations.data[:, idxs_to_include]] + joined_categorical_data, axis=1,
        ),
        base_values=explanations.base_values,
        values=np.concatenate(
            [explanations.values[:, idxs_to_include]]
            + [_[:, np.newaxis] for _ in summed_shap_values],
            axis=1,
        ),
        feature_names=list(X.columns[idxs_to_include])
        + list(categorical_groups.keys()),
    )

    return r
