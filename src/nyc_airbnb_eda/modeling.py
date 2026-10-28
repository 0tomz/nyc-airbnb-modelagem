"""Model helpers for the NYC Airbnb APS2 regression notebook."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from .preprocessing import CATEGORICAL_FEATURES, build_preprocessor


def build_model_pipeline(estimator) -> Pipeline:
    """Compose the APS1 preprocessor with a fresh estimator."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", clone(estimator)),
    ])


def inverse_log_price(values: np.ndarray) -> np.ndarray:
    """Invert log1p(price), clipping tiny/negative predictions at zero USD."""
    return np.maximum(np.expm1(values), 0.0)


def build_log_target_pipeline(estimator) -> TransformedTargetRegressor:
    """Use the APS1 feature pipeline and log1p only on the training target."""
    return TransformedTargetRegressor(
        regressor=build_model_pipeline(estimator),
        func=np.log1p,
        inverse_func=inverse_log_price,
        check_inverse=False,
    )


def original_feature_name(encoded_name: str) -> str:
    """Map a transformed feature name back to its source column."""
    name = encoded_name.split("__", maxsplit=1)[-1]
    for column in CATEGORICAL_FEATURES:
        if name == column or name.startswith(f"{column}_"):
            return column
    return name


def get_preprocessed_feature_names(preprocessor) -> np.ndarray:
    """Get output names even when FunctionTransformer steps omit name metadata."""
    names: list[str] = []
    for branch, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        source_columns = [str(column) for column in columns]
        if branch == "categorical":
            encoded = transformer.named_steps["onehot"].get_feature_names_out(source_columns)
            names.extend(f"{branch}__{feature}" for feature in encoded)
        else:
            names.extend(f"{branch}__{column}" for column in source_columns)
    return np.asarray(names, dtype=object)




def display_feature_name(encoded_name: str) -> str:
    """Readable label for plots: 'categorical__neighbourhood_Tribeca' -> 'neighbourhood = Tribeca'."""
    branch, _, name = encoded_name.partition("__")
    if branch == "categorical":
        for column in CATEGORICAL_FEATURES:
            if name.startswith(f"{column}_"):
                return f"{column} = {name[len(column) + 1:]}"
    return name or encoded_name
