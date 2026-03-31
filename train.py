"""
sklearn-autoresearch training script. Single-file, single-metric.
The agent modifies this file to try different models, features, and hyperparameters.
Usage: python train.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from prepare import (
    load_metadata, load_train_data, load_val_data,
    evaluate, format_results,
)

# ---------------------------------------------------------------------------
# Feature Engineering (edit this section)
# ---------------------------------------------------------------------------

def build_features(X):
    """
    Apply feature transformations. Receives raw training/val/test DataFrame.
    Return transformed DataFrame or array.
    """
    return X

# ---------------------------------------------------------------------------
# Model & Pipeline (edit this section)
# ---------------------------------------------------------------------------

def build_pipeline(metadata, X_train):
    """
    Build and return an sklearn Pipeline.
    metadata contains: task_type, n_features, n_train, n_classes, feature_names, etc.
    """
    task_type = metadata["task_type"]

    # Detect column types
    cat_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    # HistGradientBoosting handles NaN natively, only need to encode categoricals
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ])

    if task_type == "classification":
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            early_stopping=False,
        )
    else:
        model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
        )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline

# ---------------------------------------------------------------------------
# Training (runs automatically — no need to edit below this line)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    metadata = load_metadata()
    X_train, y_train = load_train_data()
    X_val, y_val = load_val_data()

    # Apply feature engineering
    X_train = build_features(X_train)
    X_val = build_features(X_val)

    # Build and fit
    pipeline = build_pipeline(metadata, X_train)
    pipeline.fit(X_train, y_train)

    t_train = time.time() - t_start

    # Evaluate on validation set
    results = evaluate(pipeline, X_val, y_val, metadata)

    # Print greppable summary
    print(format_results(results, metadata, training_seconds=t_train))
