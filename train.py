"""
sklearn-autoresearch training script. Single-file, single-metric.
The agent modifies this file to try different models, features, and hyperparameters.
Usage: python train.py
"""

import time
import traceback

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

def build_pipeline(metadata):
    """
    Build and return an sklearn Pipeline.
    metadata contains: task_type, n_features, n_train, n_classes, feature_names, etc.
    """
    task_type = metadata["task_type"]

    if task_type == "classification":
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
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
    pipeline = build_pipeline(metadata)
    pipeline.fit(X_train, y_train)

    t_train = time.time() - t_start

    # Evaluate on validation set
    results = evaluate(pipeline, X_val, y_val, metadata)

    # Print greppable summary
    print(format_results(results, metadata, training_seconds=t_train))
