"""
One-time data preparation and fixed evaluation for sklearn-autoresearch.
Loads any tabular dataset (CSV or sklearn built-in), splits it, and provides
the evaluation harness. This file is READ-ONLY for the agent.

Usage:
    python prepare.py                           # interactive dataset selection
    python prepare.py --dataset california       # sklearn built-in dataset
    python prepare.py --csv path/to/data.csv --target column_name

Data is stored in ~/.cache/sklearn-autoresearch/.
"""

import os
import sys
import json
import hashlib
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, r2_score, root_mean_squared_error,
    roc_auc_score, mean_absolute_error, classification_report,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TEST_SIZE = 0.1           # hold-out test fraction
VAL_SIZE = 0.1            # validation fraction (from training set)
RANDOM_STATE = 42         # reproducible splits
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "sklearn-autoresearch")

# ---------------------------------------------------------------------------
# Built-in datasets
# ---------------------------------------------------------------------------

BUILTIN_DATASETS = {
    "california": {
        "loader": "sklearn.datasets.fetch_california_housing",
        "description": "California housing prices (regression)",
    },
    "iris": {
        "loader": "sklearn.datasets.load_iris",
        "description": "Iris flower classification (classification)",
    },
    "wine": {
        "loader": "sklearn.datasets.load_wine",
        "description": "Wine recognition (classification)",
    },
    "breast_cancer": {
        "loader": "sklearn.datasets.load_breast_cancer",
        "description": "Breast cancer diagnosis (classification)",
    },
    "diabetes": {
        "loader": "sklearn.datasets.load_diabetes",
        "description": "Diabetes progression (regression)",
    },
    "digits": {
        "loader": "sklearn.datasets.load_digits",
        "description": "Handwritten digits 0-9 (classification)",
    },
}

# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------

def detect_task_type(y):
    """Auto-detect classification vs regression from the target column."""
    # String/object types are always classification
    if hasattr(y, "dtype"):
        dtype_str = str(y.dtype).lower()
        if "string" in dtype_str or "object" in dtype_str or "category" in dtype_str:
            return "classification"
        try:
            if np.issubdtype(y.dtype, np.floating):
                n_unique = len(np.unique(y))
                return "classification" if n_unique <= 20 else "regression"
            if np.issubdtype(y.dtype, np.integer):
                n_unique = len(np.unique(y))
                return "classification" if n_unique <= 50 else "regression"
        except TypeError:
            return "classification"
    # Fallback: check if values look categorical
    n_unique = len(set(y))
    if n_unique <= 20:
        return "classification"
    return "regression"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_builtin(name):
    """Load a sklearn built-in dataset, return (df, target_column_name)."""
    import importlib
    info = BUILTIN_DATASETS[name]
    module_path, func_name = info["loader"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    loader = getattr(module, func_name)
    data = loader()
    feature_names = (
        data.feature_names if hasattr(data, "feature_names") else
        [f"feature_{i}" for i in range(data.data.shape[1])]
    )
    df = pd.DataFrame(data.data, columns=list(feature_names))
    df["target"] = data.target
    return df, "target"


def _load_csv(csv_path, target_column, feature_columns=None, ignore_columns=None):
    """Load a CSV file, return (df, target_column_name).

    feature_columns: if provided, only these columns (+ target) are kept.
    ignore_columns: if provided, these columns are dropped.
    """
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        print(f"Error: target column '{target_column}' not found in CSV.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    if feature_columns and ignore_columns:
        print("Error: --features and --ignore are mutually exclusive.")
        sys.exit(1)

    if feature_columns:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            print(f"Error: feature columns not found in CSV: {missing}")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        keep = feature_columns + [target_column]
        df = df[keep]
        print(f"Using {len(feature_columns)} specified feature columns")

    if ignore_columns:
        missing = [c for c in ignore_columns if c not in df.columns]
        if missing:
            print(f"Warning: ignore columns not found (skipping): {missing}")
        to_drop = [c for c in ignore_columns if c in df.columns and c != target_column]
        df = df.drop(columns=to_drop)
        print(f"Ignored {len(to_drop)} columns: {to_drop}")

    return df, target_column


def _dataset_hash(df, target_col):
    """Deterministic hash for a dataset to cache splits."""
    content = f"{df.shape}_{target_col}_{df.columns.tolist()}_{df.iloc[0].tolist()}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def prepare_data(df, target_col):
    """
    Split data into train/val/test and save to cache.
    Returns metadata dict.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Convert Arrow-backed string columns to object dtype for numpy compatibility
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("object")

    y = df[target_col].values
    X = df.drop(columns=[target_col])

    task_type = detect_task_type(y)

    # Stratified split when possible, fallback to non-stratified for rare classes
    def safe_split(X, y, test_size, task_type):
        if task_type == "classification":
            try:
                return train_test_split(X, y, test_size=test_size,
                                        random_state=RANDOM_STATE, stratify=y)
            except ValueError:
                print("Warning: some classes too rare for stratified split, using shuffle split")
        return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = safe_split(X, y, TEST_SIZE, task_type)

    # Second split: train vs val
    X_train, X_val, y_train, y_val = safe_split(X_trainval, y_trainval, VAL_SIZE, task_type)

    # Save splits
    np.save(os.path.join(CACHE_DIR, "X_train.npy"), X_train.values, allow_pickle=True)
    np.save(os.path.join(CACHE_DIR, "X_val.npy"), X_val.values, allow_pickle=True)
    np.save(os.path.join(CACHE_DIR, "X_test.npy"), X_test.values, allow_pickle=True)
    np.save(os.path.join(CACHE_DIR, "y_train.npy"), y_train, allow_pickle=True)
    np.save(os.path.join(CACHE_DIR, "y_val.npy"), y_val, allow_pickle=True)
    np.save(os.path.join(CACHE_DIR, "y_test.npy"), y_test, allow_pickle=True)

    # Save full dataframes (preserving column names and dtypes for feature engineering)
    X_train.to_parquet(os.path.join(CACHE_DIR, "X_train.parquet"))
    X_val.to_parquet(os.path.join(CACHE_DIR, "X_val.parquet"))
    X_test.to_parquet(os.path.join(CACHE_DIR, "X_test.parquet"))

    # Metadata
    n_classes = int(len(np.unique(y))) if task_type == "classification" else 0
    metadata = {
        "task_type": task_type,
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_classes": n_classes,
        "feature_names": list(X.columns),
        "feature_dtypes": {col: str(X[col].dtype) for col in X.columns},
        "target_column": target_col,
        "metric": "f1_weighted" if task_type == "classification" else "rmse",
        "metric_direction": "higher_is_better" if task_type == "classification" else "lower_is_better",
    }
    with open(os.path.join(CACHE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_metadata():
    """Load dataset metadata."""
    meta_path = os.path.join(CACHE_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        print("Error: No prepared data found. Run prepare.py first.")
        sys.exit(1)
    with open(meta_path) as f:
        return json.load(f)


def load_train_data():
    """Load training features and labels as DataFrames/arrays."""
    X = pd.read_parquet(os.path.join(CACHE_DIR, "X_train.parquet"))
    y = np.load(os.path.join(CACHE_DIR, "y_train.npy"), allow_pickle=True)
    return X, y


def load_val_data():
    """Load validation features and labels as DataFrames/arrays."""
    X = pd.read_parquet(os.path.join(CACHE_DIR, "X_val.parquet"))
    y = np.load(os.path.join(CACHE_DIR, "y_val.npy"), allow_pickle=True)
    return X, y


def load_test_data():
    """Load test features and labels as DataFrames/arrays."""
    X = pd.read_parquet(os.path.join(CACHE_DIR, "X_test.parquet"))
    y = np.load(os.path.join(CACHE_DIR, "y_test.npy"), allow_pickle=True)
    return X, y


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE -- this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate(model, X, y, metadata=None):
    """
    Evaluate a fitted model on the given data.
    Returns a dict with the primary metric and supporting metrics.

    The primary metric is:
      - classification: weighted F1 score (higher is better)
      - regression: RMSE (lower is better)
    """
    if metadata is None:
        metadata = load_metadata()

    task_type = metadata["task_type"]
    y_pred = model.predict(X)

    if task_type == "classification":
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
        results = {
            "primary_metric": "f1_weighted",
            "f1_weighted": round(f1, 6),
            "accuracy": round(acc, 6),
        }
        # AUC if binary classification and model supports predict_proba
        if metadata["n_classes"] == 2 and hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)[:, 1]
                results["roc_auc"] = round(roc_auc_score(y, y_proba), 6)
            except Exception:
                pass
    else:
        rmse = root_mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        results = {
            "primary_metric": "rmse",
            "rmse": round(rmse, 6),
            "r2": round(r2, 6),
            "mae": round(mae, 6),
        }

    return results


def is_improvement(current_score, best_score, metadata=None):
    """
    Returns True if current_score is better than best_score.
    Handles both higher-is-better and lower-is-better metrics.
    """
    if metadata is None:
        metadata = load_metadata()
    if metadata["metric_direction"] == "higher_is_better":
        return current_score > best_score
    else:
        return current_score < best_score


def format_results(results, metadata=None, training_seconds=0.0):
    """Format results in a greppable summary block (mirrors autoresearch output)."""
    if metadata is None:
        metadata = load_metadata()

    metric_name = results["primary_metric"]
    metric_value = results[metric_name]

    lines = ["---"]
    lines.append(f"{metric_name}:       {metric_value:.6f}")
    lines.append(f"training_seconds: {training_seconds:.1f}")

    for key, value in results.items():
        if key == "primary_metric" or key == metric_name:
            continue
        if isinstance(value, float):
            lines.append(f"{key}:{'':>{15 - len(key)}} {value:.6f}")
        else:
            lines.append(f"{key}:{'':>{15 - len(key)}} {value}")

    lines.append(f"task_type:        {metadata['task_type']}")
    lines.append(f"n_features:       {metadata['n_features']}")
    lines.append(f"n_train:          {metadata['n_train']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for sklearn-autoresearch")
    parser.add_argument("--dataset", type=str, choices=list(BUILTIN_DATASETS.keys()),
                        help="Use a sklearn built-in dataset")
    parser.add_argument("--csv", type=str, help="Path to a CSV file")
    parser.add_argument("--target", type=str, help="Target column name (required with --csv)")
    parser.add_argument("--features", type=str, nargs="+",
                        help="Explicit list of feature columns to use (mutually exclusive with --ignore)")
    parser.add_argument("--ignore", type=str, nargs="+",
                        help="Columns to drop (e.g. IDs, timestamps). Mutually exclusive with --features")
    args = parser.parse_args()

    if args.csv:
        if not args.target:
            print("Error: --target is required when using --csv")
            sys.exit(1)
        print(f"Loading CSV: {args.csv}")
        df, target_col = _load_csv(args.csv, args.target,
                                   feature_columns=args.features,
                                   ignore_columns=args.ignore)
    elif args.dataset:
        print(f"Loading built-in dataset: {args.dataset}")
        print(f"  {BUILTIN_DATASETS[args.dataset]['description']}")
        df, target_col = _load_builtin(args.dataset)
    else:
        print("Available built-in datasets:")
        for name, info in BUILTIN_DATASETS.items():
            print(f"  {name:20s} — {info['description']}")
        print()
        print("Usage:")
        print("  python prepare.py --dataset california")
        print("  python prepare.py --csv data.csv --target price")
        print("  python prepare.py --csv data.csv --target price --ignore id timestamp")
        print("  python prepare.py --csv data.csv --target price --features age income zipcode")
        sys.exit(0)

    print(f"\nDataset shape: {df.shape}")
    print(f"Target column: {target_col}")

    metadata = prepare_data(df, target_col)

    print(f"\nTask type: {metadata['task_type']}")
    print(f"Primary metric: {metadata['metric']} ({metadata['metric_direction']})")
    print(f"Train: {metadata['n_train']} | Val: {metadata['n_val']} | Test: {metadata['n_test']}")
    print(f"Features: {metadata['n_features']}")
    if metadata['n_classes'] > 0:
        print(f"Classes: {metadata['n_classes']}")
    print(f"\nData cached at: {CACHE_DIR}")
    print("Done! Ready to train.")
