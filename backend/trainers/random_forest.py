import os
import time
from datetime import datetime
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from utils import metrics as metrics_module  # your utils/metrics.py


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _model_filename(base_name: str, task: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{task}_{ts}.joblib"


def train(X_train, y_train, X_val, y_val, config=None):
    """
    Train Random Forest for classification or regression.
    Args:
        X_train, X_val: numpy arrays or array-like (already preprocessed)
        y_train, y_val: 1d arrays
        config: dict with optional keys:
            - task: "classification" or "regression" (required)
            - n_estimators: int (default 100)
            - max_depth: int or None
            - random_state: int
            - model_dir: directory to save models (default "models")
    Returns:
        result dict:
        {
          "model_name": "Random Forest",
          "model_path": "models/...",
          "metrics": {...},
          "train_time": 12.34,
          "model_size": 123456,
          "extra": {...}
        }
    """

    if config is None:
        config = {}

    task = config.get("task")
    if task not in ("classification", "regression"):
        raise ValueError("config['task'] must be 'classification' or 'regression'")

    n_estimators = config.get("n_estimators", 100)
    max_depth = config.get("max_depth", None)
    random_state = config.get("random_state", 42)
    model_dir = config.get("model_dir", "models")

    _ensure_dir(model_dir)

    # Convert to numpy arrays (defensive)
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    # Select estimator
    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    # Train and time it
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predictions
    y_pred = model.predict(X_val)

    # For classification, try to get probabilities if possible (useful for ROC-AUC)
    y_proba_for_metrics = None
    if task == "classification" and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_val)
            # If binary, pass positive-class probability as 1d array for AUC;
            # otherwise pass the 2D array (roc_auc_score can accept that with multi_class)
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba_for_metrics = y_proba[:, 1]
            else:
                y_proba_for_metrics = y_proba
        except Exception:
            y_proba_for_metrics = None

    # Compute metrics using your shared utilities
    computed_metrics = metrics_module.compute_metrics(
        y_val, y_pred, task, y_proba=y_proba_for_metrics
    )

    # Save model
    filename = _model_filename("random_forest", task)
    model_path = os.path.join(model_dir, filename)
    joblib.dump(model, model_path)

    # Model file size
    model_size = os.path.getsize(model_path)

    # Optional extras
    extra = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "sklearn_version": None,
    }

    result = {
        "model_name": f"Random Forest ({'Classifier' if task == 'classification' else 'Regressor'})",
        "model_path": model_path,
        "metrics": computed_metrics,
        "train_time": train_time,
        "model_size": model_size,
        "extra": extra,
    }

    return result
