import os
import time
from datetime import datetime
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression

from utils import metrics as metrics_module


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _model_filename(task: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"logistic_regression_{task}_{ts}.joblib"


def train(X_train, y_train, X_val, y_val, config=None):
    """
    Train Logistic Regression (classification only).

    Args:
        X_train, X_val: preprocessed feature arrays
        y_train, y_val: target arrays
        config: dict with optional keys:
            - task: must be "classification"
            - max_iter: int (default 1000)
            - random_state: int
            - model_dir: directory to save models

    Returns:
        standardized result dict
    """

    if config is None:
        config = {}

    task = config.get("task")
    if task != "classification":
        raise ValueError("Logistic Regression supports classification only")

    max_iter = config.get("max_iter", 1000)
    random_state = config.get("random_state", 42)
    model_dir = config.get("model_dir", "models")

    _ensure_dir(model_dir)

    # Defensive conversion
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )

    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predict
    y_pred = model.predict(X_val)

    # Probabilities (for ROC AUC)
    y_proba_for_metrics = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_val)
            if y_proba.shape[1] == 2:
                y_proba_for_metrics = y_proba[:, 1]
            else:
                y_proba_for_metrics = y_proba
        except Exception:
            y_proba_for_metrics = None

    # Metrics
    computed_metrics = metrics_module.compute_metrics(
        y_val,
        y_pred,
        task="classification",
        y_proba=y_proba_for_metrics,
    )

    # Save model
    filename = _model_filename(task)
    model_path = os.path.join(model_dir, filename)
    joblib.dump(model, model_path)

    model_size = os.path.getsize(model_path)

    result = {
        "model_name": "Logistic Regression",
        "model_path": model_path,
        "metrics": computed_metrics,
        "train_time": train_time,
        "model_size": model_size,
        "extra": {
            "max_iter": max_iter,
        },
    }

    return result
