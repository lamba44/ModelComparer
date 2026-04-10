import os
import time
from datetime import datetime
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression

from utils import metrics as metrics_module


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _model_filename(task: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"linear_regression_{task}_{ts}.joblib"


def train(X_train, y_train, X_val, y_val, config=None):

    if config is None:
        config = {}

    task = config.get("task")
    if task != "regression":
        raise ValueError("Linear Regression supports regression only")

    return_model = bool(config.get("return_model", False))

    fit_intercept = config.get("fit_intercept", True)
    n_jobs = config.get("n_jobs", None)
    model_dir = config.get("model_dir", "models")

    _ensure_dir(model_dir)

    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    model = LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs)

    # Train
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Predict
    y_pred = model.predict(X_val)

    # Metrics
    computed_metrics = metrics_module.compute_metrics(y_val, y_pred, task="regression")

    # Save model
    filename = _model_filename(task)
    model_path = os.path.join(model_dir, filename)
    joblib.dump(model, model_path)

    model_size = os.path.getsize(model_path)

    result = {
        "model_name": "Linear Regression",
        "model_path": model_path,
        "metrics": computed_metrics,
        "train_time": train_time,
        "model_size": model_size,
        "predictions": y_pred.tolist(),
        "extra": {
            "fit_intercept": fit_intercept,
            "n_jobs": n_jobs,
        },
    }

    meta = {}  # no classes for regression

    if return_model:
        return result, model, meta

    return result
