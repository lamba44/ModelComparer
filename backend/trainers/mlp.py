# trainers/mlp.py
import os
import time
from datetime import datetime
import joblib
import numpy as np

from sklearn.neural_network import MLPClassifier, MLPRegressor

from utils import metrics as metrics_module


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _model_filename(base_name: str, task: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{task}_{ts}.joblib"


def train(X_train, y_train, X_val, y_val, config=None):
    """
    Train an MLPClassifier or MLPRegressor depending on config['task'].
    Preserves the existing result dict format used elsewhere in the project.
    If config['return_model'] is True, returns (result, model, meta).
    """

    if config is None:
        config = {}

    task = config.get("task")
    if task not in ("classification", "regression"):
        raise ValueError("config['task'] must be 'classification' or 'regression'")

    return_model = bool(config.get("return_model", False))

    hidden_layer_sizes = config.get("hidden_layer_sizes", (100,))
    max_iter = config.get("max_iter", 200)
    random_state = config.get("random_state", 42)
    learning_rate_init = config.get("learning_rate_init", 0.001)
    model_dir = config.get("model_dir", "models")
    base_name = config.get("base_name", "mlp")

    _ensure_dir(model_dir)

    # Ensure numpy arrays
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    if task == "classification":
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
        )
    else:
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            learning_rate_init=learning_rate_init,
        )

    # Train & time
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict
    y_pred = model.predict(X_val)

    # Probabilities for classification (if available)
    y_proba_for_metrics = None
    if task == "classification" and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_val)
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba_for_metrics = y_proba[:, 1]
            else:
                y_proba_for_metrics = y_proba
        except Exception:
            y_proba_for_metrics = None

    # Compute metrics
    computed_metrics = metrics_module.compute_metrics(
        y_val, y_pred, task, y_proba=y_proba_for_metrics
    )

    # Save model
    filename = _model_filename(base_name, task)
    model_path = os.path.join(model_dir, filename)
    joblib.dump(model, model_path)
    model_size = os.path.getsize(model_path)

    result = {
        "model_name": f"MLP ({'Classifier' if task == 'classification' else 'Regressor'})",
        "model_path": model_path,
        "metrics": computed_metrics,
        "train_time": train_time,
        "model_size": model_size,
        "predictions": y_pred.tolist(),
        "extra": {
            "hidden_layer_sizes": hidden_layer_sizes,
            "max_iter": max_iter,
            "learning_rate_init": learning_rate_init,
        },
    }

    meta = {}
    if task == "classification":
        meta["classes"] = getattr(model, "classes_", None)

    if return_model:
        return result, model, meta

    return result
