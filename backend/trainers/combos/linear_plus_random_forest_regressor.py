import os
from datetime import datetime
import numpy as np

from trainers import linear_regression as lin_module
from trainers import random_forest_regressor as rf_reg_module

from utils import ensemble_utils
from utils import metrics as metrics_module


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _ensemble_filename():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ensemble_linear_plus_random_forest_regressor_{ts}.json"


def _safe_predict(model, X):

    preds = model.predict(X)
    arr = np.asarray(preds)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr.ravel()
    return arr


def train(X_train, y_train, X_val, y_val, config=None):
    if config is None:
        config = {}

    task = config.get("task")
    if task != "regression":
        raise ValueError("linear_plus_random_forest_regressor supports regression only")

    model_dir = config.get("model_dir", "models")
    _ensure_dir(model_dir)

    cfg = dict(config)
    cfg["return_model"] = True

    lin_out = lin_module.train(X_train, y_train, X_val, y_val, cfg)
    if isinstance(lin_out, tuple):
        lin_result, lin_model, lin_meta = lin_out
    else:
        raise RuntimeError(
            "Linear regression trainer must return model object when return_model=True"
        )

    rf_out = rf_reg_module.train(X_train, y_train, X_val, y_val, cfg)
    if isinstance(rf_out, tuple):
        rf_result, rf_model, rf_meta = rf_out
    else:
        raise RuntimeError(
            "Random forest regressor trainer must return model object when return_model=True"
        )

    trained_components = [
        ("linear_regression", lin_result, lin_model, lin_meta),
        ("random_forest_regressor", rf_result, rf_model, rf_meta),
    ]

    preds_list = []
    component_model_paths = []
    sum_model_size = 0
    sum_train_time = 0.0

    for name, result, model, meta in trained_components:
        pred = _safe_predict(model, X_val)
        preds_list.append(pred)

        component_model_paths.append(result.get("model_path"))
        sum_model_size += int(result.get("model_size", 0))
        sum_train_time += float(result.get("train_time", 0.0))

    preds_stack = np.vstack(preds_list)  # shape: (n_models, n_samples)
    combined_pred = np.mean(preds_stack, axis=0)

    metrics = metrics_module.compute_metrics(y_val, combined_pred, task="regression")

    ensemble_meta = {
        "ensemble_name": "Linear Regression + Random Forest Regressor",
        "method": "mean_average",
        "components": component_model_paths,
        "metrics": metrics,
        "train_time_sum": sum_train_time,
        "model_size_sum": sum_model_size,
        "combine_time": 0.0,
    }

    ensemble_fname = _ensemble_filename()
    ensemble_path = os.path.join(model_dir, ensemble_fname)
    ensemble_utils.save_ensemble_metadata(ensemble_path, ensemble_meta)

    result = {
        "model_name": "Linear Regression + Random Forest Regressor",
        "model_path": ensemble_path,
        "metrics": metrics,
        "train_time": sum_train_time,
        "model_size": sum_model_size,
        "predictions": combined_pred.tolist(),
        "extra": {"components": component_model_paths, "method": "mean_average"},
    }

    return result
