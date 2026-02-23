import os
from datetime import datetime
import numpy as np

from trainers import mlp as mlp_module
from trainers import lstm as lstm_module

from utils import ensemble_utils
from utils import metrics as metrics_module


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _ensemble_filename():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ensemble_mlp_plus_lstm_{ts}.json"


def train(X_train, y_train, X_val, y_val, config=None):

    if config is None:
        config = {}

    task = config.get("task")
    if task != "classification":
        raise ValueError("mlp_plus_lstm combo supports classification only")

    model_dir = config.get("model_dir", "models")
    _ensure_dir(model_dir)

    cfg = dict(config)
    cfg["return_model"] = True

    mlp_out = mlp_module.train(X_train, y_train, X_val, y_val, cfg)
    if isinstance(mlp_out, tuple):
        mlp_result, mlp_model, mlp_meta = mlp_out
    else:
        raise RuntimeError(
            "MLP trainer must return model object when return_model=True"
        )

    lstm_out = lstm_module.train(X_train, y_train, X_val, y_val, cfg)
    if isinstance(lstm_out, tuple):
        lstm_result, lstm_model, lstm_meta = lstm_out
    else:
        raise RuntimeError(
            "LSTM trainer must return model object when return_model=True"
        )

    trained_components = [
        ("mlp", mlp_result, mlp_model, mlp_meta),
        ("lstm", lstm_result, lstm_model, lstm_meta),
    ]

    canonical_classes = np.unique(y_train).tolist()

    aligned_probas = []
    component_model_paths = []
    sum_model_size = 0
    sum_train_time = 0.0

    for name, result, model, meta in trained_components:
        proba, model_classes = ensemble_utils.predict_proba_or_onehot(
            model, X_val, task="classification", meta=meta
        )
        aligned = ensemble_utils.align_probas_to_canonical(
            proba, model_classes, canonical_classes
        )
        aligned_probas.append(aligned)

        component_model_paths.append(result.get("model_path"))
        sum_model_size += int(result.get("model_size", 0))
        sum_train_time += float(result.get("train_time", 0.0))

    combined_proba = ensemble_utils.combine_probas(aligned_probas)
    combined_pred_idx = np.argmax(combined_proba, axis=1)
    combined_preds = [canonical_classes[i] for i in combined_pred_idx]

    metrics = metrics_module.compute_metrics(
        y_val, combined_preds, task="classification", y_proba=combined_proba
    )

    combine_time = 0.0  # negligible here; could measure if desired

    ensemble_meta = {
        "ensemble_name": "MLP + LSTM",
        "method": "soft_average",
        "components": component_model_paths,
        "metrics": metrics,
        "train_time_sum": sum_train_time,
        "model_size_sum": sum_model_size,
        "combine_time": combine_time,
        "canonical_classes": canonical_classes,
    }

    ensemble_fname = _ensemble_filename()
    ensemble_path = os.path.join(model_dir, ensemble_fname)
    ensemble_utils.save_ensemble_metadata(ensemble_path, ensemble_meta)

    result = {
        "model_name": "MLP + LSTM",
        "model_path": ensemble_path,
        "metrics": metrics,
        "train_time": sum_train_time,
        "model_size": sum_model_size,
        "extra": {
            "components": component_model_paths,
            "method": "soft_average",
        },
    }

    return result
