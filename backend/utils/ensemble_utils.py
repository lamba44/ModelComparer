import os
import time
import json
import numpy as np

try:
    import tensorflow as tf

    _HAS_TF = True
    _KERAS_MODEL_TYPE = tf.keras.Model
except Exception:
    _HAS_TF = False
    _KERAS_MODEL_TYPE = None


def is_keras_model(obj):
    if not _HAS_TF:
        return False
    return isinstance(obj, _KERAS_MODEL_TYPE)


def _reshape_for_sequence_if_keras(model, X):

    if is_keras_model(model):
        X = np.asarray(X, dtype=np.float32)
        return X.reshape((X.shape[0], X.shape[1], 1))
    return X


def get_model_classes(model, meta=None):

    if meta and "classes" in meta and meta["classes"] is not None:
        return list(meta["classes"])

    if hasattr(model, "classes_"):
        return list(getattr(model, "classes_"))
    return None


def predict_proba_or_onehot(model, X, task, meta=None):
    X_in = _reshape_for_sequence_if_keras(model, X)

    if task == "regression":
        preds = model.predict(X_in)
        preds = np.asarray(preds).reshape(-1)
        return preds, None

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_in)
            proba = np.asarray(proba)
            model_classes = get_model_classes(model, meta)
            return proba, model_classes
        except Exception:
            pass

    if is_keras_model(model):
        proba = model.predict(X_in)
        proba = np.asarray(proba)
        model_classes = get_model_classes(model, meta)
        return proba, model_classes

    pred = model.predict(X_in)
    pred = np.asarray(pred).reshape(-1)
    model_classes = get_model_classes(model, meta)
    if model_classes is None:
        model_classes = np.unique(pred).tolist()

    proba = np.zeros((len(pred), len(model_classes)), dtype=float)
    class_to_idx = {c: i for i, c in enumerate(model_classes)}
    for i, v in enumerate(pred):
        j = class_to_idx.get(v, None)
        if j is None:
            continue
        proba[i, j] = 1.0
    return proba, list(model_classes)


def align_probas_to_canonical(proba, model_classes, canonical_classes):

    n_samples = proba.shape[0]

    if proba.ndim == 2 and proba.shape[1] == 1 and len(canonical_classes) == 2:
        p = proba[:, 0]
        proba = np.vstack([1.0 - p, p]).T
        model_classes = [None, None]

    canonical = list(canonical_classes)
    aligned = np.zeros((n_samples, len(canonical)), dtype=float)

    if model_classes is None:
        if proba.shape[1] == len(canonical):
            aligned = proba.copy()
            return aligned
        else:
            avg = proba.mean(axis=1)
            for j in range(len(canonical)):
                aligned[:, j] = avg
            return aligned

    # normal case: map columns
    class_to_idx_model = {c: i for i, c in enumerate(model_classes)}
    for j, c in enumerate(canonical):
        if c in class_to_idx_model:
            aligned[:, j] = proba[:, class_to_idx_model[c]]
        else:
            # model has no probability for this class -> leave zeros
            aligned[:, j] = 0.0
    return aligned


def combine_probas(aligned_probas_list, weights=None):
    if not aligned_probas_list:
        raise ValueError("no probas to combine")
    stack = np.stack(aligned_probas_list, axis=0)  # (n_models, n_samples, n_classes)
    if weights is None:
        return np.mean(stack, axis=0)
    weights = np.asarray(weights).reshape(-1, 1, 1)
    weighted = np.sum(stack * weights, axis=0) / weights.sum()
    return weighted


def combine_regressions(preds_list, weights=None):
    if not preds_list:
        raise ValueError("no preds to combine")
    stack = np.stack([np.asarray(p).reshape(-1) for p in preds_list], axis=0)
    if weights is None:
        return np.mean(stack, axis=0)
    weights = np.asarray(weights).reshape(-1, 1)
    weighted = np.sum(stack * weights, axis=0) / weights.sum()
    return weighted


def save_ensemble_metadata(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            default=lambda o: (
                o if isinstance(o, (int, float, str, bool, list, dict)) else str(o)
            ),
            indent=2,
        )
