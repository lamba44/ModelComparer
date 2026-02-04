# backend/trainers.py
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from data_pipeline import load_and_prepare_data


def _compute_classification_metrics(y_true, y_pred):
    labels = np.unique(y_true)
    accuracy = float(accuracy_score(y_true, y_pred))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class = {}
    for idx, lab in enumerate(labels):
        # ensure string keys
        try:
            key = str(int(lab))
        except Exception:
            key = str(lab)
        per_class[key] = {
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    metrics = {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "weighted": {
            "precision": float(weighted_p),
            "recall": float(weighted_r),
            "f1": float(weighted_f1),
        },
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }
    return metrics


def train_random_forest_classifier(
    csv_path,
    target_column,
    save_model=True,
    model_out_path="../models/Random_Forest.joblib",
    test_size=0.2,
    random_state=42,
    n_jobs=-1,
):
    """
    Train a RandomForestClassifier using the shared data pipeline.
    Returns a result dict with model, metrics, timings, saved path, etc.
    """

    # 1) load & prepare data (uses 80/20 split inside)
    data = load_and_prepare_data(
        csv_path=csv_path,
        target_column=target_column,
        task_type="classification",
        test_size=test_size,
        random_state=random_state,
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_columns = data["feature_columns"]

    # 2) create model
    model = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)

    # 3) train (measure time)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # 4) predict (measure time)
    t1 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time = time.perf_counter() - t1

    # 5) metrics
    metrics = _compute_classification_metrics(y_test, preds)

    # 6) save model (with metadata) if requested
    model_file_abs = None
    model_size = None
    if save_model:
        os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
        payload = {"model": model, "columns": feature_columns}
        joblib.dump(payload, model_out_path)
        model_file_abs = os.path.abspath(model_out_path)
        model_size = os.path.getsize(model_out_path)

    # 7) prepare result dict
    result = {
        "model": model,
        "model_file": model_file_abs,
        "model_size_bytes": model_size,
        "train_time_s": round(train_time, 6),
        "predict_time_s": round(predict_time, 6),
        "metrics": metrics,
        "feature_columns": feature_columns,
        "X_test_shape": X_test.shape,
        "y_test_shape": y_test.shape,
        # include a tiny sample to inspect if desired (first 5 rows)
        "sample_test_rows": {
            "X_test_head": X_test.head().reset_index(drop=True),
            "y_test_head": y_test.reset_index(drop=True).head(),
            "preds_head": pd.Series(preds[:5]).reset_index(drop=True),
        },
    }

    # 8) print summary to console for convenience
    print(
        f"\nTrained RandomForestClassifier in {train_time:.4f}s, prediction in {predict_time:.4f}s"
    )
    print(
        f"Model saved to: {model_file_abs} (size {model_size} bytes)"
        if save_model
        else "Model not saved"
    )
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print("Per-class metrics:")
    for cls, vals in metrics["per_class"].items():
        print(
            f" Class {cls}: precision={vals['precision']:.4f} recall={vals['recall']:.4f} f1={vals['f1']:.4f} support={vals['support']}"
        )
    print("\nClassification report:\n", metrics["classification_report"])

    return result


def train_mlp_classifier(
    csv_path,
    target_column,
    save_model=True,
    model_out_path="../models/MLP_Classifier.joblib",
    test_size=0.2,
    random_state=42,
    hidden_layer_sizes=(100,),
    max_iter=400,
    learning_rate_init=0.001,
):
    """
    Train an MLPClassifier (neural network) using the shared data pipeline.

    Mirrors the RandomForest trainer's return structure. Saves both the
    trained model and the StandardScaler used to scale features.
    """

    # 1) load & prepare data (uses 80/20 split inside)
    data = load_and_prepare_data(
        csv_path=csv_path,
        target_column=target_column,
        task_type="classification",
        test_size=test_size,
        random_state=random_state,
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_columns = data["feature_columns"]

    # 2) scale features (important for MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    # 3) create model
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        learning_rate_init=learning_rate_init,
        random_state=random_state,
    )

    # 4) train (measure time)
    t0 = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - t0

    # 5) predict (measure time)
    t1 = time.perf_counter()
    preds = model.predict(X_test_scaled)
    predict_time = time.perf_counter() - t1

    # 6) metrics
    metrics = _compute_classification_metrics(y_test, preds)

    # 7) save model + scaler if requested
    model_file_abs = None
    model_size = None
    if save_model:
        os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
        payload = {"model": model, "columns": feature_columns, "scaler": scaler}
        joblib.dump(payload, model_out_path)
        model_file_abs = os.path.abspath(model_out_path)
        model_size = os.path.getsize(model_out_path)

    # 8) prepare result dict (same shape as RF trainer)
    result = {
        "model": model,
        "model_file": model_file_abs,
        "model_size_bytes": model_size,
        "train_time_s": round(train_time, 6),
        "predict_time_s": round(predict_time, 6),
        "metrics": metrics,
        "feature_columns": feature_columns,
        "X_test_shape": X_test.shape,
        "y_test_shape": y_test.shape,
        "sample_test_rows": {
            "X_test_head": X_test.head().reset_index(drop=True),
            "y_test_head": y_test.reset_index(drop=True).head(),
            "preds_head": pd.Series(preds[:5]).reset_index(drop=True),
        },
    }

    # 9) print summary
    print(
        f"\nTrained MLPClassifier in {train_time:.4f}s, prediction in {predict_time:.4f}s"
    )
    print(
        f"Model saved to: {model_file_abs} (size {model_size} bytes)"
        if save_model
        else "Model not saved"
    )
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print("Per-class metrics:")
    for cls, vals in metrics["per_class"].items():
        print(
            f" Class {cls}: precision={vals['precision']:.4f} recall={vals['recall']:.4f} f1={vals['f1']:.4f} support={vals['support']}"
        )
    print("\nClassification report:\n", metrics["classification_report"])

    return result
