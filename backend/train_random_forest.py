# backend/train_random_forest.py
import time
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from data_pipeline import load_and_prepare_data

# ---------------- CONFIG ----------------
CSV_PATH = "../samplefiles/credit_card_fraud_10k.csv"  # change if needed
TARGET = "is_fraud"
TASK = "classification"  # 'classification' or 'regression'
MODEL_OUT_PATH = "../models/Random_Forest.joblib"
# ----------------------------------------


def train_and_evaluate():
    print("Preparing data...")
    d = load_and_prepare_data(csv_path=CSV_PATH, target_column=TARGET, task_type=TASK)
    X_train, X_test = d["X_train"], d["X_test"]
    y_train, y_test = d["y_train"], d["y_test"]
    feature_columns = d["feature_columns"]

    print(f"Shapes -> X_train: {X_train.shape}, X_test: {X_test.shape}")

    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    print("\nTraining Random Forest...")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"Training completed in {train_time:.4f} seconds")

    print("\nPredicting on test set...")
    t1 = time.perf_counter()
    preds = model.predict(X_test)
    predict_time = time.perf_counter() - t1
    print(f"Prediction completed in {predict_time:.4f} seconds")

    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.6f}\n")

    print("Classification report:")
    print(classification_report(y_test, preds, zero_division=0))

    labels = np.unique(y_test)
    p, r, f1, supp = precision_recall_fscore_support(
        y_test, preds, labels=labels, zero_division=0
    )
    print("\nPer-class breakdown:")
    for i, lab in enumerate(labels):
        print(
            f" Class {lab}: precision={p[i]:.4f}, recall={r[i]:.4f}, f1={f1[i]:.4f}, support={int(supp[i])}"
        )

    # Save model (with feature columns so we can align later)
    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    joblib.dump({"model": model, "columns": feature_columns}, MODEL_OUT_PATH)
    model_size = os.path.getsize(MODEL_OUT_PATH)
    print(f"\nSaved model to: {MODEL_OUT_PATH} (size: {model_size} bytes)")

    # Show sample predictions (first 5)
    print("\nSample predictions (first 5 test rows):")
    for i in range(min(5, len(y_test))):
        print(f" Actual: {int(y_test.iloc[i])} | Predicted: {int(preds[i])}")

    # Summary
    summary = {
        "model": "RandomForestClassifier",
        "accuracy": float(acc),
        "train_time_s": round(train_time, 4),
        "predict_time_s": round(predict_time, 4),
        "model_file": os.path.abspath(MODEL_OUT_PATH),
        "model_size_bytes": model_size,
    }
    print("\nSUMMARY:")
    for k, v in summary.items():
        print(f" {k}: {v}")


if __name__ == "__main__":
    train_and_evaluate()
