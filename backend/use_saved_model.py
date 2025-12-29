import joblib
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)


def load_model(model_path):
    data = joblib.load(model_path)
    model = data.get("model")
    columns = data.get("columns")
    return model, columns


def prepare_features(df, trained_columns):
    X = pd.get_dummies(df)
    # add missing cols
    for col in trained_columns:
        if col not in X.columns:
            X[col] = 0
    # drop extra cols
    extra = [c for c in X.columns if c not in trained_columns]
    if extra:
        X = X.drop(columns=extra)
    # reorder
    X = X[trained_columns]
    return X


def score_model(model_path, csv_path, target_col=None, id_col=None):
    print(f"Loading model from: {model_path}")
    model, trained_columns = load_model(model_path)
    df = pd.read_csv(csv_path)
    if id_col and id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(df))]

    if target_col and target_col in df.columns:
        y_true = df[target_col]
        X_df = df.drop(
            columns=[target_col] + ([id_col] if id_col and id_col in df.columns else [])
        )
    else:
        y_true = None
        X_df = df.drop(columns=[id_col] if id_col and id_col in df.columns else [])

    X = prepare_features(X_df, trained_columns)

    preds = model.predict(X)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)

    # print predictions
    print("\nPredictions (first 10 rows):")
    for i in range(min(10, len(preds))):
        out = f"id={ids[i]} pred={int(preds[i])}"
        if probs is not None:
            out += f" probs={probs[i].tolist()}"
        print(out)

    # if ground truth available, show metrics
    if y_true is not None:
        acc = accuracy_score(y_true, preds)
        print(f"\nAccuracy on provided CSV: {acc:.6f}")
        print("\nClassification report:")
        print(classification_report(y_true, preds, zero_division=0))
        labels = np.unique(y_true)
        p, r, f1, support = precision_recall_fscore_support(
            y_true, preds, labels=labels, zero_division=0
        )
        print("\nPer-class breakdown:")
        for idx, lab in enumerate(labels):
            print(
                f" Class {lab}: precision={p[idx]:.4f} recall={r[idx]:.4f} f1={f1[idx]:.4f} support={int(support[idx])}"
            )

    return preds, probs


if __name__ == "__main__":
    # EXAMPLE USAGE:
    # score_model("../models/Random_Forest.joblib", "../new_transactions_to_score.csv", target_col="is_fraud", id_col="transaction_id")
    # Modify the paths below as needed and run `python use_saved_model.py`
    model_path = "../models/Random_Forest.joblib"
    csv_path = "../withoutFraudLabel.csv"  # or your new file
    target_col = "is_fraud"  # set to None if you don't have ground truth
    id_col = None  # set if your CSV has an id column you want printed
    score_model(model_path, csv_path, target_col=target_col, id_col=id_col)
