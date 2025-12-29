import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

df = pd.read_csv("../credit_card_fraud_10k.csv")  # update filename if needed
TARGET = "is_fraud"
X = df.drop(columns=[TARGET])
y = df[TARGET]
X = pd.get_dummies(X)

total_rows = len(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
test_rows = len(y_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(),
}

os.makedirs("../models", exist_ok=True)
results = []
print(f"Total rows in dataset: {total_rows}")
print(f"Total rows in test set: {test_rows}")

for name, model in models.items():
    print(f"\n===== {name} =====")
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    start_pred = time.perf_counter()
    predictions = model.predict(X_test)
    predict_time = time.perf_counter() - start_pred

    accuracy = float(accuracy_score(y_test, predictions))

    labels = np.unique(y_test)
    p, r, f1, support = precision_recall_fscore_support(
        y_test, predictions, labels=labels, zero_division=0
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )

    safe_name = name.replace(" ", "_")
    model_fname = f"../models/{safe_name}.joblib"

    joblib.dump({"model": model, "columns": X.columns.tolist()}, model_fname)
    model_size = os.path.getsize(model_fname)

    print("Sample predictions (first 5 of test set):")
    for i in range(min(5, len(y_test))):
        print(f"Actual: {y_test.iloc[i]} | Predicted: {predictions[i]}")

    print(f"Accuracy: {accuracy:.6f}")
    print(
        f"Train time (s): {train_time:.4f} | Predict time (s): {predict_time:.4f} | Model size (bytes): {model_size}"
    )
    print("\nPer-class metrics:")
    for idx, lab in enumerate(labels):
        print(
            f" Class {lab}: precision={p[idx]:.4f} recall={r[idx]:.4f} f1={f1[idx]:.4f} support={int(support[idx])}"
        )
    print(
        f"\nMacro avg: precision={macro_p:.4f} recall={macro_r:.4f} f1={macro_f1:.4f}"
    )
    print(
        f"Weighted avg: precision={weighted_p:.4f} recall={weighted_r:.4f} f1={weighted_f1:.4f}"
    )

    result = {
        "model": name,
        "accuracy": accuracy,
        "train_time_s": round(train_time, 4),
        "predict_time_s": round(predict_time, 4),
        "model_file": os.path.abspath(model_fname),
        "model_size_bytes": model_size,
        "labels": labels.tolist(),
        "per_class": {},
        "macro": {
            "precision": round(float(macro_p), 4),
            "recall": round(float(macro_r), 4),
            "f1": round(float(macro_f1), 4),
        },
        "weighted": {
            "precision": round(float(weighted_p), 4),
            "recall": round(float(weighted_r), 4),
            "f1": round(float(weighted_f1), 4),
        },
    }
    for idx, lab in enumerate(labels):
        result["per_class"][str(int(lab))] = {
            "precision": round(float(p[idx]), 4),
            "recall": round(float(r[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    results.append(result)

summary_rows = []
for res in results:
    summary_rows.append(
        {
            "model": res["model"],
            "accuracy": res["accuracy"],
            "train_time_s": res["train_time_s"],
            "predict_time_s": res["predict_time_s"],
            "model_size_bytes": res["model_size_bytes"],
            "macro_f1": res["macro"]["f1"],
            "weighted_f1": res["weighted"]["f1"],
            "fraud_precision": res["per_class"].get("1", {}).get("precision", None),
            "fraud_recall": res["per_class"].get("1", {}).get("recall", None),
            "fraud_f1": res["per_class"].get("1", {}).get("f1", None),
            "fraud_support": res["per_class"].get("1", {}).get("support", None),
        }
    )

df_summary = pd.DataFrame(summary_rows)
print("\n===== SUMMARY TABLE =====")
print(df_summary.to_string(index=False))

with open("../models/results_summary.json", "w") as f:
    json.dump(
        {"total_rows": total_rows, "test_rows": test_rows, "results": results},
        f,
        indent=2,
    )

print("\nSaved detailed metrics to ../models/results_summary.json")
