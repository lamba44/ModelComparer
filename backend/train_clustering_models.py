import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "../clustering_sample.csv"  # update to your file
# A simple KMeans default; you can tune n_clusters later
KMEANS_K = 5
# DBSCAN defaults (eps, min_samples) — you may tune these
DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5

# -----------------------------
# LOAD & PREPARE DATA
# -----------------------------
df = pd.read_csv("../Mall_Customers.csv")
total_rows = len(df)

# Drop any obvious ID column if present (common name pattern), change if needed
id_cols = [c for c in df.columns if "id" in c.lower()]
if id_cols:
    df_features = df.drop(columns=id_cols)
else:
    df_features = df.copy()

# Clustering works on features only (no labels)
X = pd.get_dummies(df_features)
feature_columns = X.columns.tolist()

# -----------------------------
# MODELS
# -----------------------------
models = {
    "KMeans": KMeans(n_clusters=KMEANS_K, random_state=42),
    "DBSCAN": DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES),
}

os.makedirs("../models", exist_ok=True)
results = []

print(f"Total rows in dataset: {total_rows}")
print(f"Feature columns count: {len(feature_columns)}")

for name, model in models.items():
    print(f"\n===== {name} =====")
    start_train = time.perf_counter()
    # For clustering many estimators support fit_predict; use that to both fit and get labels
    try:
        labels = model.fit_predict(X)
        train_time = time.perf_counter() - start_train
        # For models that separate fit & predict (not common here), we could call predict
        predict_time = 0.0
    except Exception as e:
        # fallback: fit then predict if available
        start_fit = time.perf_counter()
        model.fit(X)
        train_time = time.perf_counter() - start_fit
        if hasattr(model, "predict"):
            start_pred = time.perf_counter()
            labels = model.predict(X)
            predict_time = time.perf_counter() - start_pred
        else:
            # if no predict, fall back to labels_ after fit
            labels = getattr(model, "labels_", None)
            predict_time = 0.0

    labels = np.array(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(
        unique_labels[unique_labels != -1]
    )  # exclude noise (-1) for DBSCAN
    has_noise = -1 in unique_labels

    # silhouette_score requires at least 2 clusters (excluding noise) and fewer clusters than samples
    sil_score = None
    try:
        # compute silhouette on samples that are not noise if DBSCAN produced noise
        if has_noise:
            valid_mask = labels != -1
            if valid_mask.sum() >= 2 and len(np.unique(labels[valid_mask])) >= 2:
                sil_score = float(silhouette_score(X[valid_mask], labels[valid_mask]))
        else:
            if len(np.unique(labels)) >= 2:
                sil_score = float(silhouette_score(X, labels))
    except Exception:
        sil_score = None

    inertia = None
    if hasattr(model, "inertia_"):
        try:
            inertia = float(model.inertia_)
        except Exception:
            inertia = None

    # cluster sizes
    cluster_sizes = {}
    for lab in unique_labels:
        cluster_sizes[str(int(lab))] = int((labels == lab).sum())

    # save model
    safe_name = name.replace(" ", "_")
    model_fname = f"../models/{safe_name}.joblib"
    joblib.dump({"model": model, "columns": feature_columns}, model_fname)
    model_size = os.path.getsize(model_fname)

    # print samples
    print("Sample cluster assignments (first 5 rows):")
    for i in range(min(5, len(labels))):
        print(f"Row {i}: cluster_label={int(labels[i])}")

    print(
        f"Train time (s): {train_time:.4f} | Predict time (s): {predict_time:.4f} | Model size (bytes): {model_size}"
    )
    print(f"Number of unique labels (including noise if any): {len(unique_labels)}")
    print(f"Number of clusters (excluding noise label -1): {n_clusters}")
    print(f"Has noise cluster (-1): {has_noise}")
    if inertia is not None:
        print(f"Inertia (KMeans): {inertia:.4f}")
    print(f"Silhouette score: {sil_score}")
    print("Cluster sizes:")
    for lab, size in cluster_sizes.items():
        print(f" Cluster {lab}: {size} rows")

    result = {
        "model": name,
        "train_time_s": round(train_time, 4),
        "predict_time_s": round(predict_time, 4),
        "model_file": os.path.abspath(model_fname),
        "model_size_bytes": model_size,
        "n_clusters_including_noise": int(len(unique_labels)),
        "n_clusters_excluding_noise": int(n_clusters),
        "has_noise": bool(has_noise),
        "inertia": inertia,
        "silhouette_score": sil_score,
        "cluster_sizes": cluster_sizes,
    }
    results.append(result)

# summary table
df_summary = pd.DataFrame(results)
print("\n===== SUMMARY TABLE =====")
print(df_summary.to_string(index=False))

with open("../models/results_summary_clustering.json", "w") as f:
    json.dump({"total_rows": total_rows, "results": results}, f, indent=2)

print("\nSaved detailed metrics to ../models/results_summary_clustering.json")
