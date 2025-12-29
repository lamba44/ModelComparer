import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor

    xgb_available = True
except Exception:
    xgb_available = False

df = pd.read_csv("../StudentPerformance.csv")  # update filename if needed
TARGET = "Performance Index"  # change to your numeric target column name
X = df.drop(columns=[TARGET])
y = df[TARGET]
X = pd.get_dummies(X)

total_rows = len(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
test_rows = len(y_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
}
if xgb_available:
    models["XGBoost Regressor"] = XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse"
    )

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

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    safe_name = name.replace(" ", "_")
    model_fname = f"../models/{safe_name}.joblib"

    joblib.dump({"model": model, "columns": X.columns.tolist()}, model_fname)
    model_size = os.path.getsize(model_fname)

    print("Sample predictions (first 5 of test set):")
    for i in range(min(5, len(y_test))):
        print(f"Actual: {float(y_test.iloc[i])} | Predicted: {float(predictions[i])}")

    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    print(
        f"Train time (s): {train_time:.4f} | Predict time (s): {predict_time:.4f} | Model size (bytes): {model_size}"
    )

    result = {
        "model": name,
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "r2": round(r2, 6),
        "train_time_s": round(train_time, 4),
        "predict_time_s": round(predict_time, 4),
        "model_file": os.path.abspath(model_fname),
        "model_size_bytes": model_size,
        "labels": None,
    }

    results.append(result)

summary_rows = []
for res in results:
    summary_rows.append(
        {
            "model": res["model"],
            "rmse": res["rmse"],
            "mae": res["mae"],
            "r2": res["r2"],
            "train_time_s": res["train_time_s"],
            "predict_time_s": res["predict_time_s"],
            "model_size_bytes": res["model_size_bytes"],
        }
    )

df_summary = pd.DataFrame(summary_rows)
print("\n===== SUMMARY TABLE =====")
print(df_summary.to_string(index=False))

with open("../models/results_summary_regression.json", "w") as f:
    json.dump(
        {"total_rows": total_rows, "test_rows": test_rows, "results": results},
        f,
        indent=2,
    )

print("\nSaved detailed metrics to ../models/results_summary_regression.json")
if not xgb_available:
    print(
        "\nNote: xgboost is not installed in your environment. XGBoost Regressor was skipped. Install it with 'pip install xgboost' if you want to include it next time."
    )
