"""
experiment_runner.py

- Runs selected models (single or combo) on a CSV + target.
- Prints FULL result dict for each model.
- Prints clean comparison table at end.
- Does NOT save run summaries to disk.
"""

import importlib
from datetime import datetime
from pprint import pprint

from utils.data_preprocessing import load_and_preprocess

# Map user-facing selection strings to trainer module paths.
SELECTION_TO_MODULE = {
    # classification singles
    "Random Forest Classifier": "trainers.random_forest",
    "Logistic Regression": "trainers.logistic_regression",
    "MLP": "trainers.mlp",
    "CNN": "trainers.cnn",
    "LSTM": "trainers.lstm",
    # classification combos
    "Random Forest + Logistic Regression": "trainers.combos.rf_plus_logistic",
    "Random Forest + MLP": "trainers.combos.rf_plus_mlp",
    "Random Forest + CNN": "trainers.combos.rf_plus_cnn",
    "Random Forest + LSTM": "trainers.combos.rf_plus_lstm",
    "Logistic Regression + MLP": "trainers.combos.lr_plus_mlp",
    "Logistic Regression + CNN": "trainers.combos.lr_plus_cnn",
    "Logistic Regression + LSTM": "trainers.combos.lr_plus_lstm",
    "MLP + CNN": "trainers.combos.mlp_plus_cnn",
    "MLP + LSTM": "trainers.combos.mlp_plus_lstm",
    "CNN + LSTM": "trainers.combos.cnn_plus_lstm",
    "Random Forest + MLP + CNN": "trainers.combos.rf_plus_mlp_plus_cnn",
    "Random Forest + CNN + LSTM": "trainers.combos.rf_plus_cnn_plus_lstm",
    "Logistic Regression + MLP + CNN": "trainers.combos.lr_plus_mlp_plus_cnn",
    "Random Forest + Logistic Regression + MLP": "trainers.combos.rf_plus_logistic_plus_mlp",
    "Random Forest + Logistic Regression + CNN": "trainers.combos.rf_plus_logistic_plus_cnn",
    # regression singles
    "Linear Regression": "trainers.linear_regression",
    "Random Forest Regressor": "trainers.random_forest_regressor",
    "MLP Regressor": "trainers.mlp_regressor",
    # regression combos
    "Linear Regression + RF Regressor": "trainers.combos.linear_plus_random_forest_regressor",
    "Linear Regression + MLP Regressor": "trainers.combos.linear_plus_mlp_regressor",
    "RF Regressor + MLP Regressor": "trainers.combos.rf_regressor_plus_mlp_regressor",
    "Linear Regression + RF Regressor + MLP Regressor": "trainers.combos.linear_plus_rf_plus_mlp_regressor",
}


def _human_size(num_bytes):
    if not num_bytes:
        return "-"
    n = float(num_bytes)
    if n < 1024:
        return f"{n:.0f} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n /= 1024.0
        if n < 1024.0:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PB"


def _round(val):
    if val is None:
        return "-"
    try:
        return f"{val:.4f}"
    except Exception:
        return str(val)


def _extract_metrics(result, task):
    mets = result.get("metrics", {}) or {}

    if task == "classification":
        return {
            "accuracy": _round(mets.get("accuracy")),
            "f1": _round(mets.get("f1")),
            "recall": _round(mets.get("recall")),
            "precision": _round(mets.get("precision")),
            "roc_auc": _round(mets.get("roc_auc")),
        }
    else:
        return {
            "mse": _round(mets.get("mse")),
            "rmse": _round(mets.get("rmse")),
            "mae": _round(mets.get("mae")),
            "r2": _round(mets.get("r2")),
        }


def _print_table(rows, task):
    if task == "classification":
        metric_cols = ["accuracy", "f1", "recall", "precision", "roc_auc"]
    else:
        metric_cols = ["mse", "rmse", "mae", "r2"]

    headers = ["Model"] + metric_cols + ["train_time(s)", "size"]

    col_widths = {h: len(h) for h in headers}

    for r in rows:
        col_widths["Model"] = max(col_widths["Model"], len(r["name"]))
        for m in metric_cols:
            col_widths[m] = max(col_widths[m], len(r["metrics"].get(m, "-")))
        col_widths["train_time(s)"] = max(
            col_widths["train_time(s)"], len(f"{r['train_time']:.3f}")
        )
        col_widths["size"] = max(col_widths["size"], len(_human_size(r["model_size"])))

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)

    print("\n" + header_line)
    print(sep_line)

    for r in rows:
        parts = []
        parts.append(r["name"].ljust(col_widths["Model"]))
        for m in metric_cols:
            parts.append(r["metrics"].get(m, "-").rjust(col_widths[m]))
        parts.append(f"{r['train_time']:.3f}".rjust(col_widths["train_time(s)"]))
        parts.append(_human_size(r["model_size"]).rjust(col_widths["size"]))
        print(" | ".join(parts))
    print()


def run_experiment(csv_path, target_column, task, selected_models, config=None):
    if config is None:
        config = {}

    print(f"Loading and preprocessing CSV: {csv_path}")
    X_train, X_test, y_train, y_test, _ = load_and_preprocess(
        csv_path,
        target_column,
        task,
        test_size=0.2,
        random_state=config.get("random_state", 42),
    )

    rows = []
    total_time = 0.0
    total_size = 0

    for sel in selected_models:
        print(f"\n=== Running: {sel} ===")
        module_path = SELECTION_TO_MODULE[sel]
        mod = importlib.import_module(module_path)

        trainer_cfg = dict(config)
        trainer_cfg["task"] = task
        trainer_cfg["model_dir"] = config.get("model_dir", "models")

        res = mod.train(X_train, y_train, X_test, y_test, trainer_cfg)

        result = res[0] if isinstance(res, tuple) else res

        print("\nFULL RESULT DICT:")
        pprint(result)

        disp_metrics = _extract_metrics(result, task)

        row = {
            "name": sel,
            "metrics": disp_metrics,
            "train_time": float(result.get("train_time", 0.0) or 0.0),
            "model_size": int(result.get("model_size", 0) or 0),
        }

        rows.append(row)
        total_time += row["train_time"]
        total_size += row["model_size"]

    _print_table(rows, task)

    print(f"Total train time: {total_time:.3f}s")
    print(f"Total model size: {_human_size(total_size)}")
    print(f"Run finished: {datetime.now().isoformat()}")

    return rows


if __name__ == "__main__":
    CSV = "samplefiles/placement_data.csv"
    TARGET = "salary_lpa"
    TASK = "regression"
    SELECTED = [
        "Linear Regression",
        "Random Forest Regressor",
        "Linear Regression + RF Regressor + MLP Regressor",
    ]

    run_experiment(CSV, TARGET, TASK, SELECTED, config={"model_dir": "models"})
