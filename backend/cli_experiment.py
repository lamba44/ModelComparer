import os
import sys
import pandas as pd
import importlib
from textwrap import dedent

CLASSIFICATION_OPTIONS = [
    # Single models
    "Random Forest Classifier",
    "Logistic Regression",
    "MLP",
    "CNN",
    "LSTM",
    # 2-model ensembles
    "Random Forest + Logistic Regression",
    "Random Forest + MLP",
    "Random Forest + CNN",
    "Random Forest + LSTM",
    "Logistic Regression + MLP",
    "Logistic Regression + CNN",
    "Logistic Regression + LSTM",
    "MLP + CNN",
    "MLP + LSTM",
    "CNN + LSTM",
    # 3-model ensembles
    "Random Forest + MLP + CNN",
    "Random Forest + CNN + LSTM",
    "Logistic Regression + MLP + CNN",
    "Random Forest + Logistic Regression + MLP",
    "Random Forest + Logistic Regression + CNN",
]

REGRESSION_OPTIONS = [
    # Single models
    "Linear Regression",
    "Random Forest Regressor",
    "MLP Regressor",
    # 2-model ensembles
    "Linear Regression + Random Forest Regressor",
    "Linear Regression + MLP Regressor",
    "Random Forest Regressor + MLP Regressor",
    # 3-model ensemble
    "Linear Regression + Random Forest Regressor + MLP Regressor",
]


try:
    import experiment_runner as er

    run_experiment = er.run_experiment
    SELECTION_TO_MODULE = er.SELECTION_TO_MODULE
except Exception:
    sys.path.append(os.getcwd())
    import experiment_runner as er

    run_experiment = er.run_experiment
    SELECTION_TO_MODULE = er.SELECTION_TO_MODULE


def find_samplefiles_dir():
    candidates = [
        os.path.join(os.getcwd(), "samplefiles"),
        os.path.join(os.getcwd(), "../samplefiles"),
        os.path.join(os.getcwd(), "samplefiles/"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Could not find 'samplefiles' directory. Make sure it exists."
    )


def list_csv_files(sample_dir):
    files = [f for f in os.listdir(sample_dir) if f.lower().endswith(".csv")]
    files.sort()
    return files


def print_list(items):
    for i, it in enumerate(items, start=1):
        print(f"  {i}. {it}")


def choose_one(prompt, items):
    while True:
        choice = input(prompt + " (enter number): ").strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(items):
                return items[idx - 1]
        except Exception:
            pass
        print("Invalid choice — try again.")


def choose_multiple(prompt, items, max_choices=3):
    print(
        f"(You may select 1 to {max_choices} options — comma-separated numbers, e.g. 1 or 1,3)"
    )
    while True:
        raw = input(prompt + ": ").strip()
        try:
            parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
            indices = []
            for p in parts:
                i = int(p)
                if not (1 <= i <= len(items)):
                    raise ValueError
                indices.append(i)
            indices = list(dict.fromkeys(indices))  # preserve order but remove dupes
            if len(indices) < 1 or len(indices) > max_choices:
                raise ValueError
            return [items[i - 1] for i in indices]
        except Exception:
            print(f"Invalid input. Pick between 1 and {max_choices} valid numbers.")


def infer_task_from_series(s: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(s):
        if s.nunique() > 15:
            return "regression"
    return "classification"


def build_options_for_task(task: str):
    if task == "classification":
        base_list = CLASSIFICATION_OPTIONS
    else:
        base_list = REGRESSION_OPTIONS

    available = []
    missing = []
    for opt in base_list:
        if opt in SELECTION_TO_MODULE:
            available.append(opt)
        else:
            missing.append(opt)

    if missing:
        print(
            "\nNote: the following canonical options are not registered in experiment_runner.SELECTION_TO_MODULE and will be hidden:"
        )
        for m in missing:
            print(f"  - {m}")
        print(
            "If you want them available, add mappings to SELECTION_TO_MODULE in experiment_runner.py.\n"
        )

    return available


def main():
    print(
        dedent(
            """
    =========================
      ModelComparer Project
    =========================
    """
        )
    )

    sample_dir = find_samplefiles_dir()
    print(f"Found samplefiles directory at: {sample_dir}\n")

    csv_files = list_csv_files(sample_dir)
    if not csv_files:
        print("No CSV files found in samplefiles. Place your CSVs there and re-run.")
        return

    print("Available CSV files:")
    print_list(csv_files)
    csv_choice = choose_one("Choose a CSV file", csv_files)
    csv_path = os.path.join(sample_dir, csv_choice)
    print(f"\nSelected CSV: {csv_path}\n")

    try:
        df_head = pd.read_csv(csv_path, nrows=50)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    cols = list(df_head.columns)
    print("Columns found in CSV:")
    print_list(cols)
    target = choose_one("\nSelect the column you want to predict", cols)
    print(f"Target column selected: {target}\n")

    auto = input("Auto-detect task type from column dtype? (Y/n): ").strip().lower()
    if auto in ("", "y", "yes"):
        try:
            ser = pd.read_csv(csv_path, usecols=[target], squeeze=True)
        except Exception:
            ser = df_head[target]
        task = infer_task_from_series(ser)
        print(f"Auto-detected task: {task}\n")
    else:
        print("Select Task:\n 1. -> Classification\n 2. -> Regression")
        task = choose_one("Choose task", ["classification", "regression"])
        print()

    options = build_options_for_task(task)
    if not options:
        print("No model options available for this task (check SELECTION_TO_MODULE).")
        return

    print("Available model / combo options:")
    print_list(options)
    selected = choose_multiple("Select model(s) by number", options, max_choices=3)
    print("\nYour final selection:")
    print(f" CSV: {csv_choice}")
    print(f" Target: {target}")
    print(f" Task: {task}")
    print(" Models:")
    for m in selected:
        print(f"  - {m}")

    confirm = (
        input("\nProceed and run experiment with these selections? (Y/n): ")
        .strip()
        .lower()
    )
    if confirm not in ("", "y", "yes"):
        print("Cancelled.")
        return

    print(
        "\nStarting experiment — this will call your existing experiment_runner.run_experiment(...) and stream its output.\n"
    )

    run_experiment(csv_path, target, task, selected, config={"model_dir": "models"})


if __name__ == "__main__":
    main()
