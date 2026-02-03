import pandas as pd

CSV_PATH = "../samplefiles/StudentPerformance.csv"

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


def print_list(items):
    for i, item in enumerate(items, start=1):
        print(f"{i}. {item}")


def choose_one(prompt, items):
    while True:
        choice = input(prompt + " (enter number): ").strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(items):
                return items[idx - 1]
        except ValueError:
            pass
        print("Invalid choice, try again.")


def choose_multiple(prompt, items, max_choices=3):
    print(f"\nSelect up to {max_choices} options (comma-separated, e.g. 1,3)")
    while True:
        raw = input(prompt + ": ").strip()
        try:
            indices = list(set(int(x.strip()) for x in raw.split(",")))
            if not indices or len(indices) > max_choices:
                raise ValueError
            selected = []
            for idx in indices:
                if 1 <= idx <= len(items):
                    selected.append(items[idx - 1])
                else:
                    raise ValueError
            return selected
        except ValueError:
            print(f"Invalid input. Choose 1 to {max_choices} valid numbers.")


def infer_task(series):
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() > 15:
            return "regression"
    return "classification"


def main():
    print("\n=== BASIC MODEL SELECTION FLOW ===\n")
    print(f"Loading CSV from: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    columns = list(df.columns)

    print("\nColumns found:")
    print_list(columns)

    target = choose_one("\nSelect the column you want to predict", columns)
    print(f"\nTarget column selected: {target}")

    auto = input("\nAuto-detect task type? (Y/n): ").strip().lower()
    if auto in ("", "y", "yes"):
        task = infer_task(df[target])
        print(f"Auto-detected task: {task}")
    else:
        print("\nTask types:")
        task = choose_one("Select task type", ["classification", "regression"])

    print("\nAvailable model / combination options:")
    if task == "classification":
        options = CLASSIFICATION_OPTIONS
    else:
        options = REGRESSION_OPTIONS

    print_list(options)

    selected_models = choose_multiple("Select model(s)", options, max_choices=3)

    print("\n==============================")
    print("FINAL SELECTION")
    print("==============================")
    print(f"CSV File      : {CSV_PATH}")
    print(f"Task Type     : {task}")
    print(f"Target Column : {target}")
    print("Selected Model(s):")
    for m in selected_models:
        print(f" - {m}")
    print("==============================\n")


if __name__ == "__main__":
    main()
