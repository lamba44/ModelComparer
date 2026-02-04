import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_prepare_data(
    csv_path: str,
    target_column: str,
    task_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    # 1. Load CSV
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV")

    # 2. Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 3. One-hot encode categorical columns
    X_encoded = pd.get_dummies(X)

    # 4. Train-test split (ALWAYS 80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task_type == "classification" else None,
    )

    # 5. Return everything needed
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": list(X_encoded.columns),
    }
