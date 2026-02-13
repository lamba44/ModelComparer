# utils/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_and_preprocess(
    csv_path: str,
    target_column: str,
    task: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([("scaler", StandardScaler())])

    # KEEP sparse output to avoid exploding memory usage
    categorical_pipeline = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # allow ColumnTransformer to return sparse if any transformer is sparse
    )

    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    return X_train_trans, X_test_trans, y_train.values, y_test.values, preprocessor
