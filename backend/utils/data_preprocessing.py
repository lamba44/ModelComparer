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
    # Load CSV
    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV")

    # Split features / target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Pipelines
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])

    categorical_pipeline = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    # 80–20 split (stratified for classification)
    stratify = y if task == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    # Fit only on train
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train.values, y_test.values, preprocessor
