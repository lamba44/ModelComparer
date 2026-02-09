from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import numpy as np


def classification_metrics(y_true, y_pred, y_proba=None):
    """
    Computes standard classification metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # ROC-AUC (only if probabilities are provided and binary classification)
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = None

    return metrics


def regression_metrics(y_true, y_pred):
    """
    Computes standard regression metrics.
    """

    mse = mean_squared_error(y_true, y_pred)

    metrics = {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    return metrics


def compute_metrics(y_true, y_pred, task, y_proba=None):
    """
    Unified metrics interface.
    """

    if task == "classification":
        return classification_metrics(y_true, y_pred, y_proba)
    elif task == "regression":
        return regression_metrics(y_true, y_pred)
    else:
        raise ValueError(f"Unknown task type: {task}")
