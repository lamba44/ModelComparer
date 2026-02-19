import numpy as np
import time
import json
import os


def soft_average_classification(models, X_test):
    """
    models: list of trained sklearn/keras models
    returns: predicted labels
    """
    probas = []

    for model in models:
        if hasattr(model, "predict_proba"):
            probas.append(model.predict_proba(X_test))
        else:
            # Keras models
            preds = model.predict(X_test)
            if preds.ndim == 1:
                preds = np.vstack([1 - preds, preds]).T
            probas.append(preds)

    avg_proba = np.mean(probas, axis=0)
    return np.argmax(avg_proba, axis=1), avg_proba


def save_combo_metadata(save_path, components, method):
    data = {
        "components": components,
        "method": method,
    }
    with open(save_path, "w") as f:
        json.dump(data, f)
