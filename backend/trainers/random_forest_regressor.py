from trainers import random_forest as rf_module


def train(X_train, y_train, X_val, y_val, config=None):
    if config is None:
        config = {}
    # enforce regression task
    config = dict(config)  # shallow copy to avoid side-effects
    config["task"] = "regression"
    # optional: choose a directory or leave as-is
    return rf_module.train(X_train, y_train, X_val, y_val, config)
