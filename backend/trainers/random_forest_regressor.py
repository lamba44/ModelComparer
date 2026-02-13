from trainers import random_forest as rf_module


def train(X_train, y_train, X_val, y_val, config=None):
    if config is None:
        config = {}
    config = dict(config)
    config["task"] = "regression"
    return rf_module.train(X_train, y_train, X_val, y_val, config)
