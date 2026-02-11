from trainers import mlp as mlp_module


def train(X_train, y_train, X_val, y_val, config=None):
    if config is None:
        config = {}
    config = dict(config)
    config["task"] = "regression"
    # set a clear base_name for file naming
    if "base_name" not in config:
        config["base_name"] = "mlp_regressor"
    return mlp_module.train(X_train, y_train, X_val, y_val, config)
