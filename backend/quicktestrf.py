from utils.data_preprocessing import load_and_preprocess
from trainers.random_forest_regressor import train as rf_reg_train
from trainers.mlp_regressor import train as mlp_reg_train

CSV = "samplefiles/StudentPerformance.csv"
TARGET = "Performance Index"

X_train, X_test, y_train, y_test, _ = load_and_preprocess(
    CSV, TARGET, task="regression"
)

cfg = {
    "task": "regression",
    "model_dir": "models",
}  # wrappers will set task automatically
print("RF Regressor ->", rf_reg_train(X_train, y_train, X_test, y_test, cfg))
print(
    "MLP Regressor ->",
    mlp_reg_train(
        X_train,
        y_train,
        X_test,
        y_test,
        {"max_iter": 200, "base_name": "mlp_regressor", "model_dir": "models"},
    ),
)
