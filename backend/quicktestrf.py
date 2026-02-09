from utils.data_preprocessing import load_and_preprocess
from trainers.logistic_regression import train

CSV = "samplefiles/credit_fraud.csv"
TARGET = "is_fraud"  # update if needed

X_train, X_test, y_train, y_test, _ = load_and_preprocess(
    CSV, TARGET, task="classification"
)

cfg = {
    "task": "classification",
    "max_iter": 1000,
    "random_state": 42,
    "model_dir": "models",
}

result = train(X_train, y_train, X_test, y_test, cfg)
print("RESULT:", result)
