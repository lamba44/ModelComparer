from utils.data_preprocessing import load_and_preprocess
from trainers.combos.rf_plus_logistic import train as combo_train

CSV = "samplefiles/credit_fraud.csv"
TARGET = "is_fraud"  # update appropriately

X_train, X_test, y_train, y_test, preproc = load_and_preprocess(
    CSV, TARGET, task="classification"
)

cfg = {
    "task": "classification",
    "model_dir": "models",
    "return_model": True,
    "random_state": 42,
}
res = combo_train(X_train, y_train, X_test, y_test, cfg)
print("COMBO RESULT:", res)
