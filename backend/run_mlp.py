print("Starting MLP training script...")

from trainers import train_mlp_classifier

res = train_mlp_classifier(
    csv_path="../samplefiles/credit_card_fraud_10k.csv",
    target_column="is_fraud",
    model_out_path="../models/MLP_Classifier.joblib",
)

print("\nMLP training finished.")
print("MLP accuracy:", res["metrics"]["accuracy"])
