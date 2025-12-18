import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("./../sample.csv")

# Target column
TARGET = "attrition"

# Split features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Convert categorical columns
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\nTest data predictions:")
for i in range(len(y_test)):
    print("Actual:", y_test.iloc[i], "Predicted:", predictions[i])


print("Model accuracy:", accuracy)
