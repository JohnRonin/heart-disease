# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("heart.csv")  # Ensure heart.csv is in the same folder

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Optional: Feature scaling (good for some models, not critical for RandomForest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save model and scaler
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("💾 Model and scaler saved successfully!")
