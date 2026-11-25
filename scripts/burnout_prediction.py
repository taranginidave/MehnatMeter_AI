
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("data/students.csv")

features = [
    "study_hours",
    "sleep_hours",
    "breaks_per_day",
    "stress_level",
    "fatigue_level",
    "consecutive_study_days"
]

X = data[features]
y = data["burnout_risk"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# SMOTE + Random Forest Pipeline
model = Pipeline([
    ("smote", SMOTE(
        random_state=42,
        k_neighbors=2
    )),

    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print("\n===== MEHNATMETER AI =====")

print("\nAccuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=["Low", "Medium", "High"],
    zero_division=0
))

print("\nConfusion Matrix:")

print(confusion_matrix(
    y_test,
    predictions,
    labels=["Low", "Medium", "High"]
))

# Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/burnout_model.pkl")

print("\nModel saved successfully!")
