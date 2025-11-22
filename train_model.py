# train_model.py

import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# from preprocessing_code import large_and_clinical_dataset
from preprocessing_code import load_and_clean_dataset


# -----------------------
# Load dataset
# -----------------------

df = load_and_clean_dataset("large_wearable_clinical_dataset.csv")

# Features
numeric_features = [
    "HeartRate", "SpO2", "SleepHours", "BP_Systolic", "BP_Diastolic", "Steps"
]
text_features = ["ClinicalNotes", "Medication"]

# Label
label_col = "ConditionLabel"

# Encode label
label_encoder = LabelEncoder()
df[label_col] = label_encoder.fit_transform(df[label_col])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df[numeric_features + text_features],
    df[label_col],
    test_size=0.2,
    random_state=42,
    stratify=df[label_col]
)

# -----------------------
# ColumnTransformer
# -----------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("clin_text", TfidfVectorizer(stop_words="english", max_features=500), "ClinicalNotes"),
        ("med_text", TfidfVectorizer(stop_words="english", max_features=300), "Medication"),
    ],
    remainder="drop"
)

# -----------------------
# Model pipeline
# -----------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss"
    ))
])

# -----------------------
# Train
# -----------------------

print("\nTraining model...")
model.fit(X_train, y_train)

# -----------------------
# Evaluate
# -----------------------

preds = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, preds))

# -----------------------
# Save model
# -----------------------

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

print("\nModel saved to model/model.pkl")
print("Label encoder saved to model/label_encoder.pkl")
