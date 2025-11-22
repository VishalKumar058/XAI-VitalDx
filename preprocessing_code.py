# preprocessing_code.py
import pandas as pd
import numpy as np

def load_and_clean_dataset(csv_path):
    """
    Loads and cleans the FULL dataset used for model training.
    """
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Expected columns
    required = [
        "HeartRate", "SpO2", "SleepHours", 
        "BP_Systolic", "BP_Diastolic", "Steps",
        "ClinicalNotes", "Medication",
        "ConditionLabel"
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Clean text columns
    df["ClinicalNotes"] = df["ClinicalNotes"].fillna("").astype(str)
    df["Medication"] = df["Medication"].fillna("").astype(str)

    # Clean numeric columns
    numeric_cols = ["HeartRate", "SpO2", "SleepHours", "BP_Systolic", "BP_Diastolic", "Steps"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    return df
