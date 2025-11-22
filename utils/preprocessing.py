import pandas as pd
import numpy as np

COLUMN_MAP = {
    "heart_rate": "HeartRate",
    "spo2": "SpO2",
    "Sleephours": "SleepHours",
    "bp_systolic": "BP_Systolic",
    "bp_diastolic": "BP_Diastolic",
    "steps": "Steps"
}

NUMERIC_FEATURES = [
    "HeartRate", "SpO2", "SleepHours",
    "BP_Systolic", "BP_Diastolic", "Steps"
]

def build_input_dataframe(wearable_df, clinical_notes, medications):

    df = wearable_df.copy()

    df.rename(columns=COLUMN_MAP, inplace=True)

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    row = {col: float(df.iloc[0][col]) for col in NUMERIC_FEATURES}
    row["ClinicalNotes"] = str(clinical_notes)
    row["Medication"] = str(medications)

    return pd.DataFrame([row])
