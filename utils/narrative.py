# utils/narrative.py

def generate_narrative(prediction, shap_features, clinical_notes, medications):
    """
    Converts SHAP features + symptoms + medication + prediction into a readable clinical explanation.
    """

    # 1. Base explanation templates per disease
    CONDITION_INTROS = {
        "arrhythmia": "The model suggests a possibility of arrhythmia, which often presents with irregular heart activity.",
        "hypoxia": "The model indicates potential hypoxia, typically associated with reduced oxygen saturation.",
        "hypertension": "The model points toward hypertension, based on several blood-pressure-related indicators.",
        "tachycardia": "The model suggests tachycardia, characterized by elevated heart rate levels.",
        "CHF": "The model indicates signs consistent with congestive heart failure (CHF), often linked to fatigue, breathlessness, and fluid imbalance."
    }

    intro = CONDITION_INTROS.get(prediction, "The model detected physiological patterns that align with this condition.")

    # 2. Rule-based explanations for features
    FEATURE_EXPLANATIONS = {
        "HeartRate": "Elevated heart rate levels contribute to cardiovascular stress.",
        "SpO2": "Reduced oxygen saturation (SpO2) suggests possible respiratory or oxygenation issues.",
        "SleepHours": "Lower sleep duration may indicate fatigue and reduced recovery.",
        "BP_Systolic": "High systolic blood pressure is a common marker of hypertension.",
        "BP_Diastolic": "Elevated diastolic pressure also contributes to blood pressure abnormalities.",
        "Steps": "Lower physical activity levels may point toward fatigue or reduced endurance.",
        "fatigue": "Reported fatigue is a common symptom in cardiovascular disorders.",
        "headache": "Headache can be associated with blood-pressure fluctuations.",
        "breathlessness": "Breathlessness often indicates respiratory strain or cardiac overload.",
        "chest": "Chest-related symptoms may signify cardiovascular stress.",
        "dizziness": "Dizziness may indicate reduced perfusion or abnormal vital patterns.",
        "metoprolol": "Metoprolol is typically prescribed to control heart rhythm and blood pressure.",
        "lisinopril": "Lisinopril usage suggests management of hypertension or cardiac load.",
        "atorvastatin": "Atorvastatin is often prescribed for cholesterol management, linked to heart health.",
    }

    # Convert SHAP features to readable sentences
    sentences = []
    for item in shap_features:
        feat_raw = item["feature"]

        # Normalize SHAP feature names
        feat = (
            feat_raw.replace("num__", "")
                    .replace("clin_text__", "")
                    .replace("med_text__", "")
                    .lower()
        )

        # Match rule-based explanations
        for key, explanation in FEATURE_EXPLANATIONS.items():
            if key in feat:
                sentences.append(explanation)

    # Add clinical notes interpretation
    if clinical_notes.strip():
        sentences.append(f"The clinical notes mention: '{clinical_notes}', which may correlate with the predicted condition.")

    # Add medication interpretation
    if medications.strip():
        sentences.append(f"The medications listed ('{medications}') also provide context for the model's decision.")

    # Final narrative summary
    return " ".join([intro] + sentences)
