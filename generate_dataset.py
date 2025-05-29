import pandas as pd
import random
import numpy as np

conditions = ['CHF', 'arrhythmia', 'healthy', 'hypertension', 'hypoxia']

clinical_notes = {
    'CHF': ["shortness of breath", "fatigue", "swelling in legs", "irregular heartbeat"],
    'arrhythmia': ["palpitations", "dizziness", "fainting", "irregular heart rhythm"],
    'healthy': ["feeling normal", "no major symptoms", "active lifestyle", "good sleep"],
    'hypertension': ["high blood pressure", "headache", "blurred vision", "chest pain"],
    'hypoxia': ["low oxygen levels", "confusion", "cyanosis", "difficulty breathing"]
}

medications = {
    'CHF': ["furosemide", "lisinopril"],
    'arrhythmia': ["amiodarone", "beta blockers"],
    'healthy': ["none"],
    'hypertension': ["amlodipine", "losartan"],
    'hypoxia': ["oxygen therapy", "bronchodilators"]
}

def generate_entry(condition):
    return {
        'HeartRate': random.randint(60, 110),
        'SpO2': round(random.uniform(85, 100), 1),
        'SleepHours': round(random.uniform(4, 9), 1),
        'BP_Systolic': random.randint(90, 160),
        'BP_Diastolic': random.randint(60, 100),
        'Steps': random.randint(1000, 15000),
        'ClinicalNotes': random.choice(clinical_notes[condition]),
        'Medication': random.choice(medications[condition]),
        'ConditionLabel': condition
    }

data = [generate_entry(random.choice(conditions)) for _ in range(5000)]
df = pd.DataFrame(data)
df.to_csv('synthetic_health_data.csv', index=False)
print("✅ Dataset saved as synthetic_health_data.csv")
