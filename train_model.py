import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib

# ✅ Load the dataset FIRST
data = pd.read_csv("large_wearable_clinical_dataset.csv")

# ✅ Encode target labels
label_encoder = LabelEncoder()
data['ConditionLabel'] = label_encoder.fit_transform(data['ConditionLabel'])

# ✅ Save the label encoder for later use in app.py
joblib.dump(label_encoder, 'model/label_encoder.pkl')

# ✅ Features and Target
numeric_features = ['HeartRate', 'SpO2', 'SleepHours', 'BP_Systolic', 'BP_Diastolic', 'Steps']
text_features = ['ClinicalNotes', 'Medication']
X = data[numeric_features + text_features]
y = data['ConditionLabel']

# ✅ Preprocessing
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=300, stop_words='english'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('text1', text_transformer, 'ClinicalNotes'),
    ('text2', text_transformer, 'Medication')
])

# ✅ Model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
])

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Fit model
pipeline.fit(X_train, y_train)

# ✅ Evaluation
y_pred = pipeline.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save model
joblib.dump(pipeline, "model/model.pkl")
print("💾 Model saved as model/model.pkl")
