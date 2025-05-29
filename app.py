from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Custom utility functions
from utils.preprocessing import preprocess_input
from utils.explain import explain_prediction

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from the frontend
        wearable_file = request.files['wearable']
        clinical_notes = request.form['clinical_notes']
        medications = request.form['medications']

        print("✅ Received clinical notes:", clinical_notes)
        print("✅ Received medications:", medications)
        print("✅ File object:", wearable_file)

        # Read CSV content into a DataFrame
        wearable_df = pd.read_csv(wearable_file)
        print("✅ Wearable Data:", wearable_df.head())

        # Preprocess and fuse multimodal input
        features = preprocess_input(wearable_df, clinical_notes, medications)

        # Predict using model
        prediction = model.predict([features])[0]

        # Explain prediction using SHAP
        explanation = explain_prediction(model, features)

        # Return response
        return jsonify({
            'prediction': prediction,
            'explanation': explanation
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
