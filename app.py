# # app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import joblib
# import os
# import numpy as np

# from utils.preprocessing import build_input_dataframe
# from utils.explain import explain_prediction_with_shap
# from utils.narrative import generate_narrative


# app = Flask(__name__)
# app.secret_key = "mysecretkey123"

# CORS(app)  # allow frontend (localhost) to call this API

# from auth.routes import auth_bp
# app.register_blueprint(auth_bp)


# # Paths (adjust if needed)
# MODEL_PATH = os.path.join("model", "model.pkl")
# LABEL_ENCODER_PATH = os.path.join("model", "label_encoder.pkl")

# # Load model
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model not found at {MODEL_PATH} - run train_model.py first")
# model = joblib.load(MODEL_PATH)

# label_encoder = None
# if os.path.exists(LABEL_ENCODER_PATH):
#     label_encoder = joblib.load(LABEL_ENCODER_PATH)


# def _to_json_serializable(obj):
#     """Convert numpy types to native python types for JSON."""
#     if isinstance(obj, np.generic):
#         return obj.item()
#     if isinstance(obj, list):
#         return [_to_json_serializable(x) for x in obj]
#     if isinstance(obj, dict):
#         return {k: _to_json_serializable(v) for k, v in obj.items()}
#     return obj


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         wearable_file = request.files.get('wearable')
#         clinical_notes = request.form.get('clinical_notes', "")
#         medications = request.form.get('medications', "")

#         if wearable_file is None:
#             return jsonify({'error': 'No wearable file provided'}), 400

#         # read wearable csv
#         wearable_df = pd.read_csv(wearable_file)
#         if wearable_df.shape[0] == 0:
#             return jsonify({'error': 'wearable csv is empty'}), 400

#         # Build input row for model
#         input_df = build_input_dataframe(wearable_df, clinical_notes, medications)

#         # Predict: try predict_proba, else predict
#         pred_proba = None
#         pred_idx = None
#         try:
#             pred_proba = model.predict_proba(input_df)
#             pred_idx = int(np.argmax(pred_proba, axis=1)[0])
#         except Exception:
#             pred_idx = int(model.predict(input_df)[0])

#         pred_name = label_encoder.inverse_transform([pred_idx])[0] if label_encoder is not None else str(pred_idx)

#         # Explanation (SHAP) - returns list of dicts {'feature':..., 'shap_value': ...}
#         explanation = explain_prediction_with_shap(model, input_df, top_k=10)

#         # Ensure everything JSON-serializable (convert numpy types)
#         safe_explanation = []
#         for item in explanation:
#             # If explanation returned an error dict, pass it through
#             if isinstance(item, dict) and 'error' in item:
#                 safe_explanation.append(item)
#                 continue
#             feat = item.get('feature')
#             val = item.get('shap_value')
#             safe_explanation.append({'feature': str(feat), 'shap_value': float(np.array(val).astype(float).reshape(-1)[0]) if val is not None else 0.0})

#         narrative = generate_narrative(pred_name, safe_explanation, clinical_notes, medications)

#         # result = {
#         #     "prediction": str(pred_name),
#         #     "explanation": safe_explanation,
#         #     "probabilities": (_to_json_serializable(pred_proba.tolist()[0]) if pred_proba is not None else None)
#         # }

#         result = {
#             "prediction": str(pred_name),
#             "explanation": safe_explanation,
#             "narrative": narrative,
#             "probabilities": (_to_json_serializable(pred_proba.tolist()[0]) if pred_proba is not None else None)
#         }

#         # debug print (helps when frontend seems hung)
#         print("\n================ BACKEND RAW RESULT ================")
#         print(result)
#         print("===================================================\n")

#         return jsonify(result)

#     except Exception as e:
#         # return safe error message
#         print("ERROR in /predict:", str(e))
#         return jsonify({"error": str(e)}), 500


# if __name__ == '__main__':
#     # bind to "localhost" so frontend and backend origin match on mac (avoid IPv4/IPv6 mismatch)
#     # app.run(host="localhost", port=5000, debug=True).   mainly used for windows
#     app.run(host="127.0.0.1", port=5000, debug=True)




# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import os
import numpy as np

from utils.preprocessing import build_input_dataframe
from utils.explain import explain_prediction_with_shap
from utils.narrative import generate_narrative

app = Flask(__name__)
app.secret_key = "mysecretkey123"   # required for sessions

CORS(app)

# Register authentication blueprint
from auth.routes import auth_bp
app.register_blueprint(auth_bp)

# -------------------------
# HOME / PREDICTION PAGE
# -------------------------
@app.route('/')
def home():
    # Load your existing UI here
    return render_template("frontend_with_medical_data.html")

# -------------------------
# MODEL SETUP
# -------------------------
MODEL_PATH = os.path.join("model", "model.pkl")
LABEL_ENCODER_PATH = os.path.join("model", "label_encoder.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")

model = joblib.load(MODEL_PATH)

label_encoder = None
if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

def _to_json_serializable(obj):
    """Convert numpy types for JSON."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list):
        return [_to_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    return obj

# -------------------------
# PREDICT API
# -------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        wearable_file = request.files.get('wearable')
        clinical_notes = request.form.get('clinical_notes', "")
        medications = request.form.get('medications', "")

        if wearable_file is None:
            return jsonify({'error': 'No wearable file provided'}), 400

        wearable_df = pd.read_csv(wearable_file)
        if wearable_df.shape[0] == 0:
            return jsonify({'error': 'wearable csv is empty'}), 400

        input_df = build_input_dataframe(wearable_df, clinical_notes, medications)

        pred_proba = None
        try:
            pred_proba = model.predict_proba(input_df)
            pred_idx = int(np.argmax(pred_proba, axis=1)[0])
        except:
            pred_idx = int(model.predict(input_df)[0])

        pred_name = label_encoder.inverse_transform([pred_idx])[0] if label_encoder else str(pred_idx)

        explanation = explain_prediction_with_shap(model, input_df, top_k=10)

        safe_explanation = []
        for item in explanation:
            if isinstance(item, dict) and 'error' in item:
                safe_explanation.append(item)
                continue
            feat = item.get('feature')
            val = item.get('shap_value')
            safe_explanation.append({
                'feature': str(feat),
                'shap_value': float(np.array(val).astype(float).reshape(-1)[0]) if val is not None else 0.0
            })

        narrative = generate_narrative(pred_name, safe_explanation, clinical_notes, medications)

        result = {
            "prediction": str(pred_name),
            "explanation": safe_explanation,
            "narrative": narrative,
            "probabilities": (_to_json_serializable(pred_proba.tolist()[0]) if pred_proba is not None else None)
        }

        print("\n===== BACKEND RESULT =====")
        print(result)
        print("===========================\n")

        return jsonify(result)

    except Exception as e:
        print("ERROR in /predict:", str(e))
        return jsonify({"error": str(e)}), 500

# -------------------------
# RUN SERVER
# -------------------------
if __name__ == '__main__':
    # app.run(host="127.0.0.1", port=5000, debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
