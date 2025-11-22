# utils/explain.py
import shap
import numpy as np
import pandas as pd

def flatten_value(v):
    """Convert any SHAP array into a clean scalar."""
    try:
        v = np.array(v).astype(float)

        if v.ndim == 0:
            return float(v)

        if v.size == 1:
            return float(v.reshape(-1)[0])

        return float(v.mean())
    except:
        return 0.0


def explain_prediction_with_shap(pipeline_model, input_df: pd.DataFrame, top_k: int = 10):

    preprocessor = pipeline_model.named_steps.get("preprocessor", None)
    classifier = pipeline_model.named_steps.get("classifier", pipeline_model)

    # Transform
    if preprocessor:
        X_transformed = preprocessor.transform(input_df)
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"f_{i}" for i in range(X_transformed.shape[1])]
    else:
        X_transformed = input_df.values
        feature_names = list(input_df.columns)

    # SHAP
    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)

        # Multiclass
        if isinstance(shap_values, list):
            pred_class = classifier.predict(X_transformed)[0]
            values = shap_values[pred_class][0]
        else:
            values = shap_values[0]

    except Exception as e:
        return [{"error": f"SHAP failed: {str(e)}"}]

    flat_values = [flatten_value(v) for v in values]

    feature_contribs = [
        {"feature": feature_names[i], "shap_value": flat_values[i]}
        for i in range(len(flat_values))
    ]

    feature_contribs.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return feature_contribs[:top_k]
