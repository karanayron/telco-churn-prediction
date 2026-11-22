from pathlib import Path
import joblib
import pandas as pd
from src.data_prep import add_features

MODEL_PATH = Path("models/churn_pipeline.joblib")


def predict_customer(customer_data: dict, model_path: Path = MODEL_PATH) -> dict:
    """
    Loads the saved scikit-learn Pipeline, applies the exact same feature
    engineering used in training, and returns churn prediction + probability.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run 'uv run python -m src.train' first."
        )

    pipeline = joblib.load(model_path)

    # Convert single customer dict to a 1-row DataFrame and engineer features
    df = pd.DataFrame([customer_data])
    df = add_features(df)

    churn_prob = float(pipeline.predict_proba(df)[0, 1])
    churn_pred = int(pipeline.predict(df)[0])

    return {
        "churn_prediction": "Yes" if churn_pred == 1 else "No",
        "churn_probability": round(churn_prob, 4),
        "risk_tier": "High" if churn_prob >= 0.7 else ("Medium" if churn_prob >= 0.4 else "Low"),
    }


if __name__ == "__main__":
    # Example: High-risk profile (new, month-to-month, Fiber Optic, Electronic check)
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.50,
        "TotalCharges": 286.50,
    }

    result = predict_customer(sample_customer)
    print("Prediction Result:", result)