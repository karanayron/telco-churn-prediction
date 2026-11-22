import numpy as np
import pandas as pd


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Loads the Telco Customer Churn CSV and applies deterministic cleaning:
    - Drops non-predictive identifier column (customerID)
    - Converts TotalCharges from string to float; fills tenure=0 blanks with 0.0
    - Encodes target variable Churn ('Yes' -> 1, 'No' -> 0)
    """
    df = pd.read_csv(filepath)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0.0)

    if "Churn" in df.columns and not pd.api.types.is_numeric_dtype(df["Churn"]):
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates domain-specific features using purely row-wise operations
    (safe from data leakage because no cross-row statistics are computed).
    """
    df = df.copy()

    # 1. Tenure lifecycle buckets
    bins = [-1, 12, 24, 48, 72]
    labels = ["0-12m", "13-24m", "25-48m", "49-72m"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)

    # 2. Total number of active services
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)

    # 3. Historical average monthly spend
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # 4. Current-to-historical charge ratio
    df["charge_ratio"] = df["MonthlyCharges"] / (df["avg_monthly_spend"] + 1e-5)

    return df