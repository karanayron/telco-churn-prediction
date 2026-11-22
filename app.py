import streamlit as st
from src.predict import predict_customer

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Predictor")
st.caption("Enter customer details below to estimate churn risk.")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 75.0, step=1.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )

with col2:
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Has Partner", ["No", "Yes"])

if st.button("Predict Churn Risk", type="primary", use_container_width=True):
    customer = {
        "gender": "Female",
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": "No",
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": round(monthly_charges * tenure, 2),
    }

    result = predict_customer(customer)
    prob_pct = result["churn_probability"] * 100

    st.divider()
    if result["risk_tier"] == "High":
        st.error(f"**High Churn Risk** — {prob_pct:.1f}% probability of leaving.")
    elif result["risk_tier"] == "Medium":
        st.warning(f"**Medium Churn Risk** — {prob_pct:.1f}% probability of leaving.")
    else:
        st.success(f"**Low Churn Risk** — {prob_pct:.1f}% probability of leaving.")