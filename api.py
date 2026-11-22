from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from src.predict import predict_customer

app = FastAPI(title="Telco Customer Churn API", version="1.0.0")
STATIC_DIR = Path("static")


class CustomerInput(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = Field(0, ge=0, le=1)
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = Field(12, ge=0, le=72)
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = Field(75.0, ge=0)
    TotalCharges: float | None = None


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict")
def predict(customer: CustomerInput):
    data = customer.model_dump()
    if data["TotalCharges"] is None:
        data["TotalCharges"] = round(data["MonthlyCharges"] * data["tenure"], 2)
    return predict_customer(data)