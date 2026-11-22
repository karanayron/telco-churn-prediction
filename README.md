# Telco Customer Churn Prediction

An end-to-end, leakage-free Machine Learning system that predicts telecom customer churn and serves real-time risk-tier scoring (`High`, `Medium`, `Low`) through a **FastAPI** REST API, a responsive Web UI, and a **Streamlit** dashboard.

## Business Problem
Acquiring a new telecom customer costs significantly more than retaining an existing one. Using ~7,000 historical customer records (~26.5% churn rate), this system identifies high-risk customers before they cancel so retention teams can target them with tailored offers rather than blanket discounts.

## Engineering Highlights
* **Zero Data Leakage:** All preprocessing (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) is encapsulated inside a scikit-learn `ColumnTransformer` + `Pipeline` fit strictly on training folds.
* **Domain Feature Engineering:** Engineered `tenure_group`, `num_services`, `avg_monthly_spend`, and `charge_ratio` to capture early-churn cliffs, ecosystem lock-in, and recent bill increases.
* **Imbalanced Learning:** Evaluated default models against `class_weight="balanced"` and `SMOTE` (strictly inside `imblearn.pipeline.Pipeline` during 5-fold Stratified CV).
* **Trade-off Driven Deployment:** Selected **Tuned Logistic Regression** over tree ensembles for matching F1/ROC-AUC (~0.84) with single-digit millisecond latency and exact coefficient interpretability.
* **Production Serving:** Serialized the full pipeline with `joblib` and exposed it via **FastAPI** (with Pydantic validation + Swagger docs) and **Streamlit**.

## Model Comparison (Held-Out Test Set)

| Model | F1 Score | Recall | ROC-AUC | Predict Latency | Interpretability |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LogReg (Tuned)** *(Deployed)* | **~0.63** | **~0.79** | **~0.84** | **~3 ms** | **High** (signed weights) |
| HistGB (Tuned) *(Challenger)* | ~0.63 | ~0.77 | ~0.84 | ~15 ms | Medium (requires SHAP) |
| RandomForest (Balanced) | ~0.62 | ~0.74 | ~0.83 | ~50 ms | Medium (200 trees) |
| DecisionTree (Balanced) | ~0.59 | ~0.76 | ~0.81 | ~3 ms | High (single tree) |
| KNN (Default) | ~0.56 | ~0.54 | ~0.80 | ~150 ms | Low (distance-based) |

## Project Structure

```text
telco-churn-prediction/
├── data/                    # Raw Telco Customer Churn CSV
├── models/                  # Serialized production pipeline (.joblib)
├── notebooks/               # EDA, benchmarking, and error analysis
├── src/
│   ├── data_prep.py         # Data cleaning & row-wise feature engineering
│   ├── train.py             # Stratified split, CV, tuning & serialization
│   └── predict.py           # Inference wrapper with risk-tier logic
├── static/
│   └── index.html           # Responsive web frontend for FastAPI
├── api.py                   # FastAPI backend (/predict & /docs)
├── app.py                   # Interactive Streamlit dashboard
├── pyproject.toml           # uv dependency configuration
└── requirements.txt         # Exported pip requirements
