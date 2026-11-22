import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_data(df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2, random_state: int = 42):
    """
    Splits features and target using stratification BEFORE any preprocessing
    to prevent data leakage and preserve the 73.5% / 26.5% class ratio.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer:
    - Numeric columns: median imputation + standard scaling
    - Categorical columns: most-frequent imputation + one-hot encoding (dense output)
    """
    # Using include/exclude=["number"] is safe across both pandas 2.x and 3.x
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",  # Prevents crashes if unseen categories appear in production
                    sparse_output=False,      # Dense output required by HistGradientBoostingClassifier
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor

import time
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def get_candidate_models():
    """
    Returns a dictionary of candidate models across three imbalance strategies:
    Default, class_weight='balanced', and SMOTE.
    """
    return {
        # --- 1. Baseline (Default) ---
        "LogReg (Default)": (LogisticRegression(max_iter=1000, random_state=42), False),
        "KNN (Default)": (KNeighborsClassifier(n_neighbors=15), False),
        "DecisionTree (Default)": (DecisionTreeClassifier(max_depth=6, random_state=42), False),
        "RandomForest (Default)": (RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1), False),
        "HistGB (Default)": (HistGradientBoostingClassifier(max_depth=6, random_state=42), False),

        # --- 2. Class-Weighted (Penalizes minority-class errors) ---
        "LogReg (Balanced)": (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42), False),
        "DecisionTree (Balanced)": (DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42), False),
        "RandomForest (Balanced)": (RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1), False),
        "HistGB (Balanced)": (HistGradientBoostingClassifier(max_depth=6, class_weight="balanced", random_state=42), False),

        # --- 3. SMOTE (Oversampling inside CV folds) ---
        "LogReg (SMOTE)": (LogisticRegression(max_iter=1000, random_state=42), True),
        "KNN (SMOTE)": (KNeighborsClassifier(n_neighbors=15), True),
        "RandomForest (SMOTE)": (RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1), True),
    }


def benchmark_models_cv(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Runs 5-fold Stratified CV for all candidate models and returns a sorted
    DataFrame comparing Accuracy, Precision, Recall, F1, ROC-AUC, and fit time.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = []

    for name, (model, use_smote) in get_candidate_models().items():
        preprocessor = build_preprocessor(X_train)

        if use_smote:
            # ImbPipeline ensures SMOTE only runs on the training fold, never the validation fold
            pipe = ImbPipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("smote", SMOTE(random_state=42)),
                    ("model", model),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

        start = time.perf_counter()
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.perf_counter() - start

        results.append(
            {
                "Model": name,
                "Accuracy": scores["test_accuracy"].mean(),
                "Precision": scores["test_precision"].mean(),
                "Recall": scores["test_recall"].mean(),
                "F1": scores["test_f1"].mean(),
                "ROC-AUC": scores["test_roc_auc"].mean(),
                "CV Time (s)": round(elapsed, 2),
            }
        )

    return pd.DataFrame(results).sort_values(by="F1", ascending=False).reset_index(drop=True)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def tune_top_models(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Tunes the top 2 models using 5-fold Stratified CV optimizing for F1 score:
    1. Logistic Regression (GridSearchCV - small, exact search space)
    2. HistGradientBoostingClassifier (RandomizedSearchCV - larger search space)
    Returns a dict of {model_name: fitted_search_object}.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- 1. Logistic Regression (GridSearchCV) ---
    logreg_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    logreg_params = {
        "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        "model__class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 2.5}, {0: 1, 1: 3}],
    }
    logreg_search = GridSearchCV(
        estimator=logreg_pipe,
        param_grid=logreg_params,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    logreg_search.fit(X_train, y_train)

    # --- 2. HistGradientBoostingClassifier (RandomizedSearchCV) ---
    hgb_pipe = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("model", HistGradientBoostingClassifier(random_state=42)),
        ]
    )
    hgb_params = {
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__max_iter": [100, 200, 300],
        "model__max_depth": [3, 4, 5, 6, None],
        "model__min_samples_leaf": [10, 20, 30, 50],
        "model__l2_regularization": [0.0, 0.1, 1.0, 5.0],
        "model__class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 2.5}],
    }
    hgb_search = RandomizedSearchCV(
        estimator=hgb_pipe,
        param_distributions=hgb_params,
        n_iter=30,
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    hgb_search.fit(X_train, y_train)

    return {
        "LogReg (Tuned)": logreg_search,
        "HistGB (Tuned)": hgb_search,
    }

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_models_on_test(
    models_dict: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """
    Fits each pipeline on X_train, records training and prediction times,
    and computes Accuracy, Precision, Recall, F1, and ROC-AUC on X_test.
    """
    rows = []
    fitted_models = {}

    for name, pipe in models_dict.items():
        # Measure training time
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time_ms = (time.perf_counter() - t0) * 1000

        # Measure inference time (predict + predict_proba on full test set)
        t1 = time.perf_counter()
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        pred_time_ms = (time.perf_counter() - t1) * 1000

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_prob),
                "Train Time (ms)": round(train_time_ms, 1),
                "Predict Time (ms)": round(pred_time_ms, 1),
            }
        )
        fitted_models[name] = pipe

    results_df = pd.DataFrame(rows).sort_values(by="F1", ascending=False).reset_index(drop=True)
    return results_df, fitted_models

from pathlib import Path
import joblib
from src.data_prep import add_features, load_and_clean_data


def train_and_save(
    data_path: str = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    model_path: str = "models/churn_pipeline.joblib",
) -> None:
    """
    End-to-end training entrypoint:
    Loads data -> cleans -> engineers features -> splits -> tunes -> saves best pipeline.
    """
    print("1. Loading and preparing data...")
    df = add_features(load_and_clean_data(data_path))
    X_train, X_test, y_train, y_test = split_data(df)

    print("2. Tuning top models (5-fold Stratified CV)...")
    tuned_searches = tune_top_models(X_train, y_train)

    # Deploy Tuned Logistic Regression based on Section 9 trade-off analysis
    best_pipe = tuned_searches["LogReg (Tuned)"].best_estimator_

    print("3. Evaluating final pipeline on held-out test set...")
    results_df, _ = evaluate_models_on_test(
        {"LogReg (Tuned)": best_pipe}, X_train, y_train, X_test, y_test
    )
    print(results_df.to_string(index=False))

    # Ensure models/ directory exists and save the full pipeline
    out_path = Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, out_path)
    print(f"\nSaved production pipeline to: {out_path.resolve()}")


if __name__ == "__main__":
    train_and_save()