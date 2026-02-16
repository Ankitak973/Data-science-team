# Model Utils


from typing import Tuple
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error


# -------------------------------------------------
# BASELINE MODELS
# -------------------------------------------------

def train_baseline_model(X_train, y_train, problem_type: str):
    if problem_type == "classification":
        model = LogisticRegression(max_iter=1000)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)
    return model


def evaluate_baseline_model(model, X_test, y_test, problem_type: str) -> float:
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        return f1_score(y_test, y_pred, average="weighted")
    else:
        return np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE


# -------------------------------------------------
# AUTOML (FLAML) - OPTIONAL
# -------------------------------------------------

HAS_OTOML = False
AutoML = None
try:
    # Try importing flaml first, then AutoML
    import flaml
    from flaml import AutoML
    HAS_OTOML = True
except Exception as e:
    print(f"Warning: Could not import FLAML AutoML ({e}). Using baseline only.")
    HAS_OTOML = False
    AutoML = None

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def train_automl_model(X_train, y_train, problem_type: str):
    if not HAS_OTOML:
        print("Warning: FLAML not found. Using Random Forest as robust fallback.")
        if problem_type == "classification":
            model = RandomForestClassifier(
                n_estimators=200, 
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=200, 
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        return model

    automl = AutoML()

    settings = {
        "time_budget": 10,  # reduced for speed in testing
        "task": "classification" if problem_type == "classification" else "regression",
        "log_file_name": None,
        "verbose": 0
    }

    automl.fit(X_train=X_train, y_train=y_train, **settings)
    return automl


def evaluate_automl_model(automl, X_test, y_test, problem_type: str) -> float:
    # Works for both FLAML AutoML and sklearn models (baseline fallback)
    y_pred = automl.predict(X_test)

    if problem_type == "classification":
        return f1_score(y_test, y_pred, average="weighted")
    else:
        return np.sqrt(mean_squared_error(y_test, y_pred))


# -------------------------------------------------
# SAVE / LOAD
# -------------------------------------------------

def save_model(model, path: str):
    joblib.dump(model, path)
