# Model Utils


from typing import Tuple
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
        return mean_squared_error(y_test, y_pred, squared=False)  # RMSE


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

def train_automl_model(X_train, y_train, problem_type: str):
    if not HAS_OTOML:
        print("Warning: FLAML not found. Returning baseline model as AutoML placeholder.")
        # Fallback: Train baseline again or return a dummy that mimics API?
        # Better: just return baseline model but wrapper it?
        # For simplicity, we'll return the baseline model object, assuming it has .predict
        return train_baseline_model(X_train, y_train, problem_type)

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
        return mean_squared_error(y_test, y_pred, squared=False)


# -------------------------------------------------
# SAVE / LOAD
# -------------------------------------------------

def save_model(model, path: str):
    joblib.dump(model, path)
