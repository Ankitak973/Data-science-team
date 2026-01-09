# Evaluator Agent


import json
import joblib
import numpy as np
import os
from typing import Dict

import pandas as pd
from sklearn.inspection import permutation_importance

from Backend.agents.utils.metric_utils import (
    classification_metrics,
    regression_metrics
)
from Backend.agents.utils.dataset_contract import load_dataset_contract


class EvaluationExplainabilityAgent:
    """
    Evaluates trained model and generates explainability artifacts.
    """

    def __init__(
        self,
        model_path: str,
        problem_type: str,
        report_dir: str = "Backend/reports"
    ):
        self.model = joblib.load(model_path)
        
        # FORCE CONTRACT PROBLEM TYPE
        contract = load_dataset_contract()
        self.problem_type = contract["problem_type"]
        print(f"🔒 Evaluator validating for problem type: {self.problem_type}")

        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

    # ======================================================
    # PUBLIC ENTRYPOINT
    # ======================================================

    def run(self, X_test, y_test) -> Dict:
        # 1. Predictions
        y_pred = self.model.predict(X_test)

        # 2. Metrics
        metrics = self._compute_metrics(y_test, y_pred)

        # 3. Overfitting sanity check
        sanity = self._sanity_check(metrics)

        # 4. Feature importance
        feature_importance = self._feature_importance(X_test, y_test)

        # 5. Explanation summary
        explanation = self._generate_explanation(metrics, feature_importance)

        # 6. Save report
        report = {
            "metrics": metrics,
            "sanity_check": sanity,
            "top_features": feature_importance[:5],
            "explanation": explanation
        }

        report_path = os.path.join(self.report_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    # ======================================================
    # INTERNAL
    # ======================================================

    def _compute_metrics(self, y_true, y_pred) -> Dict:
        if self.problem_type == "classification":
            metrics = classification_metrics(y_true, y_pred)
            # STRICT CHECK
            if "r2" in metrics or "mae" in metrics:
                raise ValueError("❌ VIOLATION: Regression metrics found in Classification task!")
            return metrics
        else:
            metrics = regression_metrics(y_true, y_pred)
            # STRICT CHECK
            if "accuracy" in metrics or "f1" in metrics:
                raise ValueError("❌ VIOLATION: Classification metrics found in Regression task!")
            return metrics

    def _sanity_check(self, metrics: Dict) -> str:
        if self.problem_type == "classification":
            if metrics.get("accuracy", 0) < 0.55:
                return "Model performs close to random guessing"
            return "Model performance appears reasonable"

        if metrics.get("r2", 0) < 0:
            return "Model performs worse than baseline"
        return "Model performance appears reasonable"

    def _feature_importance(self, X_test, y_test):
        try:
            result = permutation_importance(
                self.model, X_test, y_test,
                n_repeats=5, random_state=42
            )

            importances = result.importances_mean
            features = X_test.columns

            feature_scores = list(
                zip(features, importances)
            )

            feature_scores.sort(
                key=lambda x: abs(x[1]),
                reverse=True
            )

            return [
                {"feature": f, "importance": float(score)}
                for f, score in feature_scores
            ]

        except Exception:
            return []

    def _generate_explanation(self, metrics, feature_importance):
        if not feature_importance:
            return "Feature importance could not be computed."

        top = feature_importance[0]
        return (
            f"The model performance is acceptable. "
            f"The most influential feature is '{top['feature']}', "
            f"which has the highest impact on predictions."
        )
