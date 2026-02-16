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
            # Soft filtering instead of strict crashing
            keys_to_remove = [k for k in metrics if k in ["r2", "rmse", "mae"]]
            if keys_to_remove:
                print(f"⚠️ Removing unsupported regression metrics from classification report: {keys_to_remove}")
                for k in keys_to_remove:
                    del metrics[k]
            return metrics
        else:
            metrics = regression_metrics(y_true, y_pred)
            # Soft filtering instead of strict crashing
            keys_to_remove = [k for k in metrics if k in ["accuracy", "precision", "recall", "f1", "roc_auc"]]
            if keys_to_remove:
                print(f"⚠️ Removing unsupported classification metrics from regression report: {keys_to_remove}")
                for k in keys_to_remove:
                    del metrics[k]
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
        importances = None
        print(f"DEBUG: Computing importance for {X_test.shape[1]} features...")
        
        try:
            # 1. Try Permutation Importance (Model Agnostic)
            result = permutation_importance(
                self.model, X_test, y_test,
                n_repeats=3, random_state=42,
                n_jobs=-1 # Speed up
            )
            importances = result.importances_mean
            
            # If all zeros, permutation importance failed to catch signal
            if np.all(importances == 0):
                print("⚠️ Permutation importance returned all zeros. Trying intrinsic...")
                importances = None

        except Exception as e:
            print(f"⚠️ Permutation importance failed: {e}. Falling back to intrinsic attributes.")
            importances = None
            
        if importances is None:
            # 2. Try Tree-based Feature Importance
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
            
            # 3. Try Linear Model Coefficients
            elif hasattr(self.model, "coef_"):
                importances = np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
            
            # 4. Handle Pipeline (access last step)
            elif hasattr(self.model, "steps"):
                 last_step = self.model.steps[-1][1]
                 if hasattr(last_step, "feature_importances_"):
                     importances = last_step.feature_importances_
                 elif hasattr(last_step, "coef_"):
                     importances = np.abs(last_step.coef_[0]) if len(last_step.coef_.shape) > 1 else np.abs(last_step.coef_)
        
        if importances is not None:
             features = X_test.columns
             print(f"DEBUG: Found {len(importances)} importances and {len(features)} features.")
             if len(importances) == len(features):
                  feature_scores = list(zip(features, importances))
                  feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                  res = [{"feature": f, "importance": float(score)} for f, score in feature_scores]
                  print(f"DEBUG: Extracted {len(res)} scores. Top: {res[0] if res else 'None'}")
                  return res
             else:
                  print(f"❌ Mismatch: importances({len(importances)}) != features({len(features)})")
        
        return []

    def _generate_explanation(self, metrics, feature_importance):
        if not feature_importance:
            return "Feature importance could not be computed."

        # Load domain context for precision
        contract = load_dataset_contract()
        domain_ctx = contract.get("domain_context", {})
        entity = domain_ctx.get("entity_name", "Record")
        target_desc = domain_ctx.get("target_variable_description", "the outcome")

        top = feature_importance[0]
        return (
            f"Model evaluation complete for {entity} data. "
            f"The primary driver for predicting {target_desc} is '{top['feature']}'. "
            f"Strategic focus should prioritize this feature to improve outcomes."
        )
