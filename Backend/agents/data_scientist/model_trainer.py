# Model Trainer Agent


from typing import Dict
import os

from Backend.agents.utils.model_utils import (
    train_baseline_model,
    evaluate_baseline_model,
    train_automl_model,
    evaluate_automl_model,
    save_model
)


from Backend.agents.utils.dataset_contract import load_dataset_contract


class ModelTrainingAgent:
    """
    Model Training Agent
    - Trains baseline model
    - Runs AutoML
    - Selects best model
    """

    def __init__(self, problem_type: str, model_dir: str = "Backend/models"):
        # FORCE CONTRACT PROBLEM TYPE
        contract = load_dataset_contract()
        self.problem_type = contract["problem_type"]
        print(f"🔒 Model Trainer serving problem type: {self.problem_type}")
        
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    # ======================================================
    # PUBLIC ENTRYPOINT
    # ======================================================

    def run(self, X_train, X_test, y_train, y_test) -> Dict:
        # 1. Baseline model
        baseline_model = train_baseline_model(X_train, y_train, self.problem_type)
        baseline_score = evaluate_baseline_model(
            baseline_model, X_test, y_test, self.problem_type
        )

        # 2. AutoML model
        automl_model = train_automl_model(X_train, y_train, self.problem_type)
        automl_score = evaluate_automl_model(
            automl_model, X_test, y_test, self.problem_type
        )

        # 3. Select best model
        if self._is_automl_better(baseline_score, automl_score):
            best_model = automl_model
            best_score = automl_score
            best_model_name = "AutoML (FLAML)"
        else:
            best_model = baseline_model
            best_score = baseline_score
            best_model_name = "Baseline Model"

        # 4. Save model
        model_path = os.path.join(self.model_dir, "model.pkl")
        save_model(best_model, model_path)

        return {
            "best_model": best_model_name,
            "baseline_score": baseline_score,
            "best_score": best_score,
            "model_path": model_path
        }

    # ======================================================
    # INTERNAL
    # ======================================================

    def _is_automl_better(self, baseline_score, automl_score) -> bool:
        if self.problem_type == "classification":
            return automl_score > baseline_score
        else:
            return automl_score < baseline_score  # RMSE lower is better
