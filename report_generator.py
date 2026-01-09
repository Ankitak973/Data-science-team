import json
import os
from typing import Dict

from langchain_community.llms import Ollama


class DataScientistReportGenerator:
    """
    Generates a final Data Scientist report that combines:
    - Statistical metrics & explainability
    - LLM-based narrative explanation (human friendly)
    """

    def __init__(
        self,
        llm_model: str = "llama3.1",
        report_dir: str = "Backend/reports"
    ):
        self.llm = Ollama(model=llm_model)
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

    # ======================================================
    # PUBLIC ENTRYPOINT
    # ======================================================

    def run(self, pipeline_state: Dict) -> Dict:
        """
        pipeline_state = final output from data_scientist orchestrator
        """

        narrative = self._generate_llm_narrative(pipeline_state)

        report = {
            "dataset_summary": self._dataset_summary(pipeline_state),
            "problem_framing": {
                "target_column": pipeline_state["target_column"],
                "problem_type": pipeline_state["problem_type"]
            },
            "model_training": {
                "best_model": pipeline_state["best_model"],
                "baseline_score": pipeline_state["baseline_score"],
                "best_score": pipeline_state["best_score"]
            },
            "evaluation_metrics": pipeline_state["evaluation_report"]["metrics"],
            "sanity_check": pipeline_state["evaluation_report"]["sanity_check"],
            "top_features": pipeline_state["evaluation_report"]["top_features"],
            "llm_explanation": narrative
        }

        report_path = os.path.join(
            self.report_dir, "data_scientist_report.json"
        )

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return {
            "report_path": report_path,
            "report": report
        }

    # ======================================================
    # INTERNAL METHODS
    # ======================================================

    def _dataset_summary(self, state: Dict):
        return {
            "features_used": len(state.get("features_used", []))
        }

    def _generate_llm_narrative(self, state: Dict) -> str:
        """
        Converts metrics + feature importance into human language
        """

        metrics = state["evaluation_report"]["metrics"]
        top_features = state["evaluation_report"]["top_features"][:3]
        problem_type = state["problem_type"]

        prompt = f"""
You are a senior data scientist explaining model results to a non-technical stakeholder.

Problem type: {problem_type}

Model performance metrics:
{metrics}

Top influential features:
{top_features}

Explain:
1. How good the model is
2. What mainly influences predictions
3. When the model should be trusted
4. Any caution or limitation

Use simple, clear language.
Do not mention algorithms.
Do not mention code.
"""

        response = self.llm.invoke(prompt)
        return response.strip()
