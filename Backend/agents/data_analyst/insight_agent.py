from typing import Dict
# from langchain_community.llms import Ollama
class InsightAgent:
    """
    Conversational agent that explains Data Scientist outputs
    to technical and non-technical users.
    """

    def __init__(self, model_name: str = "llama3.1"):
        self.llm = None

    def answer(
        self,
        user_question: str,
        data_scientist_report: Dict
    ) -> str:
        """
        user_question: natural language question
        data_scientist_report: final structured DS report
        """

        system_prompt = """
You are a senior data scientist assistant.
You explain machine learning results using ONLY the provided context.
You must not invent metrics, features, or results.
If something is not present, say you do not have enough information.
Use clear, non-technical language unless the user asks for technical detail.
"""

        context = self._build_context(data_scientist_report)

        final_prompt = f"""
{system_prompt}

DATA SCIENCE CONTEXT:
{context}

USER QUESTION:
{user_question}

Answer clearly and truthfully.
"""

        # response = self.llm.invoke(final_prompt)
        # return response.strip()
        return "Insights disabled (Ollama removed)."

    def _build_context(self, report: Dict) -> str:
        problem = report.get("problem_framing", {})

        return f"""
Problem Type: {problem.get("problem_type", "unknown")}
Target Column: {problem.get("target_column", "unknown")}

Model Used: {report.get("model_training", {}).get("best_model", "unknown")}
Baseline Score: {report.get("model_training", {}).get("baseline_score", "unknown")}
Best Score: {report.get("model_training", {}).get("best_score", "unknown")}

Evaluation Metrics:
{report.get("evaluation_metrics", "Not available")}

Sanity Check:
{report.get("sanity_check", "Not available")}

Top Influential Features:
{report.get("top_features", "Not available")}
"""
