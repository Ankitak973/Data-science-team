import pandas as pd
import json
from langchain_ollama import OllamaLLM
from typing import Dict, Any

class DomainExpertAgent:
    """
    Identifies the domain and context of a dataset to guide downstream agents.
    Prevents hallucinations by providing a shared "source of truth" for what the data represents.
    """
    def __init__(self, model_name: str = "llama3.1"):
        self.llm = OllamaLLM(model=model_name)

    def run(self, df: pd.DataFrame, user_target: str = None) -> Dict[str, Any]:
        # 1. Prepare metadata for LLM
        columns = list(df.columns)
        # Increase sample size for better context detection (healthcare/finances etc)
        sample_data = df.head(10).to_dict(orient="records")
        target_hint = user_target if user_target else "None (Auto-detect)"
        
        prompt = f"""
        Analyze the following dataset metadata and identify its domain and context with high precision.
        
        COLUMNS: {columns}
        SAMPLE DATA (10 rows): {json.dumps(sample_data)}
        USER TARGET HINT: {target_hint}
        
        INSTRUCTIONS:
        - Analyze column names and values for clinical (e.g., diagnosis, severity, stage), behavioral (e.g., usage, payment, tenure), or financial markers.
        - If you see "cancer", "stage", "severity", "survival", it is Oncology/Healthcare.
        - If you see "tenure", "internet", "monthly", "churn", it is Telecom/Customer.
        - If you see "employee", "years_at_company", "salary", it is HR/Employer.
        
        STRICTLY IDENTIFY:
        1. Identify the 'domain' (e.g., Telecom, Oncology, HR, Real Estate, Finance). Be specific.
        2. Identify the 'target_variable' meaning. Exactly what is being predicted? (e.g., "Patient mortality risk", "Contract churn probability", "Property market value").
        3. Identify 'entity_name' (e.g., "Patient", "Customer", "Employee", "Property").
        4. Identify if 'financial_columns' are present for ROI calculations.
        5. Define the 'strategic_focus' (e.g., "Clinical Outcomes", "Employee Retention", "Asset Valuation").
        6. Determine if the target is a Risk (increase is bad) or an Opportunity (increase is good).
        
        RESPONSE FORMAT (JSON ONLY):
        {{
            "domain": "...",
            "target_variable_description": "...",
            "entity_name": "...",
            "financial_context_available": true/false,
            "strategic_focus": "...",
            "target_nature": "risk/opportunity",
            "recommended_tone": "...",
            "confidence_score": 0.0-1.0,
            "justification": "Why did you choose this domain/target?"
        }}
        """
        
        response = self.llm.invoke(prompt)
        try:
            # Clean response to ensure it's valid JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            context = json.loads(response[start:end])
            return context
        except Exception as e:
            print(f"⚠️ DomainExpertAgent failed to parse JSON: {e}")
            return {
                "domain": "General",
                "target_variable_description": "Unknown target",
                "entity_name": "Record",
                "financial_context_available": False,
                "strategic_focus": "General Performance",
                "recommended_tone": "Professional"
            }
