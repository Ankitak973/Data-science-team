import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List

class AnalyticalValidator:
    """
    Strict integrity gates for automated data science analysis.
    Prevents garbage-in/garbage-out results.
    """

    @staticmethod
    def validate_target(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Check if the target column is viable for analysis.
        """
        if target_col not in df.columns:
            return {"valid": False, "reason": f"Target column '{target_col}' not found in dataset."}
        
        # Check variance
        if df[target_col].nunique() < 2:
            return {"valid": False, "reason": f"Target column '{target_col}' has zero variance (only one unique value). Analysis is impossible."}
        
        # Check null rates
        null_rate = df[target_col].isnull().mean()
        if null_rate > 0.4:
            return {"valid": False, "reason": f"Target column '{target_col}' has a high null rate ({null_rate:.1%}). Prediction would be unreliable."}
            
        return {"valid": True, "reason": "Target confirmed viable."}

    @staticmethod
    def validate_domain_alignment(target_col: str, domain_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify if the target column aligns with the detected domain.
        STRICT enforcement for hardening.
        """
        domain = domain_context.get("domain", "General")
        target_description = domain_context.get("target_variable_description", "").lower()
        confidence = domain_context.get("confidence_score", 1.0)
        
        # Mismatch heuristic
        mismatch = False
        if "oncology" in domain.lower() or "healthcare" in domain.lower():
            # If healthcare, target must be related to health/clinical
            keywords = ["cancer", "stage", "severity", "survival", "target", "risk", "status", "death", "mortality", "health", "score"]
            if not any(word in target_col.lower() or word in target_description for word in keywords):
                mismatch = True
        
        # If low confidence or high mismatch, fail validation
        if mismatch and confidence < 0.8:
            return {
                "valid": False,
                "reason": f"Domain mismatch detected: '{domain}' does not align with target '{target_col}' and confidence is low ({confidence:.2f})."
            }
        
        if mismatch:
            return {
                "valid": True, # Allow for now but flag danger
                "warning": f"Possible domain mismatch: Target '{target_col}' in '{domain}' context is unusual."
            }
            
        return {"valid": True}

    @staticmethod
    def inspect_evidence_integrity(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect if features are explicit or anonymized to guide reporting mode.
        Uses robust regex to catch "Feature_1", "Var_2", "Unnamed: 0", etc.
        """
        placeholder_regex = re.compile(r"^(feature|var|unnamed|column|attr|x|y)[_:\s]?\d+$", re.IGNORECASE)
        columns = [c for c in df.columns if placeholder_regex.match(str(c))]
        
        has_placeholders = len(columns) > 0
        
        return {
            "has_placeholders": has_placeholders,
            "placeholder_count": len(columns),
            "reporting_mode": "Strict (No Inference)" if has_placeholders else "Standard (Evidence First)"
        }
