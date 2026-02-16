import json
import os

def infer_problem_type(df, target_col):
    """
    Rule-based detection. No LLM.
    """
    # Defensive check if dataframe is empty
    if df.empty:
        raise ValueError("Cannot infer problem type: DataFrame is empty")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    if df[target_col].dtype.kind in "if":  # int or float
        unique_vals = df[target_col].nunique()
        total_vals = len(df)
        
        # If very few unique values (e.g., < 10) relative to dataset size, it's classification
        if unique_vals < 10:
            return "classification"
        
        # If unique values are more than 5% of dataset and > 10, likely regression
        # or if unique values > 30 (common threshold for regression)
        if unique_vals > 30 or (unique_vals / total_vals > 0.05 and unique_vals > 10):
            return "regression"
        else:
            return "classification"
    else:
        return "classification"


def write_dataset_contract(df, target_col, domain_context=None, output_path="data/dataset_contract.json"):
    problem_type = infer_problem_type(df, target_col)

    contract = {
        "target_column": target_col,
        "problem_type": problem_type,
        "domain_context": domain_context
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(contract, f, indent=2)

    return contract


def load_dataset_contract(path="data/dataset_contract.json"):
    with open(path, "r") as f:
        return json.load(f)
