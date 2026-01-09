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
        if df[target_col].nunique() > 20:
            return "regression"
        else:
            return "classification"
    else:
        return "classification"


def write_dataset_contract(df, target_col, output_path="data/dataset_contract.json"):
    problem_type = infer_problem_type(df, target_col)

    contract = {
        "target_column": target_col,
        "problem_type": problem_type
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(contract, f, indent=2)

    return contract


def load_dataset_contract(path="data/dataset_contract.json"):
    with open(path, "r") as f:
        return json.load(f)
