# Target Resolver Agent


import pandas as pd
from typing import Optional, Dict
from Backend.agents.utils.dataset_contract import load_dataset_contract


class TargetResolverAgent:
    """
    Resolves:
    1. Target column
    2. Problem type (classification / regression)

    PURELY CONTRACT-BASED.
    Reads from data/dataset_contract.json
    """

    def __init__(self, cleaned_data_path: str):
        self.cleaned_data_path = cleaned_data_path
        self.df = pd.read_csv(cleaned_data_path)

    def run(
        self,
        user_target: Optional[str] = None,
        mode: str = "auto"
    ) -> Dict[str, str]:
        """
        Loads the target and problem type from the dataset contract.
        Ignores 'user_target' unless explicit manual override is needed,
        but typically we trust the contract now.
        """
        
        # Load contract
        try:
            contract = load_dataset_contract()
            print(f"📋 Target Resolver loaded contract: {contract}")
        except FileNotFoundError:
            raise RuntimeError("Dataset contract not found. Ensure MasterOrchestrator runs properly.")

        target = contract["target_column"]
        problem_type = contract["problem_type"]

        # Validate against dataframe just in case
        if target not in self.df.columns:
            raise ValueError(f"Contract target '{target}' not found in dataset columns: {list(self.df.columns)}")

        return {
            "target_column": target,
            "problem_type": problem_type,
            "selection_mode": "contract"
        }
