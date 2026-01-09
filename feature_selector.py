import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split

from Backend.agents.utils.feature_utils import (
    detect_id_columns,
    drop_low_variance_features,
    drop_highly_correlated_features,
    detect_leakage_features
)
from Backend.agents.utils.dataset_contract import load_dataset_contract


class FeatureSelectorAgent:
    """
    Feature Selection & Validation Agent
    Fully aligned with industry-standard protocol.
    """

    def __init__(self, cleaned_data_path: str, target_column: str):
        self.df = pd.read_csv(cleaned_data_path)
        
        # FORCE CONTRACT TARGET
        contract = load_dataset_contract()
        self.target_column = contract["target_column"]
        print(f"🔒 Feature Selector using contract target: {self.target_column}")

    # ======================================================
    # PUBLIC ENTRYPOINT
    # ======================================================

    def run(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        X, y = self._separate_features_and_target()
        dropped_features = []

        # 🔐 SAFETY GUARD (THIS IS THE FIX)
        X = X.select_dtypes(include=["number"])

        # 1. Remove ID-like columns
        id_cols = detect_id_columns(X)
        X = X.drop(columns=id_cols)
        dropped_features.extend(id_cols)

        # 2. Remove near-zero variance features
        X = drop_low_variance_features(X)

        # 3. Remove highly correlated features
        X, corr_dropped = drop_highly_correlated_features(X)
        dropped_features.extend(corr_dropped)

        # 4. Detect leakage-prone features
        leakage_cols = detect_leakage_features(X, y)
        X = X.drop(columns=leakage_cols)
        dropped_features.extend(leakage_cols)

        # 5. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "features_used": list(X.columns),
            "features_dropped": dropped_features
        }

    # ======================================================
    # INTERNAL
    # ======================================================

    def _separate_features_and_target(self):
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found.")

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # 🔐 LEAKAGE ASSERTION (CRITICAL)
        assert self.target_column not in X.columns, "TARGET LEAKAGE DETECTED: Target column found in features!"
        
        return X, y
