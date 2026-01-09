"""
Master Orchestrator
-------------------
Runs the full AI Data Team in sequence:

1. Data Engineer  → Cleaning & preprocessing
2. Data Analyst   → EDA + Visualizations
3. Data Scientist → Modeling + Evaluation + Report

This file contains NO business logic.
It only coordinates existing agents.
"""

from typing import Dict, Any, Optional

# -----------------------------
# DATA ENGINEER
# -----------------------------
from Backend.agents.data_engineer.data_cleaning_agent import clean_file

# -----------------------------
# DATA ANALYST
# -----------------------------
from Backend.agents.data_analyst.data_analyst_orchestrator import (
    DataAnalystOrchestrator
)

# -----------------------------
# DATA SCIENTIST
# -----------------------------
from Backend.agents.data_scientist.orchestrator import (
    run_data_scientist_pipeline
)
from Backend.agents.data_scientist.report_generator import (
    DataScientistReportGenerator
)


class MasterOrchestrator:
    """
    One-click AI Data Team Consultant
    """

    def run(
        self,
        file_path: str,
        file_type: str = "csv",
        target_column: Optional[str] = None,
        analyst_config: Optional[Dict[str, Any]] = None,
        engineer_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline.

        Parameters
        ----------
        file_path : str
            Path to uploaded dataset
        file_type : str
            csv / excel / api / db
        target_column : Optional[str]
            User-selected target (can be None → auto)
        analyst_config : dict
            Options like max_plots, corr_threshold
        engineer_config : dict
            Options like encoding, scaling, schema

        Returns
        -------
        Dict[str, Any]
            Combined output of all agents
        """

        analyst_config = analyst_config or {}
        engineer_config = engineer_config or {}

        # =====================================================
        # 1️⃣ DATA ENGINEER
        # =====================================================
        print("\n🧹 Running Data Engineer Agent...")
        engineer_result = clean_file(
            file_path=file_path,
            file_type=file_type,
            flags={
                **engineer_config,
                "target_column": target_column
            }
        )

        if engineer_result.get("error"):
            return {
                "success": False,
                "stage": "Data Engineer",
                "error": engineer_result["error"]
            }

        cleaned_data_path = engineer_result["output_path"]
        
        # -----------------------------
        # DATASET CONTRACT (HARDCODED)
        # -----------------------------
        from Backend.agents.utils.dataset_contract import write_dataset_contract
        import pandas as pd
        
        # Load cleaned dataset
        df = pd.read_csv(cleaned_data_path)
        
        # TEMP: hardcode target for now (NO AUTO yet)
        TARGET_COLUMN = "charges"
        
        contract = write_dataset_contract(df, TARGET_COLUMN)
        
        print("📌 Dataset contract locked:", contract)

        # USE THE CONTRACT TARGET
        features_target = contract["target_column"]

        # =====================================================
        # 2️⃣ DATA ANALYST
        # =====================================================
        print("\n📊 Running Data Analyst Agent...")
        analyst = DataAnalystOrchestrator(
            cleaned_file_path=cleaned_data_path,
            output_root="data"
        )

        analyst_result = analyst.run_full_pipeline(
            target_column=features_target,
            max_plots=analyst_config.get("max_plots", 30)
        )

        # =====================================================
        # 3️⃣ DATA SCIENTIST
        # =====================================================
        print("\n🤖 Running Data Scientist Agent...")
        scientist_state = run_data_scientist_pipeline(
            cleaned_data_path=cleaned_data_path,
            user_target=features_target
        )

        report_generator = DataScientistReportGenerator()
        scientist_report = report_generator.run(scientist_state)

        # =====================================================
        # FINAL RESPONSE
        # =====================================================
        print("\n✅ AI Data Team Pipeline Completed Successfully")

        return {
            "success": True,

            # Engineer outputs
            "data_engineer": engineer_result,

            # Analyst outputs
            "data_analyst": analyst_result,

            # Scientist outputs
            "data_scientist": {
                "pipeline_state": scientist_state,
                "report": scientist_report
            },

            # Shared context
            "context": {
                "cleaned_data_path": cleaned_data_path
            }
        }
