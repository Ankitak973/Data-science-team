
import sys
import os
import pandas as pd
import json

# Adjust path to find Backend
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

try:
    from Backend.agents.data_scientist.report_generator import DataScientistReportGenerator
    from Backend.agents.data_analyst.eda_agent import should_continue_eda
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
    # mock for syntax check if imports fail due to path structure in this env
    pass

def test_report_generator_robustness():
    print("\n--- Testing Report Generator Robustness ---")
    gen = DataScientistReportGenerator(report_dir="tmp/reports")
    
    # 1. Test with empty pipeline state (shouldn't crash, might produce basic error report)
    empty_state = {
        "target_column": "churn",
        "problem_type": "classification",
        "best_model": "TestModel",
        "baseline_score": 0.5,
        "best_score": 0.6,
        "evaluation_report": {
            "metrics": "{'roc_auc': 0.7}",  # String metric
            "sanity_check": {},
            "top_features": [] # Empty features
        },
        "cleaned_data_path": "non_existent.csv"
    }
    
    try:
        # This will fail to load data, but should run through _generate_llm_narrative with "Data not available"
        res = gen.run(empty_state)
        print("Empty state run success")
        print("Report path:", res["report_path"])
    except Exception as e:
        print(f"Empty state run failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Test with malformed metrics
    malformed_state = empty_state.copy()
    malformed_state["evaluation_report"]["metrics"] = "Not a dict"
    try:
        res = gen.run(malformed_state)
        print("Malformed metrics run success")
    except Exception as e:
        print(f"Malformed metrics run failed: {e}")

if __name__ == "__main__":
    test_report_generator_robustness()
