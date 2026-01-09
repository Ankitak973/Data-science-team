# Orchestrator Agent


from typing import TypedDict, Any, Optional
from langgraph.graph import StateGraph, START, END

# Agents
from Backend.agents.data_scientist.target_resolver import TargetResolverAgent
from Backend.agents.data_scientist.feature_selector import FeatureSelectorAgent
from Backend.agents.data_scientist.model_trainer import ModelTrainingAgent
from Backend.agents.data_scientist.evaluator import EvaluationExplainabilityAgent


# ======================================================
# STATE DEFINITION
# ======================================================

class DataScientistState(TypedDict):
    cleaned_data_path: str
    user_target: Optional[str]

    # From Target Resolver
    target_column: str
    problem_type: str
    selection_mode: str

    # From Feature Selector
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    features_used: list

    # From Model Trainer
    model_path: str
    baseline_score: float
    best_score: float
    best_model: str

    # From Evaluator
    evaluation_report: dict


# ======================================================
# NODES
# ======================================================

def target_resolver_node(state: DataScientistState):
    agent = TargetResolverAgent(
        cleaned_data_path=state["cleaned_data_path"]
    )

    # Manual override if present
    if state.get("user_target"):
        result = agent.run(
            user_target=state["user_target"],
            mode="manual"
        )
    else:
        result = agent.run()

    return {
        "target_column": result["target_column"],
        "problem_type": result["problem_type"],
        "selection_mode": result["selection_mode"]
    }


def feature_selection_node(state: DataScientistState):
    agent = FeatureSelectorAgent(
        cleaned_data_path=state["cleaned_data_path"],
        target_column=state["target_column"]
    )

    result = agent.run()

    return {
        "X_train": result["X_train"],
        "X_test": result["X_test"],
        "y_train": result["y_train"],
        "y_test": result["y_test"],
        "features_used": result["features_used"]
    }


def model_training_node(state: DataScientistState):
    agent = ModelTrainingAgent(
        problem_type=state["problem_type"]
    )

    result = agent.run(
        state["X_train"],
        state["X_test"],
        state["y_train"],
        state["y_test"]
    )

    return {
        "model_path": result["model_path"],
        "baseline_score": result["baseline_score"],
        "best_score": result["best_score"],
        "best_model": result["best_model"]
    }


def evaluation_node(state: DataScientistState):
    agent = EvaluationExplainabilityAgent(
        model_path=state["model_path"],
        problem_type=state["problem_type"]
    )

    report = agent.run(
        state["X_test"],
        state["y_test"]
    )

    return {
        "evaluation_report": report
    }


# ======================================================
# GRAPH BUILDING
# ======================================================

def build_data_scientist_graph():
    graph = StateGraph(DataScientistState)

    graph.add_node("target_resolver", target_resolver_node)
    graph.add_node("feature_selection", feature_selection_node)
    graph.add_node("model_training", model_training_node)
    graph.add_node("evaluation", evaluation_node)

    graph.add_edge(START, "target_resolver")
    graph.add_edge("target_resolver", "feature_selection")
    graph.add_edge("feature_selection", "model_training")
    graph.add_edge("model_training", "evaluation")
    graph.add_edge("evaluation", END)

    return graph.compile()


# ======================================================
# PUBLIC ENTRYPOINT (ONE CLICK)
# ======================================================

def run_data_scientist_pipeline(
    cleaned_data_path: str,
    user_target: str | None = None
):
    graph = build_data_scientist_graph()

    initial_state = {
        "cleaned_data_path": cleaned_data_path,
        "user_target": user_target
    }

    final_state = graph.invoke(initial_state)
    return final_state
