"""
ULTIMATE HYBRID DATA CLEANING AGENT
- LangGraph Orchestrated
- Full 13-step Data Cleaning Pipeline
- DuckDB-powered Profiling (Pre & Post)
- SQLite Run Memory
- MAS-ready & Industry-aligned
"""

import os
import sqlite3
from datetime import datetime
from typing import Annotated, Optional, Dict, Any, Literal
from typing_extensions import TypedDict

import pandas as pd
import duckdb

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
# from langchain_ollama import ChatOllama

# Imports moved inside functions to prevent cycles

# ======================================================
# SQLITE MEMORY (RUN TRACKING)
# ======================================================

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "agents", "storage", "agent_memory.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cleaning_runs (
            run_id TEXT PRIMARY KEY,
            file_type TEXT,
            status TEXT,
            rows_before INTEGER,
            rows_after INTEGER,
            duplicates_before INTEGER,
            duplicates_after INTEGER,
            missing_before INTEGER,
            missing_after INTEGER,
            started_at TEXT,
            ended_at TEXT,
            output_path TEXT,
            error TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_run(run_id, file_type, status, metrics: dict, output_path=None, error=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO cleaning_runs
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        file_type,
        status,
        metrics.get("rows_before"),
        metrics.get("rows_after"),
        metrics.get("duplicates_before"),
        metrics.get("duplicates_after"),
        metrics.get("missing_before"),
        metrics.get("missing_after"),
        metrics.get("started_at"),
        datetime.now().isoformat(),
        output_path,
        error
    ))
    conn.commit()
    conn.close()

# ======================================================
# DUCKDB PROFILING (EDA / VALIDATION LAYER)
# ======================================================

def duckdb_profile(df: pd.DataFrame) -> Dict[str, Any]:
    con = duckdb.connect()
    con.register("data", df)

    rows = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    duplicates = con.execute(
        "SELECT (SELECT COUNT(*) FROM data) - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM data))"
    ).fetchone()[0]

    missing = {
        col: con.execute(
            f'SELECT COUNT(*) FROM data WHERE "{col}" IS NULL'
        ).fetchone()[0]
        for col in df.columns
    }

    con.close()

    return {
        "rows": int(rows),
        "duplicates": int(duplicates),
        "missing": missing,
        "missing_total": int(sum(missing.values()))
    }

# ======================================================
# AGENT STATE
# ======================================================

class CleaningState(TypedDict):
    messages: Annotated[list, add_messages]
    run_id: str

    file_path: str
    file_type: str
    df: Optional[pd.DataFrame]

    data_loaded: bool
    analyzed: bool
    data_cleaned: bool

    metrics: Dict[str, Any]

    output_path: str
    cleaning_report: str
    error: str

    # pipeline controls
    scale: bool
    scale_method: str
    encode: bool
    onehot_max: int
    schema: Optional[Dict[str, Any]]
    business_rules: Optional[list]
    target_column: Optional[str]

# ======================================================
# LOAD NODE
# ======================================================

def load_node(state: CleaningState) -> CleaningState:
    try:
        from Backend.agents.data_engineer.data_loader_agent import DataLoader
        from Backend.agents.utils.cleaner_helper import validate_dataframe

        loader = DataLoader()
        fp, ft = state["file_path"], state["file_type"]

        if ft == "csv":
            df = loader.load_csv(fp)
        elif ft == "excel":
            df = loader.load_excel(fp)
        elif ft == "api":
            df = loader.load_from_api(fp)
        elif ft == "database":
            conn, query = fp.split("|", 1)
            df = loader.load_from_database(conn, query)
        else:
            raise ValueError("Unsupported file type")

        valid, issues = validate_dataframe(df)
        if not valid:
            raise ValueError(", ".join(issues))

        profile = duckdb_profile(df)

        metrics = {
            "rows_before": profile["rows"],
            "duplicates_before": profile["duplicates"],
            "missing_before": profile["missing_total"],
            "started_at": datetime.now().isoformat()
        }

        return {
            **state,
            "df": df,
            "metrics": metrics,
            "data_loaded": True,
            "messages": state["messages"] + [
                HumanMessage(
                    content=f"✅ Loaded data | Rows={profile['rows']} | "
                            f"Duplicates={profile['duplicates']} | "
                            f"Missing={profile['missing_total']}"
                )
            ]
        }

    except Exception as e:
        return {**state, "error": str(e)}

# ======================================================
# ANALYZE NODE (DUCKDB-BASED)
# ======================================================

def analyze_node(state: CleaningState) -> CleaningState:
    df = state["df"]
    profile = duckdb_profile(df)

    # Basic stats for prompt
    stats_summary = (
        f"Rows: {profile['rows']}\n"
        f"Duplicates: {profile['duplicates']}\n"
        f"Missing Values: {profile['missing']}\n"
        f"Total Missing: {profile['missing_total']}"
    )

    report_content = (
        "**Data Quality Summary (AI Analysis Disabled)**\n\n"
        f"- Rows: {profile['rows']:,}\n"
        f"- Duplicates: {profile['duplicates']:,}\n"
        f"- Missing Values: {profile['missing_total']:,}\n\n"
        "System is proceeding with standard cleaning pipeline."
    )

    return {
        **state,
        "analyzed": True,
        "messages": state["messages"] + [HumanMessage(content=report_content)]
    }

# ======================================================
# CLEAN NODE (FULL 13-STEP PIPELINE)
# ======================================================

def clean_node(state: CleaningState) -> CleaningState:
    try:
        from Backend.agents.utils.cleaner_helper import DataCleaner

        df = state["df"]
        cleaner = DataCleaner(
            schema=state.get("schema"),
            business_rules=state.get("business_rules")
        )

        # 1–2 Schema & Dtypes
        df = cleaner.apply_schema(df)
        df = cleaner.correct_dtypes(df)

        # -----------------------------
        # 0. NORMALIZE COLUMNS (Critical for Target Matching)
        # -----------------------------
        df = cleaner.normalize_columns(df)
        
        # Normalize target input to match exactly using SAME logic
        if state.get("target_column"):
            state["target_column"] = cleaner.clean_string(state["target_column"])
            
        # 3 Missing Values
        df = cleaner.handle_missing_values(df)

        # 4 Outliers (report only)
        cleaner.detect_outliers(df)

        # 5 Duplicates
        df = cleaner.remove_duplicates(df)

        # 5.5 Drop Constant Columns (Zero Variance)
        df = cleaner.drop_constant_columns(df)

        # 6 Text Standardization
        df = cleaner.standardize_text(df)

        # -------------------------------------------------
        # STEP 6.5: SAVE HUMAN READABLE DATA (Pre-Scale/Encode)
        # -------------------------------------------------
        os.makedirs("data/processed", exist_ok=True)
        human_readable_path = "data/processed/human_readable.csv"
        df.to_csv(human_readable_path, index=False)


        from sklearn.preprocessing import StandardScaler

        # -------------------------------------------------
        # STEP 0: Hard target separation (MANDATORY & ROBUST)
        # -------------------------------------------------
        target_col = state.get("target_column")
        df, y = cleaner.separate_target(df, target_col)

        # -------------------------------------------------
        # STEP 1: Drop ID & Exclusion columns (Robust)
        # -------------------------------------------------
        from Backend.agents.utils.cleaner_helper import get_excluded_columns
        
        # We pass target_col (even if dropped) just in case, but strict logic dropped it in Step 0
        exclude_cols = get_excluded_columns(df, target_col) 
        df = df.drop(columns=exclude_cols, errors="ignore")

        # -------------------------------------------------
        # STEP 2: Encode categorical FEATURES ONLY
        # -------------------------------------------------
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # -------------------------------------------------
        # STEP 2.5: Safety Check (Fail Early)
        # -------------------------------------------------
        if df.select_dtypes(include=["object"]).shape[1] > 0:
            object_cols = df.select_dtypes(include=["object"]).columns.tolist()
            raise ValueError(
                f"Non-numeric columns remain after encoding: {object_cols}. "
                "Likely ID or text leakage."
            )

        # -------------------------------------------------
        # STEP 3: Scale numeric FEATURES ONLY
        # -------------------------------------------------
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        # -------------------------------------------------
        # STEP 4: Reattach target
        # -------------------------------------------------
        if y is not None:
            df[target_col] = y

        # 9 Leakage Detection
        cleaner.detect_leakage(df, state.get("target_column"))

        # 10 Business Rules
        cleaner.apply_business_rules(df)

        # 11 Metadata
        cleaner.generate_metadata(df)

        # -----------------------------
        # FINAL SAFETY CHECK
        # -----------------------------
        object_cols = df.select_dtypes(include=["object", "category"]).columns
        # Exclude target from this check (it can optionally be object)
        target_col = state.get("target_column")
        if target_col and target_col in object_cols:
            object_cols = object_cols.drop(target_col)

        if len(object_cols) > 0:
             raise ValueError(
                f"Data Engineer Error: Non-numeric feature columns remain: {list(object_cols)}"
            )

        # 12 Save Output
        os.makedirs("data/processed", exist_ok=True)
        output = state.get("output_path") or "data/processed/cleaned_data.csv"
        cleaner.save_cleaned(df, output)

        # 13 Post-clean DuckDB Validation
        post_profile = duckdb_profile(df)

        state["metrics"].update({
            "rows_after": post_profile["rows"],
            "duplicates_after": post_profile["duplicates"],
            "missing_after": post_profile["missing_total"]
        })

        log_run(
            state["run_id"],
            state["file_type"],
            "SUCCESS",
            metrics=state["metrics"],
            output_path=output
        )

        return {
            **state,
            "data_cleaned": True,
            "output_path": output,
            "cleaning_report": cleaner.get_cleaning_report(),
            "metrics": state["metrics"], # Explicitly include metrics
            "messages": state["messages"] + [
                HumanMessage(
                    content=f"✅ Cleaning complete | "
                            f"Rows {state['metrics']['rows_before']} → {post_profile['rows']} | "
                            f"Missing {state['metrics']['missing_before']} → {post_profile['missing_total']}"
                )
            ]
        }

    except Exception as e:
        log_run(
            state["run_id"],
            state["file_type"],
            "FAILED",
            metrics=state.get("metrics", {}),
            error=str(e)
        )
        return {**state, "error": str(e)}

# ======================================================
# ROUTER
# ======================================================

def route(state: CleaningState) -> Literal["analyze", "clean", "end"]:
    if state.get("error"):
        return "end"
    if not state.get("data_loaded"):
        return "end"
    if state.get("data_cleaned"):
        return "end"
    if state.get("analyzed"):
        return "clean"
    return "analyze"

# ======================================================
# BUILD AGENT
# ======================================================

def create_cleaning_agent():
    init_db()

    graph = StateGraph(CleaningState)
    graph.add_node("load", load_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("clean", clean_node)

    graph.add_edge(START, "load")
    graph.add_conditional_edges("load", route, {"analyze": "analyze", "end": END})
    graph.add_conditional_edges("analyze", route, {"clean": "clean", "end": END})
    graph.add_edge("clean", END)

    return graph.compile()


# ======================================================
# ENTRYPOINT
# ======================================================

def clean_file(file_path: str, file_type: str = "csv", flags: Optional[Dict[str, Any]] = None):
    flags = flags or {}
    agent = create_cleaning_agent()

    state = {
        "run_id": f"run_{datetime.now().timestamp()}",
        "messages": [],
        "file_path": file_path,
        "file_type": file_type,
        "df": None,
        "data_loaded": False,
        "analyzed": False,
        "data_cleaned": False,
        "metrics": {},
        "cleaning_report": "",
        "output_path": flags.get("output_path", ""),
        "error": "",
        **flags,
    }

    result = agent.invoke(state)
    return {
        "success": result.get("data_cleaned"),
        "output_path": result.get("output_path"),
        "normalized_target": result.get("target_column"),  # <--- CRITICAL FIX
        "report": result.get("cleaning_report"),
        "metrics": result.get("metrics"),
        "error": result.get("error"),
    }
