import streamlit as st
import pandas as pd
import os
import time
import json
import shutil
from langchain_community.llms import Ollama

# Add Backend to path
import sys
sys.path.append(os.getcwd())

from Backend.agents.master_orchestrator import MasterOrchestrator

# ======================================================
# CONFIG & STYLING
# ======================================================
st.set_page_config(
    page_title="AI Data Team",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        font-size: 1.2em;
    }
    .stButton>button:hover {
        background-color: #FF2B2B;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41424C;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-user {
        background-color: #262730;
    }
    .chat-bot {
        background-color: #1E1E1E;
        border: 1px solid #41424C;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# SESSION STATE
# ======================================================
if "results" not in st.session_state:
    st.session_state.results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df_preview" not in st.session_state:
    st.session_state.df_preview = None

# ======================================================
# HELPER: CONSULTANT CHAT
# ======================================================
# ======================================================
# HELPER: CONSULTANT CHAT
# ======================================================
def get_consultant_response(role, question, context, plots_list=None):
    try:
        llm = Ollama(model="llama3.1")
        
        system_prompts = {
            "Data Engineer": "You are an expert Data Engineer. Answer based on the data cleaning report.",
            "Data Analyst": "You are a Senior Data Analyst. Answer based on the EDA insights and visualizations. If the user asks for a chart and you have a relevant one in the Available Plots list, you MUST mention it using the format <<IMAGE: filename.png>> at the end of your response.",
            "Data Scientist": "You are a Lead Data Scientist. Answer based on the machine learning model results."
        }
        
        extra_context = ""
        if role == "Data Analyst" and plots_list:
            extra_context = f"\nAVAILABLE PLOTS (Choose one if relevant):\n{', '.join(plots_list)}\n"
        
        prompt = f"""
{system_prompts.get(role, "You are a helpful AI Consultant.")}

CONTEXT:
{context}
{extra_context}

USER QUESTION:
{question}

Answer strictly based on the context provided. Be professional and concise.
If referring to a plot, ensure the filename appears exactly as listed in the available plots.
"""
        return llm.invoke(prompt)
    except Exception as e:
        return f"⚠️ Consultant unavailable: {str(e)}"

# ======================================================
# 1️⃣ DATASET INTAKE
# ======================================================
st.title("🤖 AI Data Team")

st.markdown("### 1️⃣ Dataset Intake")
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

file_path = None
if uploaded_file:
    # Save file
    os.makedirs("data/raw", exist_ok=True)
    file_path = os.path.join("data/raw", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load preview
    if st.session_state.df_preview is None or st.session_state.get('current_file') != uploaded_file.name:
        df = pd.read_csv(file_path)
        st.session_state.df_preview = df
        st.session_state.current_file = uploaded_file.name
    
    df = st.session_state.df_preview
    
    # Info Stats
    c1, c2, c3 = st.columns(3)
    c1.info(f"**Rows:** {df.shape[0]}")
    c2.info(f"**Columns:** {df.shape[1]}")
    c3.info(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
    
    # Preview
    with st.expander("👀 View Dataset Preview"):
        st.dataframe(df.head())

# ======================================================
# 2️⃣ CONFIGURATION
# ======================================================
st.markdown("### 2️⃣ Configuration")
col1, col2 = st.columns(2)

target_col = None
max_plots = 30

with col1:
    if st.session_state.df_preview is not None:
        cols = ["Auto-detect"] + list(st.session_state.df_preview.columns)
        choice = st.selectbox("🎯 Target Column", cols, index=0)
        if choice != "Auto-detect":
            target_col = choice
    else:
        st.selectbox("🎯 Target Column", ["Upload data first"], disabled=True)

with col2:
    max_plots = st.slider("📊 Max Plots", min_value=10, max_value=50, value=30)

# ======================================================
# 3️⃣ EXECUTION
# ======================================================
st.markdown("### 3️⃣ Execution")
run_btn = st.button("🚀 Run AI Data Team", use_container_width=True)

if run_btn and file_path:
    orchestrator = MasterOrchestrator()
    
    # Container for visible progress
    prog_container = st.container()
    
    with st.status("🚀 AI Data Team is working...", expanded=True) as status:
        # 1. Engineer
        st.write("🧹 Data Engineer: Cleaning data...")
        engineer_config = {"output_path": "data/processed/cleaned_data.csv"}
        
        # 2. Analyst / Scientist triggers
        # We run the whole thing via orchestrator
        results = orchestrator.run(
            file_path=file_path,
            target_column=target_col,
            analyst_config={"max_plots": max_plots},
            engineer_config=engineer_config
        )
        
        if results["success"]:
            st.session_state.results = results
            status.update(label="✅ Pipeline Completed!", state="complete", expanded=False)
        else:
            status.update(label="❌ Pipeline Failed", state="error")
            st.error(results.get("error"))

# ======================================================
# 4️⃣ RESULTS DASHBOARD
# ======================================================
if st.session_state.results:
    res = st.session_state.results
    st.markdown("---")
    st.markdown("### 4️⃣ Results Dashboard")
    
    tab_eng, tab_ana, tab_sci = st.tabs([
        "🧹 Data Engineer", "📊 Data Analyst", "🤖 Data Scientist"
    ])
    
    # --- DATA ENGINEER ---
    with tab_eng:
        eng = res.get("data_engineer", {})
        metrics = eng.get("metrics", {})
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows Removed", (metrics.get("rows_before", 0) - metrics.get("rows_after", 0)))
        c2.metric("Missing Fixed", f"{metrics.get('missing_before', 0)} → {metrics.get('missing_after', 0)}")
        c3.metric("Duplicates Dropped", f"{metrics.get('duplicates_before', 0)} → {metrics.get('duplicates_after', 0)}")
        c4.metric("Status", "Cleaned ✅")

        st.success(f"Data saved to: `{eng.get('output_path')}`")
        
        with st.expander("📄 View Cleaning Log"):
            st.text(eng.get("report"))

    # --- DATA ANALYST ---
    with tab_ana:
        ana = res.get("data_analyst", {})
        artifacts = ana.get("eda_artifacts", {})
        
        # Insights
        st.subheader("💡 Key Insights")
        insights_path = artifacts.get("insights")
        if insights_path and os.path.exists(insights_path):
            with open(insights_path, "r") as f:
                data = json.load(f)
                txt = data.get("insights", "No insights found.")
                # handle list or string
                if isinstance(txt, list):
                    for item in txt:
                        st.markdown(f"- {item}")
                else:
                    st.markdown(txt)
        
        # Plots
        st.subheader("🖼️ Visualizations")
        viz_dir = ana.get("visualization_dir")
        if viz_dir:
            plots_dir = os.path.join(viz_dir, "plots")
            if os.path.exists(plots_dir):
                plots = [f for f in os.listdir(plots_dir) if f.endswith(".png")]
                if plots:
                    # Scrollable container workaround
                    with st.container(height=500):
                        cols = st.columns(2)
                        for i, p in enumerate(plots):
                            cols[i % 2].image(os.path.join(plots_dir, p), caption=p, use_container_width=True)
        
        # Report
        report_pdf = ana.get("visualization_report")
        if report_pdf and os.path.exists(report_pdf):
             with open(report_pdf, "rb") as f:
                st.download_button("📩 Download Full EDA Report (PDF)", f, file_name="eda_report.pdf")

    # --- DATA SCIENTIST ---
    with tab_sci:
        sci = res.get("data_scientist", {}).get("report", {}).get("report", {})
        if sci:
            problem = sci.get("problem_framing", {})
            model = sci.get("model_training", {})
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Problem Type", problem.get("problem_type", "Unknown").title())
            c2.metric("Best Model", model.get("best_model"))
            c3.metric("Score", f"{model.get('best_score', 0):.4f}")
            
            st.info(f"🧠 **Sanity Check**: {sci.get('sanity_check')}")
            
            st.subheader("⭐ Top Influential Features")
            feats = sci.get("top_features", [])
            st.bar_chart(pd.DataFrame(feats).set_index("feature"))
            
            with st.expander("� Read AI Explanation"):
                st.write(sci.get("llm_explanation"))
        else:
            st.warning("No model trained (maybe skipped?)")

    # ======================================================
    # 5️⃣ CONSULTANT CHAT
    # ======================================================
    st.markdown("---")
    st.markdown("### 5️⃣ Consultant Chat")
    
    col_role, col_chat = st.columns([1, 3])
    
    with col_role:
        role = st.radio("Talk to:", ["Data Engineer", "Data Analyst", "Data Scientist"])
    
    with col_chat:
        # Display history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        # Input
        if question := st.chat_input("Ask a question about your data..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner(f"{role} is thinking..."):
                    # Prepare Context
                    context_str = ""
                    plots_list = None
                    if role == "Data Engineer":
                        eng = st.session_state.results.get("data_engineer", {})
                        context_str = f"Cleaning Report: {eng.get('report')}\nMetrics: {eng.get('metrics')}"
                    elif role == "Data Analyst":
                        ana = st.session_state.results.get("data_analyst", {})
                        # Try to load insights text
                        try:
                            with open(ana['eda_artifacts']['insights'], 'r') as f:
                                context_str = f.read()
                        except:
                            context_str = "No specific insights file found."
                        
                        # Get plots list
                        viz_dir = ana.get("visualization_dir")
                        if viz_dir:
                            plots_dir = os.path.join(viz_dir, "plots")
                            if os.path.exists(plots_dir):
                                plots_list = [f for f in os.listdir(plots_dir) if f.endswith(".png")]

                    elif role == "Data Scientist":
                         sci = st.session_state.results.get("data_scientist", {}).get("report", {}).get("report", {})
                         context_str = json.dumps(sci, indent=2)

                    response = get_consultant_response(role, question, context_str, plots_list)
                    
                    # Parse image from response
                    import re
                    img_match = re.search(r"<<IMAGE: (.+?)>>", response)
                    clean_response = re.sub(r"<<IMAGE: .+?>>", "", response).strip()
                    
                    st.write(clean_response)
                    
                    if img_match:
                        img_name = img_match.group(1)
                        if role == "Data Analyst" and plots_list and img_name in plots_list:
                            img_path = os.path.join(plots_dir, img_name)
                            st.image(img_path, caption=img_name)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})

elif run_btn and not file_path:
    st.error("⚠️ Please upload a file first.")
