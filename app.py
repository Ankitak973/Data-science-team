import streamlit as st
import pandas as pd
import os
import time
import json
import shutil
import re
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

# Add Backend to path
import sys
sys.path.append(os.getcwd())

from Backend.agents.master_orchestrator import MasterOrchestrator

# ======================================================
# CONFIG & STYLING
# ======================================================
st.set_page_config(
    page_title="AVANA-Autonomous Visual Analytics & Normative Agents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Glassmorphic CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');

    :root {
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.1);
        --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        --text-main: #FAFAFA;
        --text-dim: #94a3b8;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a, #020617);
        font-family: 'Outfit', sans-serif;
    }

    /* Glass Cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Premium Buttons */
    .stButton>button {
        background: var(--accent-gradient);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        opacity: 0.9;
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
    }

    /* Metric Overhaul */
    [data-testid="stMetricValue"] {
        font-weight: 600;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Chat Styling */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }

    .chat-bubble {
        padding: 16px 20px;
        border-radius: 15px;
        font-size: 0.95rem;
        line-height: 1.5;
        max-width: 85%;
    }

    .user-bubble {
        align-self: flex-end;
        background: var(--accent-gradient);
        color: white;
        border-bottom-right-radius: 4px;
    }

    .bot-bubble {
        align-self: flex-start;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--glass-border);
        color: var(--text-main);
        border-bottom-left-radius: 4px;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--glass-border);
    }

    /* Progress bar gradient */
    .stProgress > div > div > div {
        background: var(--accent-gradient);
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
# ======================================================
# HELPER: CONSULTANT CHAT
# ======================================================
def get_consultant_response(role, question, context, plots_list=None):
    # return "⚠️ AI Consultant is currently disabled (Ollama removed)."
    
    llm = OllamaLLM(model="llama3.1")
    
    prompt = f"""
    You are an expert {role} in a data science team.
    
    Context about the current project/data:
    {context}
    
    Available visualizations (if any): {plots_list}
    
    User Question: {question}
    
    Answer the user's question based on the context provided.
    If you are the Data Analyst and the user asks for a specific chart that is in the available visualizations list, 
    you MUST mention the chart filename in this format at the end: <<IMAGE: filename.png>>.
    
    Keep the answer concise and professional.
    """
    
    return llm.invoke(prompt)

# ======================================================
# SIDEBAR CONFIGURATION
# ======================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103892.png", width=80)
    st.markdown("## ⚙️ Configuration")
    
    target_col = None
    max_plots = 30
    
    if st.session_state.df_preview is not None:
        cols = ["Auto-detect"] + list(st.session_state.df_preview.columns)
        choice = st.selectbox("🎯 Target Column", cols, index=0)
        if choice != "Auto-detect":
            target_col = choice
    else:
        st.selectbox("🎯 Target Column", ["Upload data first"], disabled=True)

    max_plots = st.slider("📊 Max Plots", min_value=10, max_value=50, value=30)
    
    st.markdown("---")
    st.markdown("### 🤖 Agent Status")
    st.success("Data Engineer: Active")
    st.success("Data Analyst: Active")
    st.success("Data Scientist: Active")

# ======================================================
# MAIN CONTENT
# ======================================================
st.markdown("<h1 style='text-align: center; color: white;'>🤖 AVANA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Autonomous Visual Analytics & Normative Agents</p>", unsafe_allow_html=True)

# 1️⃣ DATASET INTAKE (Glass Card)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("#### 📁 Step 1: Dataset Intake")
uploaded_file = st.file_uploader("Drop your CSV here", type="csv", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

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
    
    # Info Stats (Glass Metric Cards)
    st.markdown('<div style="display: flex; gap: 20px; margin-bottom: 20px;">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="glass-card" style="padding: 15px;">', unsafe_allow_html=True)
        st.metric("Total Records", f"{df.shape[0]:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card" style="padding: 15px;">', unsafe_allow_html=True)
        st.metric("Feature Count", df.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="glass-card" style="padding: 15px;">', unsafe_allow_html=True)
        st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Preview
    with st.expander("👀 View Dataset Preview"):
        st.dataframe(df.head(), use_container_width=True)

# ======================================================
# 3️⃣ EXECUTION
# ======================================================
st.markdown("### 🚀 Execution")
run_btn = st.button("🚀 Run AVANA Pipeline", use_container_width=True)

if run_btn and file_path:
    orchestrator = MasterOrchestrator()
    
    # Container for visible progress
    prog_container = st.container()
    
    with st.status("🚀 AVANA is working...", expanded=True) as status:
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
    st.markdown("<h2 style='text-align: center;'>📊 AVANA Results Center</h2>", unsafe_allow_html=True)
    
    tab_eng, tab_ana, tab_sci = st.tabs([
        "🧹 Data Engineering", "📊 Strategic Analysis", "🤖 Risk Science"
    ])
    
    # --- DATA ENGINEER ---
    with tab_eng:
        eng = res.get("data_engineer", {})
        metrics = eng.get("metrics", {})
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 🧹 Data Cleaning Integrity")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows Removed", (metrics.get("rows_before", 0) - metrics.get("rows_after", 0)))
        c2.metric("Missing Fixed", f"{metrics.get('missing_before', 0)} → {metrics.get('missing_after', 0)}")
        c3.metric("Duplicates Dropped", f"{metrics.get('duplicates_before', 0)} → {metrics.get('duplicates_after', 0)}")
        c4.metric("Status", "Validated ✅")
        st.markdown('</div>', unsafe_allow_html=True)

        st.info(f"📁 Output Path: `{eng.get('output_path')}`")
        
        with st.expander("📄 View Engineering Log"):
            st.code(eng.get("report"))

    # --- DATA ANALYST ---
    with tab_ana:
        ana = res.get("data_analyst", {})
        artifacts = ana.get("eda_artifacts", {})
        
        # Insights
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 💡 Strategic Insights")
        insights_path = artifacts.get("insights")
        if insights_path and os.path.exists(insights_path):
            with open(insights_path, "r") as f:
                data = json.load(f)
                txt = data.get("insights", "No insights found.")
                if isinstance(txt, list):
                    for item in txt:
                        st.markdown(f"🔹 {item}")
                else:
                    st.markdown(txt)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plots
        st.subheader("🖼️ Interactive Explorations")
        viz_dir = ana.get("visualization_dir")
        if viz_dir:
            plots_dir = os.path.join(viz_dir, "plots")
            if os.path.exists(plots_dir):
                plots = [f for f in os.listdir(plots_dir) if f.endswith(".png")]
                if plots:
                    with st.container(height=600):
                        cols = st.columns(2)
                        for i, p in enumerate(plots):
                            cols[i % 2].image(os.path.join(plots_dir, p), caption=p, use_container_width=True)
        
        # Report
        report_pdf = ana.get("visualization_report")
        if report_pdf and os.path.exists(report_pdf):
             with open(report_pdf, "rb") as f:
                st.download_button("📩 Download Executive PDF Report", f, file_name="strategic_eda_report.pdf")

    # --- DATA SCIENTIST ---
    with tab_sci:
        sci = res.get("data_scientist", {}).get("report", {}).get("report", {})
        if sci:
            problem = sci.get("problem_framing", {})
            model = sci.get("model_training", {})
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("##### 🧠 Risk Assessment Engine")
            c1, c2, c3 = st.columns(3)
            c1.metric("Problem Type", problem.get("problem_type", "Unknown").title())
            c2.metric("Best Model", model.get("best_model"))
            c3.metric("Score (AUC/R²)", f"{model.get('best_score', 0):.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.info(f"🔍 **Stability Audit**: {sci.get('sanity_check')}")
            
            st.subheader("⭐ Material Risk Drivers")
            feats = sci.get("top_features", [])
            st.bar_chart(pd.DataFrame(feats).set_index("feature"))
            
            with st.expander("📖 Read Strategic Narrative"):
                st.markdown(sci.get("llm_explanation"))
        else:
            st.warning("No model trained - strategic skipping.")

# ======================================================
# 5️⃣ CONSULTANT CHAT
# ======================================================
st.markdown("---")
st.markdown("<h2 style='text-align: center;'>💬 AI Intelligence Hub</h2>", unsafe_allow_html=True)

chat_col1, chat_col2 = st.columns([1, 4])

with chat_col1:
    role = st.radio("Agent Context:", ["Data Engineer", "Data Analyst", "Data Scientist"], label_visibility="visible")
    st.markdown("---")
    if st.button("🧹 Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()

with chat_col2:
    # Display history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        div_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        st.markdown(f'<div class="chat-bubble {div_class}">{msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
                
    # Input
    if question := st.chat_input("Query the Strategic Team..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.rerun()  # Rerun to show user message immediately

# Logic to handle response AFTER rerun
if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
    last_msg = st.session_state.chat_history[-1]["content"]
    
    with chat_col2:
        with st.spinner(f"Consulting {role}..."):
            # Prepare Context (logic identical to before)
            context_str = ""
            plots_list = None
            if role == "Data Engineer":
                eng = st.session_state.results.get("data_engineer", {})
                context_str = f"Cleaning Report: {eng.get('report')}\nMetrics: {eng.get('metrics')}"
            elif role == "Data Analyst":
                ana = st.session_state.results.get("data_analyst", {})
                try:
                    with open(ana['eda_artifacts']['insights'], 'r') as f:
                        context_str = f.read()
                except:
                    context_str = "No specific insights file found."
                viz_dir = ana.get("visualization_dir")
                if viz_dir:
                    plots_dir = os.path.join(viz_dir, "plots")
                    if os.path.exists(plots_dir):
                        plots_list = [f for f in os.listdir(plots_dir) if f.endswith(".png")]
            elif role == "Data Scientist":
                 sci = st.session_state.results.get("data_scientist", {}).get("report", {}).get("report", {})
                 context_str = json.dumps(sci, indent=2)

            response = get_consultant_response(role, last_msg, context_str, plots_list)
            
            # Parse image from response
            img_match = re.search(r"<<IMAGE: (.+?)>>", response)
            clean_response = re.sub(r"<<IMAGE: .+?>>", "", response).strip()
            
            st.session_state.chat_history.append({"role": "assistant", "content": clean_response})
            
            if img_match:
                img_name = img_match.group(1)
                # Note: Image handling logic preserved but might need streamlit-specific placement
                # We'll just store the text for now as chat bubbles are HTML based.
                # For images, we can add a separate display logic if needed.
            
            st.rerun()


elif run_btn and not file_path:
    st.error("⚠️ Please upload a file first.")
