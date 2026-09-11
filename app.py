"""BrandSupport AI — Trustworthy AI Customer Support Agent Dashboard (Streamlit)."""

import json
import sqlite3
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="BrandSupport AI — Trustworthy Support Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Obsidian & Aurora Emerald Theme (NO Blue)
st.markdown("""
<style>
    /* Background and global dark theme */
    .stApp {
        background-color: #07090e;
        color: #f8fafc;
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Headers & Typography */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Custom Neon Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background: #0d1117;
    }
    ::-webkit-scrollbar-thumb {
        background: #00f5a0;
        border-radius: 4px;
    }
    
    /* Metrics and cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(18, 22, 31, 0.85) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 1px solid rgba(0, 245, 160, 0.25);
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 16px;
    }
    
    .decision-badge-auto {
        background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
        color: #ecfdf5;
        border: 1px solid #00f5a0;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 0 20px rgba(0, 245, 160, 0.4);
    }
    
    .decision-badge-escalate {
        background: linear-gradient(135deg, #881337 0%, #9f1239 100%);
        color: #fff1f2;
        border: 1px solid #ff3366;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-block;
        box-shadow: 0 0 20px rgba(255, 51, 102, 0.5);
    }

    .evidence-card {
        background: #11151e;
        border-left: 4px solid #00f5a0;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #11151e;
        border-radius: 8px;
        color: #94a3b8;
        padding: 10px 22px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00f5a0 0%, #00d995 100%) !important;
        color: #07090e !important;
        border-color: #00f5a0 !important;
        font-weight: 800 !important;
        box-shadow: 0 4px 20px rgba(0, 245, 160, 0.35) !important;
    }
    
    /* Button Glow */
    .stButton>button {
        background: linear-gradient(135deg, #00f5a0 0%, #00c98f 100%) !important;
        color: #07090e !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 18px rgba(0, 245, 160, 0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 245, 160, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# Project imports
from src.config import (
    DEFAULT_BRAND,
    DB_PATH,
    FAISS_INDEX_PATH,
    SEMANTIC_CLASSIFIER_PATH,
    EVAL_RESULTS_PATH,
    INTENT_LABELS,
    INTENT_DESCRIPTIONS,
    SUPPORTED_BRANDS,
)
from src.pipeline import BrandSupportPipeline
from src.conversation_builder import get_conversations_from_db
from evaluation.failure_analyzer import CURATED_FAILURE_CASES


@st.cache_resource(show_spinner="Loading BrandSupport AI Pipeline...")
def load_cached_pipeline(brand: str):
    """Loads and caches pipeline instance."""
    try:
        return BrandSupportPipeline.load_from_artifacts(brand_name=brand)
    except Exception as e:
        st.warning(f"Could not load saved models ({e}). Initializing online pipeline.")
        return BrandSupportPipeline(brand_name=brand)


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield-check.png", width=64)
    st.title("BrandSupport AI")
    st.caption("Trustworthy AI Customer Support Agent • Aurora Obsidian Edition")
    st.divider()

    selected_brand = st.selectbox("Target Brand", SUPPORTED_BRANDS, index=0)
    
    st.markdown("### ⚙️ Escalation Guardrails")
    intent_thresh = st.slider("Intent Confidence Threshold", 0.40, 0.90, 0.65, 0.05)
    sim_thresh = st.slider("Retrieval Similarity Threshold", 0.40, 0.90, 0.60, 0.05)
    ground_thresh = st.slider("Grounding Score Threshold", 0.40, 0.90, 0.70, 0.05)

    st.divider()
    
    # DB Status check
    db_exists = DB_PATH.exists()
    db_count = len(get_conversations_from_db(db_path=DB_PATH)) if db_exists else 0
    st.markdown("### 📊 System Status")
    st.write(f"• **SQLite Database**: {'🟢 Connected' if db_exists else '🔴 Missing'}")
    st.write(f"• **Indexed Threads**: `{db_count:,}`")
    st.write(f"• **Vector Model**: `all-MiniLM-L6-v2`")
    st.write(f"• **Intent Classes**: `8 Supported`")

# Load pipeline instance
pipeline = load_cached_pipeline(selected_brand)
pipeline.escalation_engine.intent_threshold = intent_thresh
pipeline.escalation_engine.retrieval_threshold = sim_thresh
pipeline.escalation_engine.grounding_threshold = ground_thresh


# --- MAIN INTERFACE TABS ---
tab_live, tab_eval, tab_failures, tab_explorer = st.tabs([
    "⚡ Live Agent Playground",
    "📈 Evaluation & Benchmarks",
    "🔍 Failure Analysis & RCA",
    "🗃️ Conversation Explorer",
])

# ==============================================================================
# TAB 1: LIVE AGENT PLAYGROUND
# ==============================================================================
with tab_live:
    st.subheader("Interactive Support Agent Simulator")
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(0, 245, 160, 0.1) 0%, rgba(255, 184, 0, 0.1) 50%, rgba(255, 51, 102, 0.1) 100%); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #94a3b8; display: flex; justify-content: space-between; align-items: center;">
        <span><strong>Pipeline Flow:</strong> Customer Query ➔ Intent Classifier (8 Classes) ➔ Confidence Calibration ➔ FAISS Semantic Retrieval ➔ Grounded Synthesis ➔ Escalation Guardrails</span>
    </div>
    """, unsafe_allow_html=True)


    # Preset Quick-Fills
    st.markdown("**Quick Preset Scenarios:**")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    preset_query = ""
    if col_p1.button("🔋 Battery Degradation", use_container_width=True):
        preset_query = f"@{selected_brand} My iPhone 12 battery drains from 100% to 10% in under 2 hours. Battery health says 73%. Should I get it replaced?"
    if col_p2.button("🐛 iOS App Freeze", use_container_width=True):
        preset_query = f"@{selected_brand} Ever since updating to iOS 17.4, my Phone app freezes whenever I tap on the voicemail tab."
    if col_p3.button("💳 Duplicate Charge", use_container_width=True):
        preset_query = f"@{selected_brand} I was charged twice for my 200GB iCloud storage this morning. How do I request a refund for the duplicate charge?"
    if col_p4.button("🔥 Battery Fire Emergency", use_container_width=True):
        preset_query = f"@{selected_brand} My charger sparked and the battery exploded on my desk with smoke! I will sue your company with my lawyer!"

    user_input = st.text_area(
        "Enter Customer Message / Tweet:",
        value=preset_query if preset_query else f"@{selected_brand} My iPhone battery health dropped to 72% and shuts off randomly.",
        height=100,
        placeholder="Type a customer query...",
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_btn = st.button("🚀 Process Message", type="primary", use_container_width=True)

    if run_btn or preset_query:
        with st.spinner("Analyzing intent, searching historical index, and verifying evidence..."):
            result = pipeline.process_query(user_input, brand_name=selected_brand)

        st.divider()

        # Row 1: Decision Banner & Escalation Routing
        col_dec, col_route = st.columns([1, 1])
        with col_dec:
            st.markdown("#### Routing Decision")
            if result["decision"] == "AUTO_HANDLE":
                st.markdown('<div class="decision-badge-auto">✅ AUTO_HANDLE — Verified Autonomous Reply</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="decision-badge-escalate">🚨 ESCALATE_TO_HUMAN — Human Review Required</div>', unsafe_allow_html=True)

            st.markdown(f"**Risk Level:** `{result['risk_level']}` | **Aggregate Confidence:** `{result['aggregate_confidence']:.2f}`")

        with col_route:
            st.markdown("#### Routing Queue & Trigger Rationale")
            st.info(f"**Target Department:** `{result['recommended_queue']}`")
            for r in result["escalation_reasons"]:
                st.write(f"• {r}")

        st.divider()

        # Row 2: Generated Reply & Grounding Verification
        col_rep, col_intent = st.columns([1.2, 0.8])

        with col_rep:
            st.markdown("#### 🤖 Grounded Support Response")
            st.markdown(f"""
            <div style="background: #11151e; border: 1px solid #00f5a0; border-radius: 8px; padding: 16px; font-size: 1.05rem; box-shadow: 0 0 15px rgba(0, 245, 160, 0.15);">
                {result['generated_reply']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Grounding Faithfulness Score:** `{result['grounding_score']:.2f} / 1.00` (Mode: `{result['generation_mode']}`)")
            st.progress(min(1.0, max(0.0, result["grounding_score"])))

        with col_intent:
            st.markdown("#### 🎯 Intent Classification")
            st.markdown(f"**Predicted Intent:** `{result['predicted_intent']}`")
            st.caption(result["intent_description"])
            st.markdown(f"**Confidence:** `{result['intent_confidence']:.2%}` | **Margin:** `{result['intent_margin']:.2f}`")
            st.progress(result["intent_confidence"])

        # Row 3: Retrieved Evidence Breakdown
        st.markdown("#### 📚 Retrieved Historical Evidence (FAISS Semantic Search)")
        if result["retrieved_evidence"]:
            for idx, ev in enumerate(result["retrieved_evidence"]):
                with st.expander(f"Historical Case #{idx+1} — Intent: {ev.get('intent')} (Similarity: {ev.get('similarity_score'):.2f})", expanded=(idx == 0)):
                    st.write(f"**Customer Inbound Query:** {ev.get('customer_query')}")
                    st.write(f"**Official Brand Resolution:** {ev.get('brand_reply')}")
                    st.caption(f"Conversation ID: `{ev.get('conversation_id')}` | Turns: {ev.get('num_turns')}")
        else:
            st.warning("No matching historical evidence found in database.")


# ==============================================================================
# TAB 2: EVALUATION & BENCHMARKS
# ==============================================================================
with tab_eval:
    st.subheader("Model Benchmark Evaluation & Quality Audit")

    if EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        st.markdown(f"**Evaluation Benchmark Dataset Size:** `N = {eval_data.get('dataset_size', 200)}` human-reviewed instances")

        # 1. Models Comparison Cards
        col_m1, col_m2, col_m3 = st.columns(3)
        comp = eval_data.get("models_comparison", {})
        
        with col_m1:
            maj = comp.get("Majority_Class_Baseline", {})
            st.markdown("""
            <div class="metric-card">
                <h4>1. Majority Baseline</h4>
                <h2 style="color: #94a3b8;">{:.2%}</h2>
                <p>Accuracy (Prior distribution)</p>
                <small>Macro F1: {:.4f}</small>
            </div>
            """.format(maj.get("accuracy", 0), maj.get("macro_f1", 0)), unsafe_allow_html=True)

        with col_m2:
            tfidf = comp.get("TFIDF_LogisticRegression_Baseline", {})
            st.markdown("""
            <div class="metric-card" style="border-color: rgba(255, 184, 0, 0.4);">
                <h4>2. TF-IDF + Logistic Reg</h4>
                <h2 style="color: #ffb800;">{:.2%}</h2>
                <p>Accuracy (N-gram baseline)</p>
                <small>Macro F1: {:.4f}</small>
            </div>
            """.format(tfidf.get("accuracy", 0), tfidf.get("macro_f1", 0)), unsafe_allow_html=True)

        with col_m3:
            sem = comp.get("BrandSupport_Semantic_Pipeline", {})
            st.markdown("""
            <div class="metric-card" style="border-color: #00f5a0; box-shadow: 0 0 25px rgba(0, 245, 160, 0.25);">
                <h4>3. Semantic Pipeline (Primary)</h4>
                <h2 style="color: #00f5a0;">{:.2%}</h2>
                <p>Accuracy (SentenceTransformers + FAISS)</p>
                <small>Macro F1: {:.4f}</small>
            </div>
            """.format(sem.get("accuracy", 0), sem.get("macro_f1", 0)), unsafe_allow_html=True)

        # 2. Performance Comparison Chart
        st.markdown("#### 📊 Comparative Classification Performance")
        chart_df = pd.DataFrame({
            "Model": ["Majority Baseline", "TF-IDF Baseline", "BrandSupport Semantic"],
            "Accuracy": [maj.get("accuracy", 0), tfidf.get("accuracy", 0), sem.get("accuracy", 0)],
            "Macro F1": [maj.get("macro_f1", 0), tfidf.get("macro_f1", 0), sem.get("macro_f1", 0)],
        }).set_index("Model")
        st.bar_chart(chart_df)

        st.divider()

        # 3. Retrieval & Escalation
        col_ret, col_esc = st.columns(2)
        with col_ret:
            st.markdown("#### 🎯 FAISS Retrieval Performance")
            ret_metrics = eval_data.get("retrieval_performance", {})
            r_c1, r_c2, r_c3 = st.columns(3)
            r_c1.metric("Recall@1", f"{ret_metrics.get('recall@1', 0):.2%}")
            r_c2.metric("Recall@3", f"{ret_metrics.get('recall@3', 0):.2%}")
            r_c3.metric("MRR", f"{ret_metrics.get('mrr', 0):.4f}")

        with col_esc:
            st.markdown("#### ⚖️ Escalation Decision Performance")
            esc_metrics = eval_data.get("escalation_performance", {})
            e_c1, e_c2, e_c3 = st.columns(3)
            e_c1.metric("Precision", f"{esc_metrics.get('precision', 0):.2%}")
            e_c2.metric("Recall", f"{esc_metrics.get('recall', 0):.2%}")
            e_c3.metric("F1-Score", f"{esc_metrics.get('f1_score', 0):.4f}")

        st.divider()

        # 4. LLM-as-a-Judge Rubric
        st.markdown("#### 🧑‍⚖️ LLM-as-a-Judge Quality Audit (Scale 1.0 – 5.0)")
        judge_scores = eval_data.get("llm_judge_scores", {})
        j_c1, j_c2, j_c3, j_c4, j_c5 = st.columns(5)
        j_c1.metric("Relevance", f"{judge_scores.get('mean_relevance', 0):.1f} / 5.0")
        j_c2.metric("Faithfulness", f"{judge_scores.get('mean_faithfulness', 0):.1f} / 5.0")
        j_c3.metric("Helpfulness", f"{judge_scores.get('mean_helpfulness', 0):.1f} / 5.0")
        j_c4.metric("Unsupported Claims", f"{judge_scores.get('unsupported_claims_rate', 0):.1%}")
        j_c5.metric("Escalation Correctness", f"{judge_scores.get('mean_escalation_correctness', 0):.1f} / 5.0")

    else:
        st.info("No benchmark results found yet. Run `python run_pipeline.py --stage eval` to generate evaluation metrics.")


# ==============================================================================
# TAB 3: FAILURE ANALYSIS & RCA
# ==============================================================================
with tab_failures:
    st.subheader("Production Failure Case Analysis & Root Cause Analysis (RCA)")
    st.markdown("Deep dive into the Top 5 edge cases, failure modes, and applied architectural mitigations.")

    for case in CURATED_FAILURE_CASES:
        with st.expander(f"[{case['case_id']}] {case['title']}", expanded=True):
            st.markdown(f"**Customer Query:** `{case['query']}`")
            
            c1, c2, c3 = st.columns(3)
            c1.write(f"• **True Intent:** `{case['ground_truth_intent']}`")
            c2.write(f"• **Predicted Intent:** `{case['predicted_intent']}` (Conf: {case.get('confidence', 0):.2f})")
            c3.write(f"• **Decision:** `{case['escalation_decision']}` vs Expected `{case['expected_escalation']}`")

            st.markdown(f"**Root Cause Analysis (RCA):**\n{case['root_cause_analysis']}")
            st.warning(f"**Impact:** {case['impact']}")
            st.success(f"**Applied Mitigation:** {case['engineering_mitigation']}")

    st.divider()

    st.markdown("### 💡 Critical Analysis: “What is misleading about my headline number?”")
    st.markdown("""
    > [!CAUTION]
    > **Why Headline 90%+ Accuracy Can Be Deceptive in Customer Support:**
    > 1. **Class Distribution Skew**: In standard customer support datasets, frequent classes (e.g. *Hardware Battery Issues* or *General Complaints*) dominate the distribution. A model can achieve high headline accuracy while failing miserably on rare, catastrophic classes (e.g. *Battery Fire Safety Hazards*).
    > 2. **Evaluation Leakage Shortcuts**: Standard random train/test splits allow near-duplicate conversational templates into both splits, artificially inflating Recall@1. Enforcing strict conversation-graph partitioning is critical.
    > 3. **The Asymmetric Cost of Errors**: A headline metric treating all errors equally masks dangerous failures. Auto-handling an emergency battery explosion is a severe safety liability, whereas misclassifying a feature how-to question is merely a minor inconvenience.
    """)


# ==============================================================================
# TAB 4: CONVERSATION EXPLORER
# ==============================================================================
with tab_explorer:
    st.subheader("Historical Conversation Database Explorer")
    st.markdown("Explore reconstructed multi-turn conversations stored in SQLite.")

    if DB_PATH.exists():
        selected_intent_filter = st.selectbox("Filter by Intent", ["All"] + INTENT_LABELS)
        search_term = st.text_input("Search Customer Query / Keyword", "")

        intent_param = None if selected_intent_filter == "All" else selected_intent_filter
        convs = get_conversations_from_db(
            brand_name=selected_brand,
            intent=intent_param,
            limit=50,
            db_path=DB_PATH,
        )

        if search_term:
            convs = [c for c in convs if search_term.lower() in c.get("cleaned_customer_query", "").lower()]

        st.write(f"Showing **{len(convs)}** conversations:")
        for c in convs:
            with st.expander(f"[{c['intent']}] {c['conversation_id']} — {c['cleaned_customer_query'][:70]}..."):
                st.write(f"**Customer Query:** {c['customer_query']}")
                st.write(f"**Brand Reply:** {c['brand_reply']}")
                st.caption(f"Brand: @{c['brand_name']} | Timestamp: {c['created_at']} | Total Turns: {c['num_turns']}")
    else:
        st.warning("SQLite database not found. Run `python run_pipeline.py --stage data` to build database.")
