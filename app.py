import streamlit as st
import json
import os
from PIL import Image
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="SchemAI-Bench | Visual Task Factory",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ SchemAI-Bench: Visual Task Factory")
st.markdown("**Automated Synthetic Schematics & Verifiable Ground-Truth for Frontier Multimodal AI Evaluation**")
st.caption("Innodata AI Hackathon — Powered by Hack2skill | Track: Visual Task Factory")

# Paths
TASKS_PATH = os.path.join("dataset", "tasks.json")
RESULTS_PATH = os.path.join("dataset", "evaluation_results.json")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Visual Task Explorer", "📊 Benchmark Dashboard", "⚙️ Methodology & Architecture"])

# ----------------- TAB 1: TASK EXPLORER -----------------
with tab1:
    if os.path.exists(TASKS_PATH):
        with open(TASKS_PATH, "r", encoding="utf-8") as f:
            tasks = json.load(f)
            
        st.subheader(f"Generated Verifiable Tasks ({len(tasks)} Total)")
        
        task_ids = [f"{t['task_id']} — {t['question_type']}" for t in tasks]
        selected_idx = st.selectbox("Select a Generated Task to Inspect:", range(len(tasks)), format_func=lambda i: task_ids[i])
        
        selected_task = tasks[selected_idx]
        col_img, col_info = st.columns([1.3, 1])
        
        with col_img:
            img_rel = selected_task["image_filename"]
            img_full = os.path.join("dataset", img_rel)
            if os.path.exists(img_full):
                st.image(Image.open(img_full), caption=f"Synthetic Schematic: {img_rel}", use_container_width=True)
            else:
                st.error("Image file missing.")
                
        with col_info:
            st.markdown(f"### 📋 Task: `{selected_task['task_id']}`")
            st.info(f"**Question:**\n\n{selected_task['question_text']}")
            
            st.success(f"**Mathematical Ground Truth (Reference Answer):** `{selected_task['reference_answer']}`")
            st.markdown(f"**Format:** `{selected_task['answer_format']}` | **Category:** `{selected_task['question_type']}`")
            
            with st.expander("🔬 Independent Verification Evidence"):
                st.write(selected_task["visual_evidence_summary"])
                st.json(selected_task.get("graph_metadata", {}))
    else:
        st.warning("No tasks found. Run `python run_factory.py --generate` first!")

# ----------------- TAB 2: BENCHMARK RESULTS -----------------
with tab2:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
            
        m1, m2, m3 = st.columns(3)
        m1.metric("Evaluated Model", results.get("model_name", "N/A"))
        m2.metric("Overall Accuracy", f"{results.get('overall_accuracy_pct', 0)}%")
        m3.metric("Evaluated Tasks", results.get("total_tasks", 0))
        
        st.subheader("Performance Breakdown by Visual Question Type")
        breakdown = results.get("category_breakdown", {})
        if breakdown:
            df = pd.DataFrame([
                {"Category": k.replace("_", " ").title(), "Accuracy (%)": v["accuracy_pct"], "Correct": v["correct"], "Total": v["total"]}
                for k, v in breakdown.items()
            ])
            chart = alt.Chart(df).mark_bar(cornerRadius=6).encode(
                x=alt.X('Category:N', title="Visual Task Category"),
                y=alt.Y('Accuracy (%):Q', title="Accuracy (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color('Category:N', legend=None),
                tooltip=['Category', 'Accuracy (%)', 'Correct', 'Total']
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(df, use_container_width=True)
            
        st.markdown("### ⚠️ Key Insights from Multimodal Model Failure Modes")
        st.markdown("""
        1. **Spatial Hallucination in Wire Tracing:** Frontier multimodal models frequently hallucinate connections between components positioned in close 2D proximity even if no conductive trace connects them.
        2. **Dense Occlusion & Counting Errors:** Models struggle with counting repetitive glyphs (e.g. Resistors vs Inductors) when background engineering grid lines are present.
        3. **Failure on Multi-Hop Logic:** Multimodal reasoning breaks down when forced to traverse more than 2 orthogonal wire bends.
        """)
    else:
        st.info("Run `python run_factory.py --evaluate` to view benchmark metrics.")

# ----------------- TAB 3: METHODOLOGY -----------------
with tab3:
    st.markdown("""
    ### 🏗️ SchemAI-Bench Architecture
    
    ```mermaid
    flowchart LR
        subgraph Generation ["1. Task Factory"]
            A[Planar Graph Topology Engine] --> B[Orthogonal Wire & Symbol Renderer]
            B --> C[Synthetic Schematics]
            A --> D[Topological Ground-Truth Solver]
        end
        subgraph Verification ["2. Independent Ground Truth"]
            D --> E[Exact BFS/DFS & Component Sets]
            E --> F[Verifiable Reference Answers]
        end
        subgraph Evaluation ["3. Multimodal Benchmark"]
            C --> G[Multimodal Model API / Gemini]
            G --> H[Model Output]
            H --> I[Deterministic Verifier]
            F --> I
            I --> J[Accuracy & Hallucination Metrics]
        end
    ```
    
    ### Why Independent Verification Matters:
    - Many AI benchmarks use an LLM to evaluate another LLM. This causes **evaluator bias** and circular hallucinations.
    - **SchemAI-Bench** derives ground truth directly from the underlying mathematical graph structure **before** the image is rendered.
    - Zero human labeling errors, 100% mathematical repeatability.
    """)
