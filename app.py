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

# Header
st.title("⚡ SchemAI-Bench: Visual Task Factory")
st.markdown("**Automated Multi-Domain Schematics & Verifiable Ground-Truth for Frontier Multimodal AI Evaluation**")
st.caption("Innodata AI Hackathon — Powered by Hack2skill | Track: Visual Task Factory | Innovator: Chahat Kumar")

# Paths
TASKS_PATH = os.path.join("dataset", "tasks.json")
RESULTS_PATH = os.path.join("dataset", "evaluation_results.json")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Visual Task Explorer", 
    "📊 Benchmark Dashboard", 
    "🏗️ Pipeline Architecture", 
    "📝 Submission Answers & Script"
])

# ----------------- TAB 1: TASK EXPLORER -----------------
with tab1:
    if os.path.exists(TASKS_PATH):
        with open(TASKS_PATH, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        domains = sorted(list(set(t.get("domain", "all") for t in tasks)))
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            selected_domain = st.selectbox("Filter by Domain:", ["All"] + domains)
            
        filtered_tasks = tasks if selected_domain == "All" else [t for t in tasks if t.get("domain") == selected_domain]
        
        with col_f2:
            st.write(f"Showing **{len(filtered_tasks)}** of **{len(tasks)}** generated verifiable tasks")

        task_labels = [f"[{t.get('domain', '').upper()}] {t['task_id']} — {t['question_type']}" for t in filtered_tasks]
        selected_idx = st.selectbox("Select a Generated Task to Inspect:", range(len(filtered_tasks)), format_func=lambda i: task_labels[i])
        
        selected_task = filtered_tasks[selected_idx]
        col_img, col_info = st.columns([1.3, 1])
        
        with col_img:
            img_rel = selected_task["image_filename"]
            img_full = os.path.join("dataset", img_rel)
            if os.path.exists(img_full):
                st.image(Image.open(img_full), caption=f"Synthetic Technical Blueprint: {img_rel}", use_container_width=True)
            else:
                st.error("Image file missing.")
                
        with col_info:
            st.markdown(f"### 📋 Task: `{selected_task['task_id']}`")
            st.markdown(f"**Domain:** `{selected_task.get('domain', 'N/A')}` | **Format:** `{selected_task['answer_format']}`")
            st.info(f"**Question Asked to Multimodal Model:**\n\n{selected_task['question_text']}")
            
            st.success(f"**Mathematical Ground Truth (Reference Answer):** `{selected_task['reference_answer']}`")
            
            with st.expander("🔬 Independent Topological Evidence"):
                st.write(selected_task["visual_evidence_summary"])
                st.json(selected_task.get("graph_metadata", {}))
    else:
        st.warning("No tasks found. Run `python run_factory.py --generate` first!")

# ----------------- TAB 2: BENCHMARK RESULTS -----------------
with tab2:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            results = json.load(f)
            
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Evaluated Model", results.get("model_name", "N/A"))
        m2.metric("Overall Accuracy", f"{results.get('overall_accuracy_pct', 0)}%")
        m3.metric("Evaluated Tasks", results.get("total_tasks", 0))
        m4.metric("Correct Tasks", results.get("total_correct", 0))
        
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
            
        st.markdown("### ⚠️ Key Insights: Why Frontier Multimodal Models Fail")
        st.markdown("""
        1. **Proximity Hallucination in Wiring & Corridors:** Frontier models assume two components or rooms placed near each other on a 2D plane are connected, even when wires or walls explicitly isolate them.
        2. **Dense Occlusion & Counting Errors:** Engineering grid lines and repeated glyphs (Resistors, Inductors, Doorways) induce counting hallucinations.
        3. **Failure on Multi-Hop Path Tracing:** While models can detect direct lines, their spatial reasoning collapses when tracing paths through multiple 90-degree orthogonal bends.
        """)
    else:
        st.info("Run `python run_factory.py --evaluate` to view benchmark metrics.")

# ----------------- TAB 3: METHODOLOGY -----------------
with tab3:
    st.markdown("""
    ### 🏗️ SchemAI-Bench System Architecture
    
    ```mermaid
    flowchart LR
        subgraph G ["1. Synthetic Task Factory"]
            A[Planar Graph Topology Engine] --> B[Technical Blueprints & CAD Renderer]
            B --> C[Synthetic Schematics & Floor Plans]
            A --> D[Topological Ground-Truth Solver]
        end
        subgraph V ["2. Zero-LLM Independent Ground Truth"]
            D --> E[Exact BFS/DFS & Component Sets]
            E --> F[100% Verifiable Reference Answers]
        end
        subgraph E2 ["3. Multimodal Benchmark Harness"]
            C --> G2[Multimodal Model API / Gemini]
            G2 --> H[Raw Model Predictions]
            H --> I[Deterministic Verifier]
            F --> I
            I --> J[Accuracy, Hallucination & Failure Analytics]
        end
    ```

    ### Why Reference Answers are Truly Independent:
    - Most existing AI benchmarks use an LLM-as-a-Judge, which introduces evaluator bias and circular hallucinations.
    - **SchemAI-Bench** derives ground truth directly from mathematical graph theory **before** the image is rendered.
    - Zero human labeling errors, 100% mathematical repeatability.
    """)

# ----------------- TAB 4: SUBMISSION ANSWERS & SCRIPT -----------------
with tab4:
    st.markdown("### 📋 Ready-to-Copy Portal Answers & Video Script")
    st.markdown("You can copy the answers below directly into the Hack2skill submission form!")
    
    with st.expander("📄 Click to View Portal Form Answers"):
        st.markdown("""
        **Project Title:** `SchemAI-Bench: Automated Visual Task Factory for Frontier Multimodal AI Evaluation`  
        **Tagline:** `A procedural, zero-leakage benchmark verifying topological wire-tracing, spatial logic, and counting in multimodal AI.`  
        **Chosen Track:** `Visual Task Factory`  
        **Image Families:** `1) Standardized Planar Electronic Schematics (CAD/Blueprints), 2) Architectural Facility & Emergency Egress Floor Plans.`  
        **Verification Method:** `Mathematical Graph Traversal (BFS/DFS, Dijkstra, Node Frequency) derived before rendering.`
        """)
