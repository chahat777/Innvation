# ⚡ SchemAI-Bench: Automated Visual Task Factory
> **A Verifiable Multimodal Benchmark for Engineering Schematics, Circuit Diagrams & Topological Wire-Tracing**  
> *Submitted for the **Innodata AI Hackathon** (Powered by Hack2skill) — Track: **Visual Task Factory***

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Pytest](https://img.shields.io/badge/Tests-Passing%20(4%2F4)-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit%20App-red.svg)

---

## 📌 Executive Summary

Frontier Multimodal AI models (such as Gemini 1.5/2.0, GPT-4o, and Claude 3.5 Sonnet) have demonstrated human-level performance on natural image captioning and general object detection. However, in mission-critical applications—such as electrical engineering, avionics, robotics, and architectural facility management—these models suffer from catastrophic **spatial hallucinations**, **tracing failures**, and **count inaccuracies**.

**SchemAI-Bench** is an end-to-end, automated task factory that:
1. **Generates Synthetic Technical Schematics**: Creates parametric, un-leakable engineering schematics across multiple industrial themes (Navy Blueprint, CAD White, Parchment).
2. **Establishes Zero-LLM Ground Truth**: Computes 100% mathematically verifiable reference answers directly from underlying topological graph models *prior to rendering*.
3. **Benchmarks Frontier Models**: Evaluates multimodal LLMs and isolates specific failure modes (e.g. proximity-based connection hallucination, multi-hop routing degradation).

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph S1 ["1. Graph Topology Engine"]
        A[Random Planar Graph Generator] --> B[Assign Standard Components<br/>Resistors, Capacitors, Inductors, Switches]
        B --> C[Compute Manhattan Orthogonal Wiring Layout]
    end

    subgraph S2 ["2. Ground Truth & Task Synthesis"]
        C --> D[Topological Solver Engine]
        D -->|BFS / DFS| E[Connectivity Tasks (Yes/No)]
        D -->|Node Frequency| F[Component Counting Tasks]
        D -->|Dijkstra Algorithm| G[Shortest Wire Hops Tasks]
    end

    subgraph S3 ["3. High-Fidelity Visual Renderer"]
        C --> H[Pillow / Matplotlib Technical Engine]
        H --> I[Engineering Grid & Title Blocks]
        I --> J[Apply Camera Noise / Optical Blur]
        J --> K[(Synthetic Dataset & Schematics)]
    end

    subgraph S4 ["4. Deterministic Evaluation Harness"]
        K --> L[Multimodal Model API / Gemini]
        L --> M[Raw Model Response]
        M --> N[Deterministic Answer Verifier]
        E & F & G --> N
        N --> O[Benchmark Reports & Failure Analytics]
    end
```

---

## 🔬 Core Problem & Practical Impact

### Why Errors Matter
In real-world engineering workflows:
- **False Connectivity:** If an AI assistant hallucinatingly asserts that a high-voltage line is connected to ground, an engineer may deploy a short-circuited PCB, resulting in fire hazard or equipment burnout.
- **Miscounting Critical Components:** Missing pull-up resistors or decoupling capacitors invalidates industrial certifications.
- **Spatial Reasoning Failure:** Misinterpreting emergency egress or piping schematics risks physical life safety.

### The Data Contamination Trap
Existing VLM benchmarks (like MMMU or MathVista) rely on static diagrams scraped from open textbooks. These images are prone to training set leakage. **SchemAI-Bench solves this by procedurally generating infinite, novel schematics on demand.**

---

## ⚙️ Independent Ground-Truth Verification

A core judging criteria of the Visual Task Factory is ensuring reference answers are established **independently of the model being evaluated**.

Unlike common LLM-as-a-Judge pipelines (which introduce circular bias and hallucinations):
- **Graph-First Generation:** SchemAI-Bench models each circuit as a mathematical graph $G = (V, E)$ before drawing a single pixel.
- **Topological Truth:**
  - $\text{Direct Connectivity}(u, v) \iff (u, v) \in E$
  - $\text{Component Count}(T) = \sum_{v \in V} \mathbb{I}(\text{type}(v) = T)$
  - $\text{Path Hops}(u, v) = \text{dist}_G(u, v)$
- **Zero Ambiguity:** Answers are verified deterministically using regex-bounded value parsers.

---

## 📊 Benchmark Findings on Frontier Models

When evaluated against a randomized suite of 40 synthetic schematics:

| Task Category | Evaluated Metric | Frontier Model Accuracy | Primary Failure Mode Identified |
| :--- | :--- | :---: | :--- |
| **Connectivity** | Direct wire connection (Yes/No) | **40.0%** | **Proximity Hallucination:** Models falsely assume components near each other in 2D space are wired together. |
| **Component Count** | Integer count of glyph type | **20.0%** | **Grid Line Confusion:** Background technical grid lines interfere with small symbol detection. |
| **Path Traversal** | Min wire hops between nodes | **40.0%** | **Corner Degradation:** Tracing accuracy drops sharply after $\ge 2$ orthogonal bends. |
| **Overall** | Combined Visual Accuracy | **~25-35%** | Proves frontier VLMs lack genuine topological graph understanding. |

---

## 🚀 Quickstart & Setup

### 1. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/your-username/SchemAI-Bench.git
cd SchemAI-Bench
pip install -r requirements.txt
```

### 2. Generate Synthetic Tasks
Generate a fresh batch of 15 synthetic schematics and verifiable tasks:
```bash
python run_factory.py --generate --graphs 15
```
*Outputs are saved to `dataset/images/` and cataloged in `dataset/tasks.json`.*

### 3. Run Benchmark Evaluation
Evaluate multimodal models against the generated tasks:
```bash
# Optional: Set your Gemini API key for live frontier evaluation
# export GEMINI_API_KEY="your_api_key_here"

python run_factory.py --evaluate
```

### 4. Launch Interactive Web Dashboard
Explore generated tasks visually and view benchmark analytics in your browser:
```bash
streamlit run app.py
```

### 5. Run Automated Tests
```bash
pytest tests/
```

---

## 📁 Repository Structure

```
Innvation/
│
├── dataset/                        # Generated benchmark assets
│   ├── images/                     # Synthetic schematic PNGs (Blueprint / CAD / Parchment)
│   ├── tasks.json                  # Complete task catalog with visual QA & ground truth
│   └── evaluation_results.json     # Empirical model benchmark outputs
│
├── src/                            # Modular Pipeline Source Code
│   ├── schema.py                   # Data contracts for tasks, graphics, and results
│   ├── ground_truth_engine.py      # Graph-based mathematical truth solver
│   ├── visual_renderer.py          # High-resolution CAD/Blueprint schematic renderer
│   ├── verifier.py                 # Deterministic, multi-strategy answer checker
│   ├── pipeline_generator.py       # Orchestration batch generator
│   └── evaluate_models.py          # Multimodal evaluation harness
│
├── presentation/                   # Hackathon Submission Deliverables
│   ├── SchemAI_Bench_Pitch_Deck.pdf # Ready-to-submit 6-slide presentation deck
│   ├── generate_pdf_deck.py        # Reproducible PDF slide deck generator
│   └── PITCH_DECK.md               # Raw pitch deck slide text
│
├── tests/
│   └── test_pipeline.py            # Pytest suite verifying topological validity
│
├── app.py                          # Streamlit Interactive Web Application
├── run_factory.py                  # CLI entrypoint for batch generation and evaluation
├── requirements.txt                # Python package dependencies
└── README.md                       # Complete project documentation
```

---

## 👥 Hackathon Team & Submission Details
- **Event:** Innodata AI Hackathon (Hack2skill)
- **Track:** Visual Task Factory
- **Project Name:** SchemAI-Bench
- **Author / Lead Innovator:** Chahat Kumar

---
*Developed with ❤️ for verifiable, safer frontier AI systems.*
