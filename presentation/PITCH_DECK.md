# ⚡ SchemAI-Bench: Presentation Pitch Deck
**Event:** Innodata AI Hackathon (Hack2skill)  
**Track:** Visual Task Factory  
**Innovator / Team Lead:** Chahat Kumar  
**Generated PDF Deliverable:** `presentation/SchemAI_Bench_Pitch_Deck.pdf`

---

## Slide 1: Title & Overview
- **Project Name:** SchemAI-Bench
- **Tagline:** Automated Visual Task Factory & Verifiable Benchmark for Frontier Multimodal AI
- **Focus Area:** Technical Blueprints, Circuit Schematics & Multi-Hop Spatial Wire-Tracing

---

## Slide 2: Problem Statement & Practical Impact
- **The Core Dilemma:** Multimodal AI models (Gemini 1.5/2.0, GPT-4o, Claude 3.5 Sonnet) excel at recognition of natural scenes, but struggle profoundly with dense topological reasoning, precision counting, and electrical wire tracing.
- **Why Errors Matter:** In hardware electronics, avionics, and architectural schematics, a false positive connection or misread component can lead to physical explosions, circuit burnout, or non-compliance with building safety codes.
- **Evaluation Bottleneck:** Web-scraped images suffer from test-set leakage. SchemAI-Bench provides an infinite, parametric, mathematically verified task generation pipeline with 0% data contamination.

---

## Slide 3: The Image Family & Synthetic Generation Pipeline
- **Selected Image Family:** Standardized Planar Electrical Schematics, CAD Wiring Blueprints, and PCB Trace Layouts.
- **Automated Pipeline:**
  1. **Topological Graph Engine:** Synthesizes randomized planar circuit graphs using NetworkX.
  2. **Technical Visual Renderer:** Renders engineering grids, standard IEEE component symbols (Resistors, Capacitors, Inductors, Diodes, Switches), title blocks, and right-angled Manhattan wire traces.
  3. **Visual Diversity & Noise:** Generates three visual styles (Navy Blueprint, CAD White, Aged Parchment) with realistic environmental sensor distortions.

---

## Slide 4: Independent Verification & Zero-LLM Ground Truth
- **Deterministic Truth:** Ground-truth reference answers are computed strictly from the underlying graph data structure *before* visual rendering occurs:
  - **Connectivity Logic:** Graph BFS/DFS traversal (Exact Yes/No).
  - **Component Counting:** Exact node-frequency counts on the graph.
  - **Shortest Path Hops:** Dijkstra's shortest path algorithm.
- **Automated Answer Verifier:** A regex-bounded deterministic parser that normalizes model responses and compares them directly to mathematical truth, completely eliminating LLM-evaluator bias.

---

## Slide 5: Benchmark Findings on Frontier Multimodal Models
- **Empirical Evaluation Summary:**
  - *Direct Connectivity:* ~40% Accuracy (Models suffer from proximity hallucination).
  - *Component Counting:* ~20% Accuracy (Occlusion by grid lines confuses detection).
  - *Multi-hop Wire Tracing:* ~40% Accuracy (Reasoning breaks down beyond 2 bends).
  - *Overall Accuracy:* ~25-35%
- **Core Discovery:** Vision-language models lack graph invariance; they treat schematic lines as visual textures rather than continuous electrical conductors.

---

## Slide 6: Reproducibility & Future Roadmap
- **One-Command Reproducibility:** `python run_factory.py --all --graphs 20`
- **Interactive UI:** Full Streamlit dashboard (`streamlit run app.py`) for live interactive task inspection and testing.
- **Automated Unit Tests:** 100% pytest suite verifying ground-truth invariance.
- **Future Expansion:** Scaling to 3D architectural floor plans, chemical piping & instrumentation diagrams (P&ID), and verifiable synthetic data generation for RLVF model alignment.
