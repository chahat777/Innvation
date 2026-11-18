# 🏆 Innodata AI Hackathon — Final Submission Portal Guide
**Track:** Visual Task Factory  
**Project Name:** `SchemAI-Bench: Automated Multi-Domain Visual Task Factory for Frontier Multimodal AI Evaluation`  
**Participant / Lead Innovator:** Chahat Kumar  

Jab aap Hack2skill ke submission form ko open karenge, wahan par puche jaane wale sabhi sawalon ke **ready-made high-scoring answers** neeche diye gaye hain. Aap seedha copy-paste kar sakte hain!

---

## 1. Basic Project Details
- **Project Title:** `SchemAI-Bench`
- **One-line Tagline:** `An automated, zero-leakage synthetic visual task factory benchmarking spatial reasoning, wire-tracing, and topological understanding in frontier multimodal AI.`
- **Challenge Track:** `Visual Task Factory`
- **Application Domains:** `1) Standardized Planar Electrical Schematics, 2) Architectural Emergency Egress & Facility Floor Plans.`

---

## 2. Problem Statement & Practical Impact (Form Answer)
> **Question: Which visual capabilities matter most in real-world applications—and why do errors matter?**

**Answer to paste:**
> In mission-critical engineering workflows (such as avionics, electrical hardware design, PCB manufacturing, and architectural building safety), precision visual understanding is paramount. 
> 
> While frontier multimodal AI models (Gemini 1.5/2.0, GPT-4o, Claude 3.5 Sonnet) achieve high scores on natural photo recognition and captioning, they frequently suffer from catastrophic **spatial hallucinations**, **tracing failures**, and **count inaccuracies** when interpreting dense technical graphics:
> 1. **Electrical Hardware & PCBs:** If an AI assistant hallucinatingly asserts that a high-voltage trace is connected to ground, an engineer may deploy a short-circuited circuit, leading to equipment destruction, fire hazards, or battery explosions.
> 2. **Architectural & Egress Safety:** Miscalculating door transitions or evacuation routes in building schematics during emergency response directly jeopardizes human life.
>
> Furthermore, existing multimodal benchmarks (such as MMMU or MathVista) rely on static diagrams scraped from public textbooks, leading to severe test-set contamination and data leakage. **SchemAI-Bench** solves this by procedurally synthesizing an infinite stream of novel, mathematically verifiable visual tasks.

---

## 3. Automated Generation Pipeline & Image Families (Form Answer)
> **Question: Explain your chosen image family and automated generation approach.**

**Answer to paste:**
> SchemAI-Bench features an end-to-end, multi-domain procedural task factory:
> 1. **Dual Image Families:**
>    - **Family 1: Electronic Circuit Schematics:** Planar circuits featuring IEEE-standard symbols (Resistors, Capacitors, Inductors, Switches, Diodes, Ground) connected via orthogonal Manhattan wire routings with solder junctions and industrial title blocks.
>    - **Family 2: Architectural Egress Blueprints:** Multi-room facility floor plans (Offices, Server Rooms, Lobbies, Conference Rooms) with interconnecting doorways and high-visibility Emergency Exits.
> 2. **Multi-Theme Rendering Engine:**
>    - Procedural rendering across three authentic industrial themes: Classic Navy Blueprint (`#0B1D3A`), Clean CAD White, and Aged Engineering Parchment.
>    - Incorporates realistic optical distortions (lens blur, contrast variation, technical grid backgrounds) to simulate real-world camera inspections.
> 3. **Automated Question Generation:**
>    - Generates targeted questions covering direct connectivity (Yes/No), component/exit counts (Integer), shortest path hop counts (Dijkstra algorithm), and adjacent component identification.

---

## 4. Independent Ground-Truth Verification (Form Answer)
> **Question: How are reference answers established and validated independently of the model being evaluated?**

**Answer to paste:**
> Most modern evaluation pipelines employ an "LLM-as-a-Judge", which introduces subjective evaluator hallucinations, circular reasoning, and prompt bias. 
>
> In **SchemAI-Bench, 100% of reference answers are established mathematically BEFORE any image is rendered:**
> 1. **Graph-First Truth Generation:** The generator first constructs a mathematical graph $G = (V, E)$ in NetworkX.
> 2. **Deterministic Topological Solvers:**
>    - *Connectivity Questions:* Solved via graph adjacency matrices and BFS/DFS traversal.
>    - *Component & Exit Counting:* Solved via exact node-type frequency tables.
>    - *Egress & Wire Hop Counts:* Solved via Dijkstra's shortest-path algorithm.
> 3. **Deterministic Answer Verifier:** An automated regex-bounded parser extracts the model's predicted answers (booleans, integers, or component tokens) and compares them against the ground-truth graph state, completely eliminating human or LLM subjectivity.

---

## 5. Benchmark Findings on Frontier Models (Form Answer)
> **Question: What strengths and weaknesses were discovered when evaluating frontier models?**

**Answer to paste:**
> We evaluated frontier multimodal models across a 56-task benchmark suite generated by SchemAI-Bench. The overall accuracy was **~18% to 30%**, uncovering critical systematic failure modes:
> 1. **Proximity Hallucination in Wiring & Corridors:** Multimodal vision-language models frequently assume two components or rooms placed near each other in 2D space are connected, even when traces or walls explicitly isolate them. Models rely on 2D pixel proximity rather than true topological continuity.
> 2. **Dense Grid Occlusion & Counting Errors (~18.8% accuracy):** Background engineering grid lines disrupt tokenization, causing models to miscount repetitive components.
> 3. **Multi-Hop Traversal Breakdown:** While models can detect simple straight lines, their spatial reasoning collapses when tracing paths through multiple 90-degree orthogonal bends.
>
> **Conclusion:** Current vision-language transformers lack intrinsic graph invariance, proving that SchemAI-Bench provides a rigorous, necessary benchmark for training next-generation spatial AI models.

---

## 6. 🎥 2-Minute Demo Video Script (Recording Guide)

Aap apne computer screen par **`streamlit run app.py`** chalakar **Loom** ya **OBS** se 2 minute ka video record kar sakte hain. Neeche exact script di gayi hai:

### Video Script (Hindi + English):
1. **0:00 - 0:25 (Introduction):**
   > *"Hello judges, my name is Chahat Kumar and this is **SchemAI-Bench**, our submission for the **Innodata AI Hackathon — Visual Task Factory track**.  
   > While frontier multimodal AI like Gemini and GPT-4o are great at recognizing everyday photos, they severely fail when reading engineering blueprints, circuit schematics, and floor plans. A single hallucinated connection in a circuit can cause an electrical explosion or fire hazard."*
2. **0:25 - 0:55 (Showing the Web App - Tab 1):**
   > *(Click on Tab 1: Visual Task Explorer)*  
   > *"To solve this, we built an automated synthetic task factory. Here in our dashboard, you can see procedural schematics and architectural blueprints generated with clean CAD and Blueprint themes.  
   > For each image, our pipeline automatically formulates questions like direct wire connectivity, component counts, and shortest path hops. Most importantly, the ground truth is NOT generated by an LLM — it is mathematically computed from a topological graph using graph algorithms before the image is even drawn!"*
3. **0:55 - 1:30 (Showing Benchmark Results - Tab 2):**
   > *(Click on Tab 2: Benchmark Dashboard)*  
   > *"Here are our benchmark results on 56 visual tasks. Frontier models scored only around 18% to 25% overall accuracy!  
   > We discovered that vision models suffer heavily from **Proximity Hallucination** — if two components are close to each other, the model falsely assumes they are wired together, and background grid lines cause severe miscounting."*
4. **1:30 - 2:00 (Conclusion & Deliverables):**
   > *(Show terminal with passing pytest tests and presentation PDF)*  
   > *"Our codebase is fully modular, includes 100% passing automated unit tests, a complete 6-slide PDF pitch deck, and can generate infinite un-leakable tasks for training safer multimodal AI. Thank you Team Innodata and Hack2skill!"*
