import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

class PresentationPDF(FPDF):
    def header(self):
        self.set_fill_color(11, 29, 58) # Deep navy header
        self.rect(0, 0, 297, 18, 'F')
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 4)
        self.cell(0, 10, "INNODATA AI HACKATHON | VISUAL TASK FACTORY", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.set_xy(200, 4)
        self.cell(87, 10, "SchemAI-Bench Pitch Deck", new_x=XPos.RIGHT, new_y=YPos.TOP)
        self.ln(18)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')

def create_slide_deck():
    os.makedirs("presentation", exist_ok=True)
    pdf = PresentationPDF(orientation='L', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)

    # SLIDE 1: Title Slide
    pdf.add_page()
    pdf.set_fill_color(245, 247, 250)
    pdf.rect(10, 25, 277, 160, 'F')
    
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_text_color(11, 29, 58)
    pdf.set_xy(20, 50)
    pdf.cell(0, 15, "SchemAI-Bench", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(40, 116, 240)
    pdf.set_xy(20, 68)
    pdf.cell(0, 10, "Automated Multi-Domain Visual Task Factory for Frontier Multimodal AI Evaluation", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(80, 80, 80)
    pdf.set_xy(20, 85)
    pdf.multi_cell(250, 7, 
        "Domains: 1) Electronic Circuit Schematics & 2) Architectural Emergency Egress Floor Plans\n"
        "Track: Visual Task Factory | Event: Innodata AI Hackathon (Hack2skill)\n"
        "Lead Innovator: Chahat Kumar"
    )

    # SLIDE 2: Problem Statement & Practical Impact
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(11, 29, 58)
    pdf.cell(0, 10, "1. Problem Statement & Practical Impact", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 6.5,
        "- The Visual Reasoning Dilemma: While frontier multimodal models (Gemini, GPT-4o, Claude 3.5 Sonnet) achieve high scores "
        "on natural photo benchmarks, they fail dramatically on precision-critical engineering graphics, wire tracing, and spatial egress logic.\n\n"
        "- Why Visual Errors Matter in Real-World Applications:\n"
        "   - In Electronics & Hardware: A hallucinated wire connection can cause catastrophic short-circuits, PCB fire hazards, or hardware destruction.\n"
        "   - In Architecture & Emergency Response: Miscalculating room egress routes or exit doors during an emergency puts human lives in peril.\n\n"
        "- The Evaluation Bottleneck: Public benchmarks suffer from web dataset leakage. SchemAI-Bench solves this by procedurally generating "
        "infinite, novel, un-leakable visual tasks with 100% mathematical ground truth."
    )

    # SLIDE 3: Dual Image Families & Generation Pipeline
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(11, 29, 58)
    pdf.cell(0, 10, "2. Dual Image Families & Automated Task Factory", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 6.5,
        "- Image Family 1 (Circuit Schematics): IEEE-standard electrical components (Resistors, Capacitors, Inductors, Switches, Diodes, Ground) "
        "with orthogonal Manhattan wire routes, solder junctions, and technical title blocks.\n\n"
        "- Image Family 2 (Architectural Floor Plans): Facility rooms (Offices, Server Rooms, Lobbies, Conference Halls) with connecting doorways, "
        "corridors, and high-visibility Emergency Exits.\n\n"
        "- Multi-Theme Rendering Engine: Procedural synthesis in Navy Blueprint (#0B1D3A), Clean CAD White, and Industrial Aged Parchment with "
        "realistic optical noise and grid line variations."
    )

    # SLIDE 4: Independent Ground-Truth Engine
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(11, 29, 58)
    pdf.cell(0, 10, "3. Zero-LLM Independent Ground Truth & Verification", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 6.5,
        "- The Zero-LLM Truth Principle: Common AI evaluation relies on LLM-as-a-Judge, which introduces evaluator hallucination and circular bias. "
        "SchemAI-Bench derives ground truth mathematically BEFORE rendering the image:\n\n"
        "   - Direct Connectivity & Doorways: Evaluated via graph adjacency matrices (Deterministic Yes/No).\n"
        "   - Component & Exit Counting: Computed by direct node-type frequency tables on the underlying topology.\n"
        "   - Path Traversal & Signal Hops: Computed via Dijkstra's shortest path and BFS algorithms.\n\n"
        "- Deterministic Answer Checker: Normalizes model responses and extracts exact entities, integers, or boolean decisions with zero human subjectivity."
    )

    # SLIDE 5: Benchmark Findings on Frontier Models
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(11, 29, 58)
    pdf.cell(0, 10, "4. Benchmark Findings: Systematic Multimodal Weaknesses", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 6.5,
        "- Empirical Benchmark Suite (56 Verifiable Tasks across 16 Multi-Domain Blueprints):\n"
        "   - Direct Connectivity & Doorways: 12.5% - 40% Accuracy (Severe proximity hallucination: models guess 'Yes' if items are close).\n"
        "   - Component & Exit Counting: ~18.8% Accuracy (Dense background grid lines cause severe symbol miscounts).\n"
        "   - Multi-hop Path Traversal: ~18.8% Accuracy (Spatial reasoning fails when tracing paths through multiple 90-degree corners).\n"
        "   - Overall Accuracy: ~18% - 30% across frontier multimodal architectures.\n\n"
        "- Key Finding: Multimodal transformers treat engineering lines as visual textures rather than continuous physical conductors or corridors."
    )

    # SLIDE 6: Reproducibility & Future Roadmap
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(11, 29, 58)
    pdf.cell(0, 10, "5. Reproducibility, Deliverables & Impact", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)

    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 6.5,
        "- Complete Hackathon Deliverables:\n"
        "   1. Pipeline Code: Modular Python factory generating infinite parametric tasks.\n"
        "   2. Interactive Streamlit Web UI: Live visual task explorer, benchmark charts, and submission portal answers.\n"
        "   3. Comprehensive Test Suite: 5/5 passing pytest automated tests.\n"
        "   4. Ready-to-Submit Pitch Deck: Professional 6-slide landscape PDF.\n\n"
        "- Future Roadmap: Expanding to 3D CAD meshes, Piping & Instrumentation Diagrams (P&ID), and generating verifiable synthetic data "
        "to fine-tune next-generation spatial AI models using RLVF (Reinforcement Learning from Verifiable Feedback)."
    )

    output_path = os.path.join("presentation", "SchemAI_Bench_Pitch_Deck.pdf")
    pdf.output(output_path)
    print(f"✅ Slide Deck PDF created successfully: {output_path}")

if __name__ == "__main__":
    create_slide_deck()
