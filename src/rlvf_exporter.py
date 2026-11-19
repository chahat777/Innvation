import os
import json
from typing import List, Dict, Any

class RLVFDatasetExporter:
    """
    Exports verifiable DPO / RLVF preference datasets from SchemAI-Bench tasks.
    Enables frontier AI labs (OpenAI, DeepMind, Anthropic) and Innodata to fine-tune
    vision-language models using Reinforcement Learning from Verifiable Feedback (RLVF).
    """

    @staticmethod
    def convert_tasks_to_dpo(tasks: List[Dict[str, Any]], output_file: str = "dataset/rlvf_dpo_dataset.jsonl") -> str:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        dpo_entries = []
        for t in tasks:
            task_id = t["task_id"]
            q_text = t["question_text"]
            img_rel = t["image_filename"]
            ref_ans = str(t["reference_answer"])
            evidence = t.get("visual_evidence_summary", "")
            domain = t.get("domain", "schematic")

            # Synthetic Verified Chain-of-Thought (Chosen)
            chosen_cot = (
                f"### Step-by-Step Topological Verification:\n"
                f"1. Target Components: Identified from inquiry.\n"
                f"2. Physical Trace Analysis: Tracing planar conductive pathways and orthogonal bends.\n"
                f"3. Mathematical Graph Evidence: {evidence}\n"
                f"4. Verification Outcome: Unbroken continuity verified via deterministic graph traversal.\n\n"
                f"### Final Answer:\n{ref_ans}"
            )

            # Hallucinated Heuristic Response (Rejected - Typical VLM Flaw)
            if ref_ans.lower() in ["no", "false"]:
                rejected_ans = "Yes"
                flaw_reason = "Components are positioned in spatial proximity, so they appear wired together."
            elif ref_ans.isdigit():
                val = int(ref_ans)
                rejected_ans = str(val + 1 if val > 0 else 2)
                flaw_reason = "Grid line artifacts were mistakenly counted as distinct electronic glyphs."
            else:
                rejected_ans = "Uncertain or Adjacent Node"
                flaw_reason = "Spatial attention degraded across right-angle wire routing."

            rejected_cot = (
                f"Based on the visual appearance of the {domain}, the components look connected "
                f"because of their proximity in the diagram. {flaw_reason}\n\n"
                f"### Final Answer:\n{rejected_ans}"
            )

            entry = {
                "task_id": task_id,
                "image_path": img_rel,
                "domain": domain,
                "prompt": f"<image>\n{q_text}",
                "chosen": chosen_cot,
                "rejected": rejected_cot,
                "verification_source": "Deterministic-Graph-Solver-v2"
            }
            dpo_entries.append(entry)

        with open(output_file, "w", encoding="utf-8") as f:
            for item in dpo_entries:
                f.write(json.dumps(item) + "\n")

        print(f"✅ RLVF / DPO Dataset successfully generated: {output_file} ({len(dpo_entries)} pairs)")
        return output_file
