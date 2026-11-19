import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from src.schema import QuestionType
from src.verifier import AnswerVerifier


class RLVFDatasetExporter:
    """
    Exports verifiable DPO / RLVF preference datasets from SchemAI-Bench tasks.
    Enables frontier AI labs (OpenAI, DeepMind, Anthropic) and Innodata to fine-tune
    vision-language models using Reinforcement Learning from Verifiable Feedback (RLVF).
    
    Supports:
    1. Legacy Flat-String DPO Format (convert_tasks_to_dpo)
    2. Versioned Multimodal Conversational DPO Format (export_dpo_dataset)
    """

    @staticmethod
    def _is_safe_image_path(path: str) -> bool:
        """
        Guards against directory traversal and unsafe absolute filesystem paths.
        """
        if not path or not isinstance(path, str):
            return False
        norm = path.replace("\\", "/").strip()
        parts = [p for p in norm.split("/") if p]
        if ".." in parts:
            return False
        # Disallow absolute filesystem paths (Unix root or Windows drive letter)
        if norm.startswith("/") or re.match(r'^[a-zA-Z]:', norm):
            return False
        return True

    @staticmethod
    def _normalize_image_path(path: str) -> str:
        """
        Normalizes image paths to POSIX forward slashes, stripping leading './'.
        """
        if not path:
            return ""
        norm = str(path).replace("\\", "/").strip()
        if norm.startswith("./"):
            norm = norm[2:]
        return norm

    @classmethod
    def _generate_preference_pair(
        cls, task: Dict[str, Any]
    ) -> Optional[Tuple[str, str, bool, bool, QuestionType, Any, Any, Any, Any]]:
        """
        Generates verifier-backed chosen and rejected CoT responses for a task.
        Returns:
            (chosen_cot, rejected_cot, c_ok, r_ok, q_type, c_val, c_err, r_val, r_err)
        """
        ref_ans = task.get("reference_answer")
        q_type_str = task.get("question_type") or task.get("task_type") or ""
        domain = task.get("domain", "circuit_schematic")
        evidence = task.get("visual_evidence_summary", "")

        # Resolve QuestionType
        q_type = None
        if q_type_str:
            try:
                q_type = QuestionType(q_type_str)
            except (ValueError, KeyError):
                for qt in QuestionType:
                    if qt.value.lower() == str(q_type_str).lower() or qt.name.lower() == str(q_type_str).lower():
                        q_type = qt
                        break

        if q_type is None:
            if str(ref_ans).strip().lower() in ["yes", "no", "true", "false"]:
                q_type = QuestionType.CONNECTIVITY_EXISTS
            elif isinstance(ref_ans, int) or (isinstance(ref_ans, str) and str(ref_ans).strip().isdigit()):
                q_type = QuestionType.COMPONENT_COUNT
            elif isinstance(ref_ans, list):
                q_type = QuestionType.NEIGHBOR_IDENTIFICATION
            else:
                q_type = QuestionType.CONNECTIVITY_EXISTS

        # Determine chosen and rejected answers
        if q_type == QuestionType.CONNECTIVITY_EXISTS:
            is_ref_yes = str(ref_ans).strip().lower() in ["yes", "true"]
            chosen_ans = "Yes" if is_ref_yes else "No"
            rejected_ans = "No" if is_ref_yes else "Yes"
            if is_ref_yes:
                flaw_reason = "An orthogonal wire corner was misinterpreted as a disconnect, falsely concluding no electrical continuity exists."
            else:
                flaw_reason = "Components are positioned in spatial proximity, so they appear wired together despite lacking conductive connection."

        elif q_type in [QuestionType.COMPONENT_COUNT, QuestionType.SHORTEST_PATH_LENGTH]:
            try:
                val = int(ref_ans)
            except (ValueError, TypeError):
                nums = re.findall(r'\b\d+\b', str(ref_ans))
                val = int(nums[0]) if nums else 0

            chosen_ans = str(val)
            rejected_val = val + 1 if val >= 0 else 1
            rejected_ans = str(rejected_val)
            flaw_reason = "Grid line intersections and wire artifacts were mistakenly counted as distinct electronic glyphs."

        elif q_type == QuestionType.NEIGHBOR_IDENTIFICATION:
            if isinstance(ref_ans, list) and len(ref_ans) > 0:
                chosen_ans = str(ref_ans[0])
                valid_set_upper = [str(x).strip().upper() for x in ref_ans]
            else:
                chosen_ans = str(ref_ans)
                valid_set_upper = [str(ref_ans).strip().upper()]

            source_node = str(task.get("graph_metadata", {}).get("source", "")).strip().upper()
            nodes = task.get("graph_metadata", {}).get("serialized_graph", {}).get("nodes", [])
            if not nodes:
                nodes = task.get("graph_metadata", {}).get("nodes", [])

            candidates = []
            for n in nodes:
                lbl = str(n.get("label", "") if isinstance(n, dict) else n).strip().upper()
                if lbl and lbl not in valid_set_upper and lbl != source_node:
                    candidates.append(lbl)

            if candidates:
                rejected_ans = candidates[0]
            else:
                rejected_ans = "U99" if "U99" not in valid_set_upper else "NC1"

            flaw_reason = "Spatial attention degraded across right-angle wire routing, identifying a distant or unconnected component."

        else:
            chosen_ans = str(ref_ans)
            rejected_ans = f"NOT_{ref_ans}"
            flaw_reason = "Visual heuristic attention produced a hallucinated topological relationship."

        chosen_cot = (
            f"### Step-by-Step Topological Verification:\n"
            f"1. Target Components: Identified from inquiry.\n"
            f"2. Physical Trace Analysis: Tracing planar conductive pathways and orthogonal bends.\n"
            f"3. Mathematical Graph Evidence: {evidence}\n"
            f"4. Verification Outcome: Unbroken continuity verified via deterministic graph traversal.\n\n"
            f"### Final Answer:\n{chosen_ans}"
        )

        rejected_cot = (
            f"Based on the visual appearance of the {domain}, the components look connected "
            f"because of their proximity in the diagram. {flaw_reason}\n\n"
            f"### Final Answer:\n{rejected_ans}"
        )

        # Pre-flight independent verification
        c_ok, c_val, c_err = AnswerVerifier.verify(chosen_cot, ref_ans, q_type)
        r_ok, r_val, r_err = AnswerVerifier.verify(rejected_cot, ref_ans, q_type)

        return (chosen_cot, rejected_cot, c_ok, r_ok, q_type, c_val, c_err, r_val, r_err)

    @classmethod
    def convert_tasks_to_dpo(
        cls, tasks: List[Dict[str, Any]], output_file: str = "dataset/rlvf_dpo_dataset.jsonl"
    ) -> str:
        """
        Legacy export format: exports flat-string DPO pairs.
        Preserves backward compatibility with existing consumers and benchmark pipelines.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        dpo_entries = []
        for t in tasks:
            task_id = t.get("task_id", "")
            q_text = t.get("question_text", "")
            img_rel = t.get("image_filename", "")
            domain = t.get("domain", "schematic")

            if not cls._is_safe_image_path(img_rel):
                raise ValueError(f"Unsafe image path detected (path traversal): {img_rel}")

            norm_img_path = cls._normalize_image_path(img_rel)

            pair_data = cls._generate_preference_pair(t)
            if pair_data is None:
                continue

            chosen_cot, rejected_cot, c_ok, r_ok, _, _, _, _, _ = pair_data

            # Strict verification requirement: chosen must be True, rejected must be False
            if not c_ok or r_ok:
                continue

            entry = {
                "task_id": task_id,
                "image_path": norm_img_path,
                "domain": domain,
                "prompt": f"<image>\n{q_text}",
                "chosen": chosen_cot,
                "rejected": rejected_cot,
                "verification_source": "Deterministic-Graph-Solver-v2",
            }
            dpo_entries.append(entry)

        with open(output_file, "w", encoding="utf-8") as f:
            for item in dpo_entries:
                f.write(json.dumps(item) + "\n")

        print(f"✅ RLVF / DPO Dataset successfully generated: {output_file} ({len(dpo_entries)} pairs)")
        return output_file








    

    @classmethod
    def export_dpo_dataset(
        cls,
        tasks: List[Dict[str, Any]],
        output_file: str = "dataset/rlvf_dpo_conversational.jsonl",
        dataset_version: str = "2.0.0",
        split_ratio: Optional[float] = None,
        train_file: Optional[str] = None,
        test_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Multimodal Conversational DPO / RLVF exporter.
        Generates role/content message records compatible with vision-language post-training.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        total_source_tasks = len(tasks)
        verified_records = []
        excluded_tasks = []
        seen_ids = set()
        duplicate_ids_count = 0
        chosen_pass_count = 0
        rejected_fail_count = 0

        for t in tasks:
            task_id = t.get("task_id", "")
            if not task_id:
                excluded_tasks.append({"task_id": "", "reason": "MISSING_TASK_ID"})
                continue

            if task_id in seen_ids:
                duplicate_ids_count += 1
                excluded_tasks.append({"task_id": task_id, "reason": "DUPLICATE_TASK_ID"})
                continue
            seen_ids.add(task_id)

            img_rel = t.get("image_filename", "")
            if not cls._is_safe_image_path(img_rel):
                excluded_tasks.append({
                    "task_id": task_id,
                    "reason": f"PATH_TRAVERSAL_DETECTED: Unsafe image path '{img_rel}'",
                })
                continue

            norm_img_path = cls._normalize_image_path(img_rel)
            q_text = t.get("question_text", "")
            domain = t.get("domain", "circuit_schematic")

            pair_data = cls._generate_preference_pair(t)
            if pair_data is None:
                excluded_tasks.append({"task_id": task_id, "reason": "PAIR_GENERATION_FAILED"})
                continue

            chosen_cot, rejected_cot, c_ok, r_ok, q_type, c_val, c_err, r_val, r_err = pair_data

            if not c_ok:
                excluded_tasks.append({
                    "task_id": task_id,
                    "reason": f"CHOSEN_VERIFICATION_FAILED: {c_err}",
                })
                continue
            chosen_pass_count += 1

            if r_ok:
                excluded_tasks.append({
                    "task_id": task_id,
                    "reason": f"REJECTED_VERIFICATION_FAILED: Expected False but verifier passed with parsed value {r_val}",
                })
                continue
            rejected_fail_count += 1

            task_type_str = q_type.value if hasattr(q_type, "value") else str(q_type)

            record = {
                "id": task_id,
                "domain": domain,
                "task_type": task_type_str,
                "images": [norm_img_path],
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": q_text},
                        ],
                    }
                ],
                "chosen": [
                    {
                        "role": "assistant",
                        "content": chosen_cot,
                    }
                ],
                "rejected": [
                    {
                        "role": "assistant",
                        "content": rejected_cot,
                    }
                ],
                "chosen_text": chosen_cot,
                "rejected_text": rejected_cot,
                "metadata": {
                    "dataset_version": dataset_version,
                    "verifier_version": "AnswerVerifier-v2",
                    "question_type": task_type_str,
                    "answer_format": t.get("answer_format", ""),
                    "reference_answer": t.get("reference_answer"),
                    "visual_evidence_summary": t.get("visual_evidence_summary", ""),
                    "verification_source": "Deterministic-Graph-Solver-v2",
                    "chosen_verification": {
                        "is_correct": c_ok,
                        "parsed": str(c_val),
                        "error": c_err,
                    },
                    "rejected_verification": {
                        "is_correct": r_ok,
                        "parsed": str(r_val),
                        "error": r_err,
                    },
                },
            }
            verified_records.append(record)

        # Write unified output file
        with open(output_file, "w", encoding="utf-8") as f:
            for rec in verified_records:
                f.write(json.dumps(rec) + "\n")

        # Handle train / test split if requested
        actual_train_file = None
        actual_test_file = None
        train_count = len(verified_records)
        test_count = 0

        if split_ratio is not None and 0.0 < split_ratio < 1.0:
            base, ext = os.path.splitext(output_file)
            actual_train_file = train_file or f"{base}_train{ext}"
            actual_test_file = test_file or f"{base}_test{ext}"

            os.makedirs(os.path.dirname(os.path.abspath(actual_train_file)), exist_ok=True)
            os.makedirs(os.path.dirname(os.path.abspath(actual_test_file)), exist_ok=True)

            split_idx = int(len(verified_records) * split_ratio)
            train_records = verified_records[:split_idx]
            test_records = verified_records[split_idx:]
            train_count = len(train_records)
            test_count = len(test_records)

            with open(actual_train_file, "w", encoding="utf-8") as f:
                for rec in train_records:
                    f.write(json.dumps(rec) + "\n")

            with open(actual_test_file, "w", encoding="utf-8") as f:
                for rec in test_records:
                    f.write(json.dumps(rec) + "\n")

        report = {
            "total_source_tasks": total_source_tasks,
            "exported_preference_pairs": len(verified_records),
            "excluded_tasks": len(excluded_tasks),
            "chosen_verification_pass_count": chosen_pass_count,
            "rejected_verification_failure_count": rejected_fail_count,
            "duplicate_ids_count": duplicate_ids_count,
            "dataset_version": dataset_version,
            "output_file": output_file,
            "train_file": actual_train_file,
            "test_file": actual_test_file,
            "train_count": train_count,
            "test_count": test_count,
            "exclusion_details": excluded_tasks,
        }

        print(
            f"✅ Conversational DPO Dataset successfully exported: {output_file} "
            f"({len(verified_records)}/{total_source_tasks} pairs, {len(excluded_tasks)} excluded)"
        )
        return report