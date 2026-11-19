import re
from typing import Any, Tuple, Optional
from src.schema import QuestionType

class AnswerVerifier:
    """
    Independent deterministic verifier that parses model responses and matches them
    against mathematically guaranteed ground truth.
    Supports both direct plain answers and responses containing explicit '### Final Answer:' sections.
    """

    FINAL_ANSWER_REGEX = re.compile(r'(?:#{1,6}\s*)?Final Answer\s*:\s*', re.IGNORECASE)

    @staticmethod
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        return str(text).strip().lower()

    @classmethod
    def extract_final_answer(cls, raw_pred: str) -> Tuple[str, bool]:
        """
        Extracts the final-answer section if a 'Final Answer:' marker is present.
        Returns: (extracted_text, has_marker)
        """
        if not raw_pred:
            return "", False

        matches = list(cls.FINAL_ANSWER_REGEX.finditer(raw_pred))
        if not matches:
            return raw_pred.strip(), False

        # Extract content following the last 'Final Answer:' marker
        extracted = raw_pred[matches[-1].end():].strip()
        return extracted, True

    @classmethod
    def verify(cls, raw_pred: str, reference: Any, q_type: QuestionType) -> Tuple[bool, Any, Optional[str]]:
        """
        Parses raw model response and checks validity against reference ground truth.
        When '### Final Answer:' marker is present, performs task-specific verification
        ONLY on that extracted section, preventing numbered rationale steps or narrative keywords
        from polluting answer extraction.

        Returns (is_correct, parsed_value, error_category).
        """
        if raw_pred is None:
            return False, None, "EMPTY_RESPONSE"

        raw_str = str(raw_pred)
        if not raw_str.strip():
            return False, None, "EMPTY_RESPONSE"

        # Safe Enum casting
        q_type = QuestionType(q_type) if isinstance(q_type, str) else q_type

        # Extract target text
        target_text, has_marker = cls.extract_final_answer(raw_str)
        if has_marker and not target_text:
            return False, None, "EMPTY_FINAL_ANSWER"

        raw_clean = cls.normalize_text(target_text)
        if not raw_clean:
            return False, None, "EMPTY_RESPONSE"

        if q_type == QuestionType.CONNECTIVITY_EXISTS:
            # Boolean evaluation on target extracted answer
            has_yes = bool(re.search(r'\b(yes|true|connected|is connected)\b', raw_clean))
            has_no = bool(re.search(r'\b(no|false|not connected|disconnected)\b', raw_clean))

            if has_yes and not has_no:
                parsed = "Yes"
            elif has_no and not has_yes:
                parsed = "No"
            else:
                parsed = raw_clean

            expected = "Yes" if str(reference).lower() in ["yes", "true"] else "No"
            is_correct = (parsed == expected)
            err = None if is_correct else "CONNECTIVITY_MISMATCH"
            return is_correct, parsed, err

        elif q_type in [QuestionType.COMPONENT_COUNT, QuestionType.SHORTEST_PATH_LENGTH]:
            # Integer evaluation on target extracted answer
            numbers = re.findall(r'\b\d+\b', raw_clean)
            if not numbers:
                return False, None, "NO_NUMERICAL_VALUE"

            try:
                parsed_int = int(numbers[0])
            except ValueError:
                return False, None, "PARSE_ERROR"

            expected_int = int(reference)
            is_correct = (parsed_int == expected_int)
            err = None if is_correct else f"COUNT_ERROR_DIFF_{abs(parsed_int - expected_int)}"
            return is_correct, parsed_int, err

        elif q_type == QuestionType.NEIGHBOR_IDENTIFICATION:
            # Component Label / Entity set check on target extracted answer
            valid_set = [str(x).upper() for x in (reference if isinstance(reference, list) else [reference])]

            tokens = re.findall(r'\b[A-Z]+[0-9]*\b', target_text.upper())
            matching = [t for t in tokens if t in valid_set]

            if matching:
                return True, matching[0], None
            else:
                return False, tokens[0] if tokens else raw_clean, "WRONG_COMPONENT_IDENTIFIED"

        # Fallback exact string matching
        is_match = (raw_clean == cls.normalize_text(str(reference)))
        return is_match, raw_clean, None if is_match else "MISMATCH"
