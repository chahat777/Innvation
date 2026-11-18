import re
from typing import Any, Tuple, Optional
from src.schema import QuestionType

class AnswerVerifier:
    """
    Independent deterministic verifier that parses model responses and matches them
    against mathematically guaranteed ground truth.
    """

    @staticmethod
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        return str(text).strip().lower()

    @classmethod
    def verify(cls, raw_pred: str, reference: Any, q_type: QuestionType) -> Tuple[bool, Any, Optional[str]]:
        """
        Returns (is_correct, parsed_value, error_category).
        """
        raw_clean = cls.normalize_text(raw_pred)
        
        if not raw_clean:
            return False, None, "EMPTY_RESPONSE"

        if q_type == QuestionType.CONNECTIVITY_EXISTS:
            # Boolean evaluation
            # Check for definitive affirmative / negative
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
            # Integer evaluation
            # Find numbers in model response
            numbers = re.findall(r'\b\d+\b', raw_clean)
            if not numbers:
                return False, None, "NO_NUMERICAL_VALUE"
            
            # Usually the last or first number is the answer
            try:
                parsed_int = int(numbers[0])
            except ValueError:
                return False, None, "PARSE_ERROR"
                
            expected_int = int(reference)
            is_correct = (parsed_int == expected_int)
            err = None if is_correct else f"COUNT_ERROR_DIFF_{abs(parsed_int - expected_int)}"
            return is_correct, parsed_int, err

        elif q_type == QuestionType.NEIGHBOR_IDENTIFICATION:
            # Component Label / Entity set check
            # reference can be a list of valid neighbor labels like ['R1', 'C2']
            valid_set = [str(x).upper() for x in (reference if isinstance(reference, list) else [reference])]
            
            # Extract standard labels (e.g. R1, C2, GND1, etc.)
            tokens = re.findall(r'\b[A-Z]+[0-9]*\b', raw_pred.upper())
            matching = [t for t in tokens if t in valid_set]
            
            if matching:
                return True, matching[0], None
            else:
                return False, tokens[0] if tokens else raw_clean, "WRONG_COMPONENT_IDENTIFIED"

        # Fallback exact string matching
        is_match = (raw_clean == cls.normalize_text(str(reference)))
        return is_match, raw_clean, None if is_match else "MISMATCH"
