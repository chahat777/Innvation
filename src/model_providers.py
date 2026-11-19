import os
import re
import time
import base64
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

from src.schema import QuestionType
from src.verifier import AnswerVerifier

# Load local environment variables from .env if present
load_dotenv()


def mask_secrets(text: str, specific_key: Optional[str] = None) -> str:
    """
    Sanitizes error messages and logs to ensure no API keys or secrets are leaked.
    """
    if not text:
        return ""
    sanitized = str(text)
    if specific_key and specific_key.strip():
        sanitized = sanitized.replace(specific_key.strip(), "***REDACTED_KEY***")
    
    # Generic regex masks for common provider key shapes
    sanitized = re.sub(r'AIza[0-9A-Za-z\-_]{20,}', 'AIza***REDACTED***', sanitized)
    sanitized = re.sub(r'sk-ant-[0-9A-Za-z\-_]{20,}', 'sk-ant-***REDACTED***', sanitized)
    sanitized = re.sub(r'sk-[0-9A-Za-z\-_]{20,}', 'sk-***REDACTED***', sanitized)
    return sanitized


@dataclass
class ProviderEvaluationResult:
    provider: str
    model_name: str
    status: str  # "REAL_API", "NOT_CONFIGURED", "API_ERROR", "SIMULATED"
    raw_response: str
    parsed_answer: Any
    is_correct: Optional[bool]
    latency_seconds: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "status": self.status,
            "raw_response": self.raw_response,
            "parsed_answer": self.parsed_answer,
            "is_correct": self.is_correct,
            "latency_seconds": round(self.latency_seconds, 3),
            "error_message": self.error_message
        }


class BaseVisionProvider:
    provider_name: str = "Base"
    default_model: str = "base-model"

    def get_api_key(self) -> Optional[str]:
        raise NotImplementedError

    def is_configured(self) -> bool:
        key = self.get_api_key()
        if not key or not key.strip():
            return False
        cleaned = key.strip()
        # Guard against dummy / placeholder values like sk-xxxx, your_key, etc.
        if re.match(r'^(sk-[xX]+|your_.*|YOUR_.*|<.*>|TODO.*)$', cleaned):
            return False
        return True

    @staticmethod
    def encode_image_base64(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def evaluate(
        self,
        image_path: str,
        prompt: str,
        reference_answer: Any,
        question_type: QuestionType,
        timeout: float = 30.0
    ) -> ProviderEvaluationResult:
        raise NotImplementedError


class GeminiVisionProvider(BaseVisionProvider):
    provider_name = "Google DeepMind"
    default_model = "gemini-1.5-flash"

    def get_api_key(self) -> Optional[str]:
        return os.getenv("GEMINI_API_KEY")

    def evaluate(
        self,
        image_path: str,
        prompt: str,
        reference_answer: Any,
        question_type: QuestionType,
        timeout: float = 30.0
    ) -> ProviderEvaluationResult:
        api_key = self.get_api_key()
        if not self.is_configured():
            return ProviderEvaluationResult(
                provider=self.provider_name,
                model_name=self.default_model,
                status="NOT_CONFIGURED",
                raw_response="",
                parsed_answer=None,
                is_correct=None,
                latency_seconds=0.0,
                error_message="GEMINI_API_KEY is not configured in .env."
            )

        start_time = time.time()
        try:
            raw_text = ""
            # Check availability of google.genai (new SDK) vs google.generativeai (legacy)
            try:
                from google import genai
                use_modern_genai = True
            except ImportError:
                use_modern_genai = False

            if use_modern_genai:
                from google import genai
                from PIL import Image
                client = genai.Client(api_key=api_key)
                with Image.open(image_path) as img:
                    response = client.models.generate_content(
                        model=self.default_model,
                        contents=[prompt, img]
                    )
                raw_text = response.text.strip() if response.text else ""
            else:
                import google.generativeai as legacy_genai
                from PIL import Image
                legacy_genai.configure(api_key=api_key)
                model = legacy_genai.GenerativeModel(self.default_model)
                with Image.open(image_path) as img:
                    response = model.generate_content(
                        [prompt, img],
                        request_options={"timeout": timeout}
                    )
                raw_text = response.text.strip() if response.text else ""

            latency = time.time() - start_time
            q_type = QuestionType(question_type) if isinstance(question_type, str) else question_type
            is_correct, parsed_val, err_cat = AnswerVerifier.verify(raw_text, reference_answer, q_type)

            return ProviderEvaluationResult(
                provider=self.provider_name,
                model_name=self.default_model,
                status="REAL_API",
                raw_response=raw_text,
                parsed_answer=parsed_val,
                is_correct=is_correct,
                latency_seconds=latency
            )
        except Exception as e:
            latency = time.time() - start_time
            clean_err = mask_secrets(str(e), api_key)
            return ProviderEvaluationResult(
                provider=self.provider_name,
                model_name=self.default_model,
                status="API_ERROR",
                raw_response="",
                parsed_answer=None,
                is_correct=None,
                latency_seconds=latency,
                error_message=f"Gemini API Error: {clean_err}"
            )


class OpenAIVisionProvider(BaseVisionProvider):
    provider_name = "OpenAI"
    default_model = "gpt-4o"

    def get_api_key(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")

    def evaluate(
        self,
        image_path: str,
        prompt: str,
        reference_answer: Any,
        question_type: QuestionType,
        timeout: float = 30.0
    ) -> ProviderEvaluationResult:
        api_key = self.get_api_key()
        if not self.is_configured():
            return ProviderEvaluationResult(
                provider=self.provider_name,
                model_name=self.default_model,
                status="NOT_CONFIGURED",
                raw_response="",
                parsed_answer=None,
                is_correct=None,
                latency_seconds=0.0,
                error_message="OPENAI_API_KEY is not configured in .env."
            )

        start_time = time.time()
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, timeout=timeout)
            b64_image = self.encode_image_base64(image_path)

            response = client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=250
            )

            raw_text = response.choices[0].message.content.strip() if response.choices else ""
            latency = time.time() - start_time
            q_type = QuestionType(question_type) if isinstance(question_type, str) else question_type
            is_correct, parsed_val, err_cat = AnswerVerifier.verify(raw_text, reference_answer, q_type)

            return ProviderEvaluationResult(
                provider=self.provider_name,
                model_name=self.default_model,
                status="REAL_API",
                raw_response=raw_text,
                parsed_answer=parsed_val,
                is_correct=is_correct,
                latency_seconds=latency
            )
        except Exception as e:
            latency = time.time() - start_time
            clean_err = mask_secrets(str(e), api_key)
            return ProviderEvaluationResult(
                provider=self.provider_name,
                model_name=self.default_model,
                status="API_ERROR",
                raw_response="",
                parsed_answer=None,
                is_correct=None,
                latency_seconds=latency,
                error_message=f"OpenAI API Error: {clean_err}"
            )


class AnthropicVisionProvider(BaseVisionProvider):
    provider_name = "Anthropic"
    default_model = "claude-3-5-sonnet-20241022"

    def get_api_key(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY")

    def evaluate(
        self,
        image_path: str,
        prompt: str,
        reference_answer: Any,
        question_type: QuestionType,
        timeout: float = 30.0
    ) -> ProviderEvaluationResult:
        api_key = self.get_api_key()
        if not self.is_configured():
            return ProviderEvaluationResult(
                provider=self.provider_name,
                model_name=self.default_model,
                status="NOT_CONFIGURED",
                raw_response="",
                parsed_answer=None,
                is_correct=None,
                latency_seconds=0.0,
                error_message="ANTHROPIC_API_KEY is not configured in .env."
            )

        start_time = time.time()
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
            b64_image = self.encode_image_base64(image_path)

            message = client.messages.create(
                model=self.default_model,
                max_tokens=250,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64_image
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )

            raw_text = ""
            for block in message.content:
                if getattr(block, "type", None) == "text":
                    raw_text += getattr(block, "text", "")
            raw_text = raw_text.strip()

            latency = time.time() - start_time
            q_type = QuestionType(question_type) if isinstance(question_type, str) else question_type
            is_correct, parsed_val, err_cat = AnswerVerifier.verify(raw_text, reference_answer, q_type)

            return ProviderEvaluationResult(
                provider=self.provider_name,
                model_name=self.default_model,
                status="REAL_API",
                raw_response=raw_text,
                parsed_answer=parsed_val,
                is_correct=is_correct,
                latency_seconds=latency
            )
        except Exception as e:
            latency = time.time() - start_time
            clean_err = mask_secrets(str(e), api_key)
            return ProviderEvaluationResult(
                provider=self.provider_name,
                model_name=self.default_model,
                status="API_ERROR",
                raw_response="",
                parsed_answer=None,
                is_correct=None,
                latency_seconds=latency,
                error_message=f"Anthropic API Error: {clean_err}"
            )


class MultiModelArenaRunner:
    """
    Coordinates multi-model vision evaluations across Google Gemini, OpenAI GPT-4o, and Anthropic Claude.
    """

    def __init__(self):
        self.providers: List[BaseVisionProvider] = [
            GeminiVisionProvider(),
            OpenAIVisionProvider(),
            AnthropicVisionProvider()
        ]

    def get_provider_status_overview(self) -> Dict[str, Dict[str, Any]]:
        overview = {}
        for p in self.providers:
            overview[p.provider_name] = {
                "model": p.default_model,
                "configured": p.is_configured(),
                "key_env": (
                    "GEMINI_API_KEY" if p.provider_name == "Google DeepMind"
                    else "OPENAI_API_KEY" if p.provider_name == "OpenAI"
                    else "ANTHROPIC_API_KEY"
                )
            }
        return overview

    def evaluate_task(
        self,
        image_path: str,
        prompt: str,
        reference_answer: Any,
        question_type: QuestionType,
        selected_providers: Optional[List[str]] = None,
        timeout: float = 30.0
    ) -> Dict[str, ProviderEvaluationResult]:
        """
        Executes live API calls on selected configured providers.
        """
        results = {}
        for p in self.providers:
            if selected_providers and p.provider_name not in selected_providers:
                continue
            res = p.evaluate(
                image_path=image_path,
                prompt=prompt,
                reference_answer=reference_answer,
                question_type=question_type,
                timeout=timeout
            )
            results[p.provider_name] = res
        return results

    @staticmethod
    def get_simulated_result(
        task_id: str,
        provider_name: str,
        model_name: str,
        reference_answer: Any,
        question_type: QuestionType,
        eval_results_path: str = "dataset/evaluation_results.json"
    ) -> ProviderEvaluationResult:
        """
        Loads pre-computed benchmark results for offline demonstration.
        Strictly labeled as SIMULATED so as never to mislead users.
        """
        raw_resp = "Simulated benchmark response unavailable."
        parsed_val = None
        is_corr = False
        latency = 0.05

        if os.path.exists(eval_results_path):
            try:
                with open(eval_results_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                detailed = data.get("detailed_results", [])
                match = next((r for r in detailed if r.get("task_id") == task_id), None)
                if match:
                    raw_resp = match.get("raw_response", match.get("predicted_answer", ""))
                    parsed_val = match.get("parsed_answer")
                    is_corr = match.get("is_correct", False)
                    latency = match.get("latency_seconds", 0.05)
            except Exception:
                pass

        if parsed_val is None:
            # Deterministic simulation fallback
            q_type = QuestionType(question_type) if isinstance(question_type, str) else question_type
            is_corr, parsed_val, _ = AnswerVerifier.verify(raw_resp, reference_answer, q_type)

        return ProviderEvaluationResult(
            provider=provider_name,
            model_name=model_name,
            status="SIMULATED",
            raw_response=raw_resp,
            parsed_answer=parsed_val,
            is_correct=is_corr,
            latency_seconds=latency,
            error_message="Offline benchmark simulation (API keys not configured)."
        )
