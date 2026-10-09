import re
from abc import ABC, abstractmethod
from typing import Any

from src.models import SourceExcerpt
from src.prompting import SYSTEM_PROMPT, PromptBuilder


MODEL_NAME = "Qwen/Qwen3-0.6B"

NO_CONTEXT_ANSWER = (
    "The provided context does not contain the answer."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class AnswerGenerator(ABC):
    """Produces an answer grounded in retrieved excerpts."""

    @abstractmethod
    def generate(
        self,
        question: str,
        excerpts: list[SourceExcerpt],
    ) -> str:
        """Answer the question using only the excerpts."""


class QwenAnswerGenerator(AnswerGenerator):
    """Answer generation with a local Qwen chat model."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_context_tokens: int = 2500,
        max_new_tokens: int = 160,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._max_context_tokens = max_context_tokens
        self._max_new_tokens = max_new_tokens
        self._tokenizer: Any = None
        self._model: Any = None

    def generate(
        self,
        question: str,
        excerpts: list[SourceExcerpt],
    ) -> str:
        if not excerpts:
            return NO_CONTEXT_ANSWER

        tokenizer, model = self._load()

        prompt_builder = PromptBuilder(
            count_tokens=lambda text: len(
                tokenizer.encode(text, add_special_tokens=False)
            ),
            max_context_tokens=self._max_context_tokens,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt_builder.build_user_prompt(
                    question=question,
                    excerpts=excerpts,
                ),
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(
            model.device
        )

        import torch

        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )
        except (RuntimeError, ValueError, MemoryError) as error:
            raise RuntimeError(
                f"Answer generation failed: {error}"
            ) from error

        generated = output[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        return self._clean(text)

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            try:
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                )

                self._tokenizer = (
                    AutoTokenizer.from_pretrained(self._model_name)
                )
                self._model = (
                    AutoModelForCausalLM.from_pretrained(
                        self._model_name,
                        dtype=torch.float32,
                    )
                )
                self._model.to(self._resolve_device(torch))
                self._model.eval()
            except OSError as error:
                raise RuntimeError(
                    f"Could not load model {self._model_name}: "
                    f"{error}"
                ) from error

        return self._tokenizer, self._model

    def _resolve_device(self, torch: Any) -> str:
        if self._device is not None:
            return self._device

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @staticmethod
    def _clean(text: str) -> str:
        cleaned = _THINK_BLOCK.sub("", text)
        cleaned = cleaned.split("</think>")[-1].strip()

        return cleaned or NO_CONTEXT_ANSWER
