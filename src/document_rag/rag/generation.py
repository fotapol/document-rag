"""Lazy deterministic generation with Qwen3 and an unmerged PEFT adapter."""

from __future__ import annotations

from typing import Any, cast

from document_rag.rag.config import RAGConfig
from document_rag.rag.errors import RAGGenerationError
from document_rag.rag.models import GroundedPrompt


class QwenBaseLoraComparisonGenerator:
    """Generate paired base and adapter answers from one loaded PEFT model."""

    def __init__(self, config: RAGConfig) -> None:
        """Create one lazy runtime shared by both comparison conditions."""

        self._runtime = _QwenLoraRuntime(config)

    def generate_pair(self, prompt: GroundedPrompt) -> tuple[str, str]:
        """Return base then adapter output for exactly the same prompt."""

        base_answer = self._runtime.generate(prompt, adapter_enabled=False)
        adapter_answer = self._runtime.generate(prompt, adapter_enabled=True)
        return base_answer, adapter_answer


class QwenLoraGenerator:
    """Load the pinned base model and financial LoRA only when first used."""

    def __init__(self, config: RAGConfig) -> None:
        """Store model configuration without downloading or allocating weights."""

        self._runtime = _QwenLoraRuntime(config)

    def generate(self, prompt: GroundedPrompt) -> str:
        """Generate one deterministic answer from a grounded chat prompt."""

        return self._runtime.generate(prompt, adapter_enabled=True)


class _QwenLoraRuntime:
    """Lazy tokenizer/model runtime supporting temporary adapter disablement."""

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def generate(self, prompt: GroundedPrompt, *, adapter_enabled: bool) -> str:
        """Generate deterministically with the adapter enabled or disabled."""

        try:
            tokenizer, model = self._load_model()

            if adapter_enabled:
                answer = self._generate_loaded(prompt, tokenizer=tokenizer, model=model)
            else:
                with model.disable_adapter():
                    answer = self._generate_loaded(prompt, tokenizer=tokenizer, model=model)
        except RAGGenerationError:
            raise
        except Exception as exc:
            raise RAGGenerationError("The answer model could not generate a response.") from exc

        if not answer:
            raise RAGGenerationError("The answer model returned an empty response.")

        return answer

    def _generate_loaded(self, prompt: GroundedPrompt, *, tokenizer: Any, model: Any) -> str:
        import torch

        rendered_prompt = cast(
            str,
            tokenizer.apply_chat_template(
                prompt.to_messages(),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
        )
        inputs: Any = tokenizer(
            rendered_prompt,
            return_tensors="pt",
        )
        input_length = int(inputs["input_ids"].shape[1])

        if input_length > self._config.max_input_tokens:
            raise RAGGenerationError(
                "The grounded prompt exceeds DOCUMENT_RAG_MAX_INPUT_TOKENS; "
                "reduce DOCUMENT_RAG_TOP_K or increase the input limit."
            )

        inputs = inputs.to(model.device)
        pad_token_id = tokenizer.pad_token_id

        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id

        with torch.inference_mode():
            output: Any = model.generate(
                **inputs,
                max_new_tokens=self._config.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        return cast(
            str,
            tokenizer.decode(
                output[0, input_length:],
                skip_special_tokens=True,
            ),
        ).strip()

    def _load_model(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model

        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            adapter_revision: dict[str, Any] = (
                {}
                if self._config.adapter_model_revision is None
                else {"revision": self._config.adapter_model_revision}
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self._config.adapter_model_id,
                **adapter_revision,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self._config.base_model_id,
                revision=self._config.base_model_revision,
                dtype="auto",
                device_map=self._config.generation_device_map,
            )

            model = PeftModel.from_pretrained(
                base_model,
                self._config.adapter_model_id,
                is_trainable=False,
                **adapter_revision,
            )
            model.eval()
        except Exception as exc:
            raise RAGGenerationError(
                "Could not load the pinned Qwen base model and financial LoRA adapter."
            ) from exc

        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model
