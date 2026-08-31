"""Offline contract tests for paired Qwen generation orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from document_rag.rag.config import RAGConfig
from document_rag.rag.generation import (
    QwenBaseLoraComparisonGenerator,
    QwenLoraGenerator,
)
from document_rag.rag.models import ChatMessage, GroundedPrompt


@dataclass
class FakeRuntime:
    """Record adapter state without importing Torch, Transformers, or PEFT."""

    calls: list[bool] = field(default_factory=list)

    def generate(self, prompt: GroundedPrompt, *, adapter_enabled: bool) -> str:
        assert prompt.messages
        self.calls.append(adapter_enabled)
        return "adapter" if adapter_enabled else "base"


def test_comparison_uses_one_runtime_with_base_then_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paired generation should differ only by adapter enablement."""

    runtime = FakeRuntime()
    generator = QwenBaseLoraComparisonGenerator(RAGConfig())
    monkeypatch.setattr(generator, "_runtime", runtime)

    assert generator.generate_pair(_prompt()) == ("base", "adapter")
    assert runtime.calls == [False, True]


def test_application_generator_keeps_the_adapter_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refactoring must preserve the existing web application's behavior."""

    runtime = FakeRuntime()
    generator = QwenLoraGenerator(RAGConfig())
    monkeypatch.setattr(generator, "_runtime", runtime)

    assert generator.generate(_prompt()) == "adapter"
    assert runtime.calls == [True]


def _prompt() -> GroundedPrompt:
    return GroundedPrompt(
        messages=(
            ChatMessage(role="system", content="Use context."),
            ChatMessage(role="user", content="Question"),
        )
    )
