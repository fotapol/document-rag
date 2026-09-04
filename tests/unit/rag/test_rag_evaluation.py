"""Offline tests for frozen-context base-versus-LoRA RAG evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.rag.config import RAGConfig
from document_rag.rag.errors import RAGEvaluationError
from document_rag.rag.evaluation import (
    RAGCaseExpectation,
    assess_answer,
    load_frozen_rag_suite,
    run_frozen_rag_evaluation,
)
from document_rag.rag.models import GroundedPrompt
from document_rag.rag.prompting import UNSUPPORTED_ANSWER


@dataclass
class FakePairedGenerator:
    """Return deterministic answers based on the frozen question text."""

    prompts: list[GroundedPrompt] = field(default_factory=list)

    def generate_pair(self, prompt: GroundedPrompt) -> tuple[str, str]:
        self.prompts.append(prompt)

        if "CEO" in prompt.messages[1].content:
            return UNSUPPORTED_ANSWER, "The CEO is John Doe."

        return "$14.1 million [Source 1].", "14.1"


def test_assessment_separates_numeric_units_and_citations() -> None:
    """A benchmark-style scalar should get numeric credit but fail RAG requirements."""

    expectation = RAGCaseExpectation(
        answerable=True,
        expected_values=("14.1",),
        expected_units=("$", "million"),
        required_source_numbers=(1,),
    )

    complete = assess_answer(
        "Revenue was $14.10 million [Source 1].",
        expectation=expectation,
        context_count=1,
    )
    scalar_only = assess_answer(
        "14.1",
        expectation=expectation,
        context_count=1,
    )

    assert complete.overall_pass is True
    assert complete.numeric_correct is True
    assert complete.units_correct is True
    assert complete.citations_correct is True
    assert scalar_only.content_correct is True
    assert scalar_only.numeric_correct is True
    assert scalar_only.units_correct is False
    assert scalar_only.citations_correct is False
    assert scalar_only.overall_pass is False


def test_assessment_scores_trailing_result_after_reasoning_operands() -> None:
    """A trailing result remains scoreable after its calculation operands."""

    assessment = assess_answer(
        "Revenue rose from $12.4 million to $14.1 million, an increase of $1.7 million [Source 1].",
        expectation=RAGCaseExpectation(
            answerable=True,
            expected_values=("1.7",),
            expected_units=("$", "million"),
            required_source_numbers=(1,),
        ),
        context_count=1,
    )

    assert assessment.numeric_correct is True
    assert assessment.overall_pass is True


@pytest.mark.parametrize(
    ("expected", "answer"),
    [
        ("53.8", "The answer is -53.8%. [Source 1]"),
        ("9.9", "The answer is 99.9%. [Source 1]"),
        ("4", "The answer is 44%. [Source 1]"),
        ("71", "The answer is 171 million. [Source 1]"),
    ],
)
def test_assessment_rejects_wrong_signed_or_partial_numeric_matches(
    expected: str,
    answer: str,
) -> None:
    """Signs and complete decimal values are part of numeric correctness."""

    assessment = assess_answer(
        answer,
        expectation=RAGCaseExpectation(
            answerable=True,
            expected_values=(expected,),
            required_source_numbers=(1,),
        ),
        context_count=1,
    )

    assert assessment.numeric_correct is False
    assert assessment.content_correct is False
    assert assessment.overall_pass is False


def test_explicit_final_answer_cannot_be_satisfied_by_an_operand() -> None:
    """A cited input value must not hide a wrong explicitly stated result."""

    assessment = assess_answer(
        "The inputs are $7 million and $17 million. The answer is $24 million [Source 1].",
        expectation=RAGCaseExpectation(
            answerable=True,
            expected_values=("7",),
            expected_units=("$", "million"),
            required_source_numbers=(1,),
        ),
        context_count=1,
    )

    assert assessment.numeric_correct is False
    assert assessment.overall_pass is False


def test_leading_decimal_answer_is_scored_as_a_decimal() -> None:
    """Benchmark values such as .1 must not be parsed as the integer one."""

    assessment = assess_answer(
        "The answer is 0.1%. [Source 1]",
        expectation=RAGCaseExpectation(
            answerable=True,
            expected_values=(".1",),
            expected_units=("%",),
            required_source_numbers=(1,),
        ),
        context_count=1,
    )

    assert assessment.numeric_correct is True
    assert assessment.overall_pass is True


def test_expected_phrase_does_not_match_inside_a_larger_word() -> None:
    """A required `no` answer cannot receive credit from the word `cannot`."""

    assessment = assess_answer(
        "I cannot determine this from the context [Source 1].",
        expectation=RAGCaseExpectation(
            answerable=True,
            expected_phrases=("no",),
            required_source_numbers=(1,),
        ),
        context_count=1,
    )

    assert assessment.content_correct is False
    assert assessment.overall_pass is False


def test_source_number_cannot_satisfy_an_expected_numeric_value() -> None:
    """Citation labels are metadata and must not count as financial values."""

    assessment = assess_answer(
        "The answer is unavailable [Source 1].",
        expectation=RAGCaseExpectation(
            answerable=True,
            expected_values=("1",),
            required_source_numbers=(1,),
        ),
        context_count=1,
    )

    assert assessment.numeric_correct is False
    assert assessment.content_correct is False
    assert assessment.overall_pass is False


def test_full_context_header_is_accepted_as_a_source_citation() -> None:
    """Copying the prompt's full source label should remain traceable and valid."""

    assessment = assess_answer(
        "$14.1 million [Source 5 | page 1 | chunk_id chunk:annual].",
        expectation=RAGCaseExpectation(
            answerable=True,
            expected_values=("14.1",),
            expected_units=("$", "million"),
            required_source_numbers=(5,),
        ),
        context_count=5,
    )

    assert assessment.cited_source_numbers == (5,)
    assert assessment.citations_correct is True
    assert assessment.overall_pass is True


def test_unsupported_case_requires_the_exact_grounded_refusal() -> None:
    """A plausible unsupported claim must fail even when it is well formed."""

    expectation = RAGCaseExpectation(answerable=False)

    assert assess_answer(
        UNSUPPORTED_ANSWER,
        expectation=expectation,
        context_count=1,
    ).overall_pass
    hallucination = assess_answer(
        "The CEO is John Doe.",
        expectation=expectation,
        context_count=1,
    )
    assert hallucination.refusal_correct is False
    assert hallucination.overall_pass is False


def test_run_writes_deterministic_paired_predictions_and_metrics(tmp_path: Path) -> None:
    """One prompt per case should produce stable, directly comparable artifacts."""

    suite_path = tmp_path / "suite.json"
    _write_suite(suite_path)
    first_generator = FakePairedGenerator()
    second_generator = FakePairedGenerator()
    first = run_frozen_rag_evaluation(
        suite_path=suite_path,
        output_directory=tmp_path / "first",
        generator=first_generator,
        config=RAGConfig(),
    )
    second = run_frozen_rag_evaluation(
        suite_path=suite_path,
        output_directory=tmp_path / "second",
        generator=second_generator,
        config=RAGConfig(),
    )

    assert first.predictions_path.read_bytes() == second.predictions_path.read_bytes()
    assert first.metrics_path.read_bytes() == second.metrics_path.read_bytes()
    assert len(first_generator.prompts) == 2
    assert first.base_metrics.overall.correct == 2
    assert first.base_metrics.overall.count == 2
    assert first.adapter_metrics.numeric.correct == 1
    assert first.adapter_metrics.overall.correct == 0
    predictions = [
        json.loads(line) for line in first.predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert predictions[0]["base"]["answer"] == "$14.1 million [Source 1]."
    assert predictions[0]["adapter"]["answer"] == "14.1"
    assert predictions[0]["prompt_sha256"]
    assert predictions[0]["prompt_messages"] == first_generator.prompts[0].to_messages()
    metrics = json.loads(first.metrics_path.read_text(encoding="utf-8"))
    assert metrics["evaluation_config"]["do_sample"] is False
    assert metrics["evaluation_config"]["base_model_revision"] == RAGConfig().base_model_revision
    assert metrics["metrics"]["delta_adapter_minus_base"]["overall"] == -1.0
    assert set(metrics["metrics_by_category"]) == {"simple_lookup", "unsupported"}
    assert set(metrics["metrics_by_context_mode"]) == {"production"}


def test_loader_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """Version checks prevent silently interpreting a changed suite format."""

    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 99,
                "name": "future",
                "document_sha256": "abc",
                "retrieval_config": {},
                "cases": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RAGEvaluationError, match="schema_version"):
        load_frozen_rag_suite(path)


def test_answerable_expectation_requires_a_citation() -> None:
    """Supported cases must test grounded citations, not only answer content."""

    with pytest.raises(RAGEvaluationError, match="source citation"):
        RAGCaseExpectation(answerable=True, expected_values=("1",))


def _write_suite(path: Path) -> None:
    revenue_chunk = _chunk(
        "chunk:revenue",
        "Revenue was $14.1 million.",
    )
    governance_chunk = _chunk(
        "chunk:governance",
        "The supplied report does not identify a CEO.",
    )
    payload = {
        "schema_version": 1,
        "name": "offline-diagnostic",
        "document_sha256": "a" * 64,
        "retrieval_config": {
            "type": "hybrid_rrf",
            "top_k": 1,
        },
        "cases": [
            _case_record(
                case_id="simple",
                category="simple_lookup",
                question="What was revenue?",
                expectation={
                    "answerable": True,
                    "expected_values": ["14.1"],
                    "expected_units": ["$", "million"],
                    "required_source_numbers": [1],
                },
                chunk=revenue_chunk,
            ),
            _case_record(
                case_id="unsupported",
                category="unsupported",
                question="Who is the CEO?",
                expectation={"answerable": False},
                chunk=governance_chunk,
            ),
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _case_record(
    *,
    case_id: str,
    category: str,
    question: str,
    expectation: dict[str, object],
    chunk: DocumentChunk,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "category": category,
        "context_mode": "production",
        "production_gold_rank": 1 if expectation["answerable"] else None,
        "question": question,
        "expectation": expectation,
        "retrieved": [
            {
                "rank": 1,
                "score": 1.0,
                "chunk": chunk.to_record(),
            }
        ],
    }


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="document:test",
        document_sha256="a" * 64,
        filename="report.pdf",
        chunk_index=0,
        page_start=1,
        page_end=1,
        source_element_ids=(f"source:{chunk_id}",),
        text=text,
        char_count=len(text),
        token_count=len(text.split()),
        block_count=1,
    )
