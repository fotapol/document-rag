"""Frozen-context comparison of base Qwen and the financial LoRA adapter."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Literal, Protocol, cast

from document_rag.ingestion.chunking import DocumentChunk, RetrievalUnitKind
from document_rag.rag.config import RAGConfig
from document_rag.rag.errors import RAGEvaluationError
from document_rag.rag.models import GroundedPrompt
from document_rag.rag.prompting import (
    SYSTEM_INSTRUCTION,
    UNSUPPORTED_ANSWER,
    build_grounded_prompt,
)
from document_rag.retrieval.benchmark import write_bytes_atomically
from document_rag.retrieval.models import RetrievalResult

RAG_EVALUATION_SCHEMA_VERSION = 1
RAG_PREDICTIONS_FILENAME = "rag_predictions.jsonl"
RAG_METRICS_FILENAME = "rag_metrics.json"
type RAGCaseCategory = Literal[
    "simple_lookup",
    "table_lookup",
    "reasoning",
    "unsupported",
]
type RAGContextMode = Literal["production", "oracle_augmented"]
_CONTEXT_MODES: tuple[RAGContextMode, ...] = ("production", "oracle_augmented")
_CASE_CATEGORIES: tuple[RAGCaseCategory, ...] = (
    "simple_lookup",
    "table_lookup",
    "reasoning",
    "unsupported",
)
_NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?!\w)")
_ANSWER_MARKER_PATTERN = re.compile(r"\bthe\s+answer\s+is\b", re.IGNORECASE)
_SOURCE_PATTERN = re.compile(
    r"\[\s*Source\s+(\d+)(?:\s*\]|\s*\|[^\]]*\])",
    re.IGNORECASE,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


class PairedAnswerGenerator(Protocol):
    """Generate base and adapter answers for one already-frozen prompt."""

    def generate_pair(self, prompt: GroundedPrompt) -> tuple[str, str]:
        """Return base output followed by adapter output."""


@dataclass(frozen=True, slots=True)
class RAGCaseExpectation:
    """Machine-checkable requirements for one diagnostic question."""

    answerable: bool
    expected_values: tuple[str, ...] = ()
    expected_units: tuple[str, ...] = ()
    expected_phrases: tuple[str, ...] = ()
    required_source_numbers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Reject ambiguous or internally inconsistent expectations."""

        if self.answerable and not (self.expected_values or self.expected_phrases):
            raise RAGEvaluationError(
                "An answerable case requires expected_values and/or expected_phrases."
            )

        if self.answerable and not self.required_source_numbers:
            raise RAGEvaluationError("An answerable case requires at least one source citation.")

        if not self.answerable and (
            self.expected_values
            or self.expected_units
            or self.expected_phrases
            or self.required_source_numbers
        ):
            raise RAGEvaluationError("An unsupported case cannot define supported-answer fields.")

        if any(not value.strip() for value in self.expected_values):
            raise RAGEvaluationError("Expected numeric values cannot be empty.")

        for value in self.expected_values:
            _parse_decimal(value, label="Expected numeric value")

        if any(not unit.strip() for unit in self.expected_units):
            raise RAGEvaluationError("Expected units cannot be empty.")

        if any(not phrase.strip() for phrase in self.expected_phrases):
            raise RAGEvaluationError("Expected phrases cannot be empty.")

        if any(source_number <= 0 for source_number in self.required_source_numbers):
            raise RAGEvaluationError("Required source numbers must be positive.")

        if len(set(self.required_source_numbers)) != len(self.required_source_numbers):
            raise RAGEvaluationError("Required source numbers must be unique.")

    def to_record(self) -> dict[str, object]:
        """Serialize the expected answer contract."""

        return {
            "answerable": self.answerable,
            "expected_phrases": list(self.expected_phrases),
            "expected_units": list(self.expected_units),
            "expected_values": list(self.expected_values),
            "required_source_numbers": list(self.required_source_numbers),
        }


@dataclass(frozen=True, slots=True)
class FrozenRAGCase:
    """One question paired with immutable retrieved context and expectations."""

    case_id: str
    category: RAGCaseCategory
    context_mode: RAGContextMode
    production_gold_rank: int | None
    question: str
    expectation: RAGCaseExpectation
    retrieved: tuple[RetrievalResult, ...]

    def __post_init__(self) -> None:
        """Validate identities, rankings, and required citations."""

        if not self.case_id.strip():
            raise RAGEvaluationError("case_id must not be empty.")

        if not self.question.strip():
            raise RAGEvaluationError(f"Case {self.case_id!r} has an empty question.")

        if not self.retrieved:
            raise RAGEvaluationError(f"Case {self.case_id!r} has no frozen context.")

        expected_ranks = tuple(range(1, len(self.retrieved) + 1))

        if tuple(result.rank for result in self.retrieved) != expected_ranks:
            raise RAGEvaluationError(
                f"Case {self.case_id!r} retrieval ranks must be contiguous from 1."
            )

        if self.production_gold_rank is not None and self.production_gold_rank <= 0:
            raise RAGEvaluationError(
                f"Case {self.case_id!r} production_gold_rank must be positive."
            )

        chunk_ids = tuple(result.chunk_id for result in self.retrieved)

        if len(set(chunk_ids)) != len(chunk_ids):
            raise RAGEvaluationError(f"Case {self.case_id!r} contains duplicate retrieved chunks.")

        if any(
            source_number > len(self.retrieved)
            for source_number in self.expectation.required_source_numbers
        ):
            raise RAGEvaluationError(
                f"Case {self.case_id!r} requires a source outside its frozen context."
            )


@dataclass(frozen=True, slots=True)
class FrozenRAGSuite:
    """A versioned collection of cases sharing one frozen input file."""

    name: str
    document_sha256: str
    retrieval_config: tuple[tuple[str, object], ...]
    cases: tuple[FrozenRAGCase, ...]
    suite_sha256: str

    def __post_init__(self) -> None:
        """Validate suite identity and unique case IDs."""

        if not self.name.strip():
            raise RAGEvaluationError("Suite name must not be empty.")

        if not self.document_sha256.strip():
            raise RAGEvaluationError("document_sha256 must not be empty.")

        if not self.cases:
            raise RAGEvaluationError("A RAG evaluation suite must contain at least one case.")

        case_ids = tuple(case.case_id for case in self.cases)

        if len(set(case_ids)) != len(case_ids):
            raise RAGEvaluationError("RAG evaluation case IDs must be unique.")


@dataclass(frozen=True, slots=True)
class AnswerAssessment:
    """Independent checks for one generated answer."""

    answer: str
    content_correct: bool
    refusal_correct: bool
    numeric_correct: bool | None
    units_correct: bool | None
    citations_correct: bool | None
    cited_source_numbers: tuple[int, ...]
    overall_pass: bool

    def to_record(self) -> dict[str, object]:
        """Serialize answer text and every diagnostic dimension."""

        return {
            "answer": self.answer,
            "citations_correct": self.citations_correct,
            "cited_source_numbers": list(self.cited_source_numbers),
            "content_correct": self.content_correct,
            "numeric_correct": self.numeric_correct,
            "overall_pass": self.overall_pass,
            "refusal_correct": self.refusal_correct,
            "units_correct": self.units_correct,
        }


@dataclass(frozen=True, slots=True)
class RAGCaseComparison:
    """Base and adapter outcomes for one identical grounded prompt."""

    case: FrozenRAGCase
    prompt: GroundedPrompt
    prompt_sha256: str
    base: AnswerAssessment
    adapter: AnswerAssessment

    def to_record(self) -> dict[str, object]:
        """Serialize a fully auditable per-case comparison."""

        return {
            "adapter": self.adapter.to_record(),
            "base": self.base.to_record(),
            "case_id": self.case.case_id,
            "category": self.case.category,
            "context_mode": self.case.context_mode,
            "expectation": self.case.expectation.to_record(),
            "prompt_messages": self.prompt.to_messages(),
            "prompt_sha256": self.prompt_sha256,
            "production_gold_rank": self.case.production_gold_rank,
            "question": self.case.question,
            "retrieved": [
                {
                    "chunk": result.chunk.to_record(),
                    "rank": result.rank,
                    "score": result.score,
                }
                for result in self.case.retrieved
            ],
        }


@dataclass(frozen=True, slots=True)
class MetricScore:
    """Correct count and accuracy for one applicable metric."""

    correct: int
    count: int

    @property
    def accuracy(self) -> float | None:
        """Return accuracy or ``None`` when no case is applicable."""

        return self.correct / self.count if self.count else None

    def to_record(self) -> dict[str, object]:
        """Serialize metric denominator as well as its rate."""

        return {
            "accuracy": self.accuracy,
            "correct": self.correct,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class ModelEvaluationMetrics:
    """Aggregate answer-quality dimensions for one model condition."""

    overall: MetricScore
    content: MetricScore
    refusal: MetricScore
    numeric: MetricScore
    units: MetricScore
    citations: MetricScore

    def to_record(self) -> dict[str, object]:
        """Serialize all aggregate diagnostic metrics."""

        return {
            "citations": self.citations.to_record(),
            "content": self.content.to_record(),
            "numeric": self.numeric.to_record(),
            "overall": self.overall.to_record(),
            "refusal": self.refusal.to_record(),
            "units": self.units.to_record(),
        }


@dataclass(frozen=True, slots=True)
class RAGEvaluationResult:
    """Artifacts and in-memory results from a complete paired evaluation."""

    predictions_path: Path
    metrics_path: Path
    comparisons: tuple[RAGCaseComparison, ...]
    base_metrics: ModelEvaluationMetrics
    adapter_metrics: ModelEvaluationMetrics
    suite: FrozenRAGSuite


def load_frozen_rag_suite(path: Path) -> FrozenRAGSuite:
    """Read and validate a versioned frozen-context JSON suite."""

    try:
        content = path.resolve().read_bytes()
        payload: object = json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        raise RAGEvaluationError(f"Could not read RAG evaluation suite: {path}.") from exc

    root = _require_mapping(payload, label="RAG evaluation suite")
    schema_version = _require_int(root, "schema_version")

    if schema_version != RAG_EVALUATION_SCHEMA_VERSION:
        raise RAGEvaluationError(f"Unsupported RAG evaluation schema_version: {schema_version}.")

    case_payloads = _require_list(root, "cases")
    cases = tuple(
        _parse_case(case_payload, index=index) for index, case_payload in enumerate(case_payloads)
    )
    return FrozenRAGSuite(
        name=_require_str(root, "name"),
        document_sha256=_require_str(root, "document_sha256"),
        retrieval_config=tuple(
            sorted(
                _require_mapping(
                    root.get("retrieval_config"),
                    label="retrieval_config",
                ).items()
            )
        ),
        cases=cases,
        suite_sha256=sha256(content).hexdigest(),
    )


def run_frozen_rag_evaluation(
    *,
    suite_path: Path,
    output_directory: Path,
    generator: PairedAnswerGenerator,
    config: RAGConfig,
) -> RAGEvaluationResult:
    """Generate paired answers, score them, and write deterministic artifacts."""

    suite = load_frozen_rag_suite(suite_path)
    comparisons = tuple(_evaluate_case(case, generator=generator) for case in suite.cases)
    base_metrics = aggregate_assessments(comparison.base for comparison in comparisons)
    adapter_metrics = aggregate_assessments(comparison.adapter for comparison in comparisons)
    output_root = output_directory.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / RAG_PREDICTIONS_FILENAME
    metrics_path = output_root / RAG_METRICS_FILENAME
    write_bytes_atomically(predictions_path, _serialize_comparisons(comparisons))
    write_bytes_atomically(
        metrics_path,
        _serialize_json(
            _build_metrics_payload(
                suite=suite,
                comparisons=comparisons,
                base_metrics=base_metrics,
                adapter_metrics=adapter_metrics,
                config=config,
            )
        ),
    )
    return RAGEvaluationResult(
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        comparisons=comparisons,
        base_metrics=base_metrics,
        adapter_metrics=adapter_metrics,
        suite=suite,
    )


def assess_answer(
    answer: str,
    *,
    expectation: RAGCaseExpectation,
    context_count: int,
) -> AnswerAssessment:
    """Score content, numeric values, units, citations, and refusal separately."""

    normalized_answer = answer.strip()

    if not normalized_answer:
        raise RAGEvaluationError("Generated answers must not be empty.")

    refused = normalized_answer == UNSUPPORTED_ANSWER
    refusal_correct = refused if not expectation.answerable else not refused
    expected_numbers = tuple(
        _parse_decimal(value, label="Expected numeric value")
        for value in expectation.expected_values
    )
    answer_without_source_labels = _SOURCE_PATTERN.sub("", normalized_answer)
    answer_numbers = _answer_numbers(answer_without_source_labels)
    numeric_correct = (
        all(expected in answer_numbers for expected in expected_numbers)
        if expected_numbers
        else None
    )
    normalized_text = _normalize_text(normalized_answer)
    units_correct = (
        all(_normalize_text(unit) in normalized_text for unit in expectation.expected_units)
        if expectation.expected_units
        else None
    )
    phrases_correct = all(
        _contains_normalized_phrase(normalized_text, phrase)
        for phrase in expectation.expected_phrases
    )
    cited_sources = tuple(
        dict.fromkeys(int(match.group(1)) for match in _SOURCE_PATTERN.finditer(normalized_answer))
    )
    citations_correct = (
        set(expectation.required_source_numbers).issubset(cited_sources)
        and all(1 <= source_number <= context_count for source_number in cited_sources)
        if expectation.required_source_numbers
        else None
    )
    content_correct = (
        refusal_correct
        if not expectation.answerable
        else phrases_correct and (numeric_correct is not False)
    )
    applicable_checks = (
        content_correct,
        refusal_correct,
        *(
            check
            for check in (numeric_correct, units_correct, citations_correct)
            if check is not None
        ),
    )
    return AnswerAssessment(
        answer=normalized_answer,
        content_correct=content_correct,
        refusal_correct=refusal_correct,
        numeric_correct=numeric_correct,
        units_correct=units_correct,
        citations_correct=citations_correct,
        cited_source_numbers=cited_sources,
        overall_pass=all(applicable_checks),
    )


def aggregate_assessments(assessments: Iterable[AnswerAssessment]) -> ModelEvaluationMetrics:
    """Aggregate answer checks while retaining every metric denominator."""

    materialized = tuple(assessments)
    return ModelEvaluationMetrics(
        overall=_score_required(materialized, attribute="overall_pass"),
        content=_score_required(materialized, attribute="content_correct"),
        refusal=_score_required(materialized, attribute="refusal_correct"),
        numeric=_score_optional(materialized, attribute="numeric_correct"),
        units=_score_optional(materialized, attribute="units_correct"),
        citations=_score_optional(materialized, attribute="citations_correct"),
    )


def _evaluate_case(
    case: FrozenRAGCase,
    *,
    generator: PairedAnswerGenerator,
) -> RAGCaseComparison:
    prompt = build_grounded_prompt(question=case.question, results=case.retrieved)
    prompt_sha256 = _hash_prompt(prompt)
    base_answer, adapter_answer = generator.generate_pair(prompt)
    return RAGCaseComparison(
        case=case,
        prompt=prompt,
        prompt_sha256=prompt_sha256,
        base=assess_answer(
            base_answer,
            expectation=case.expectation,
            context_count=len(case.retrieved),
        ),
        adapter=assess_answer(
            adapter_answer,
            expectation=case.expectation,
            context_count=len(case.retrieved),
        ),
    )


def _build_metrics_payload(
    *,
    suite: FrozenRAGSuite,
    comparisons: tuple[RAGCaseComparison, ...],
    base_metrics: ModelEvaluationMetrics,
    adapter_metrics: ModelEvaluationMetrics,
    config: RAGConfig,
) -> dict[str, object]:
    return {
        "evaluation_config": {
            "adapter_model_id": config.adapter_model_id,
            "adapter_model_revision": config.adapter_model_revision,
            "base_model_id": config.base_model_id,
            "base_model_revision": config.base_model_revision,
            "do_sample": False,
            "generation_device_map": config.generation_device_map,
            "max_input_tokens": config.max_input_tokens,
            "max_new_tokens": config.max_new_tokens,
            "system_instruction_sha256": sha256(SYSTEM_INSTRUCTION.encode("utf-8")).hexdigest(),
        },
        "metrics": {
            "adapter": adapter_metrics.to_record(),
            "base": base_metrics.to_record(),
            "delta_adapter_minus_base": _metric_deltas(base_metrics, adapter_metrics),
        },
        "metrics_by_category": {
            category: {
                "adapter": aggregate_assessments(
                    comparison.adapter
                    for comparison in comparisons
                    if comparison.case.category == category
                ).to_record(),
                "base": aggregate_assessments(
                    comparison.base
                    for comparison in comparisons
                    if comparison.case.category == category
                ).to_record(),
            }
            for category in _CASE_CATEGORIES
            if any(comparison.case.category == category for comparison in comparisons)
        },
        "metrics_by_context_mode": {
            context_mode: {
                "adapter": aggregate_assessments(
                    comparison.adapter
                    for comparison in comparisons
                    if comparison.case.context_mode == context_mode
                ).to_record(),
                "base": aggregate_assessments(
                    comparison.base
                    for comparison in comparisons
                    if comparison.case.context_mode == context_mode
                ).to_record(),
            }
            for context_mode in _CONTEXT_MODES
            if any(comparison.case.context_mode == context_mode for comparison in comparisons)
        },
        "schema_version": RAG_EVALUATION_SCHEMA_VERSION,
        "suite": {
            "case_count": len(suite.cases),
            "document_sha256": suite.document_sha256,
            "name": suite.name,
            "retrieval_config": dict(suite.retrieval_config),
            "suite_sha256": suite.suite_sha256,
        },
    }


def _metric_deltas(
    base: ModelEvaluationMetrics,
    adapter: ModelEvaluationMetrics,
) -> dict[str, float | None]:
    return {
        name: _accuracy_delta(getattr(base, name), getattr(adapter, name))
        for name in ("overall", "content", "refusal", "numeric", "units", "citations")
    }


def _accuracy_delta(base: MetricScore, adapter: MetricScore) -> float | None:
    if base.accuracy is None or adapter.accuracy is None:
        return None

    return adapter.accuracy - base.accuracy


def _score_required(
    assessments: tuple[AnswerAssessment, ...],
    *,
    attribute: Literal["overall_pass", "content_correct", "refusal_correct"],
) -> MetricScore:
    values = tuple(cast(bool, getattr(assessment, attribute)) for assessment in assessments)
    return MetricScore(correct=sum(values), count=len(values))


def _score_optional(
    assessments: tuple[AnswerAssessment, ...],
    *,
    attribute: Literal["numeric_correct", "units_correct", "citations_correct"],
) -> MetricScore:
    values = tuple(
        value
        for assessment in assessments
        if (value := cast(bool | None, getattr(assessment, attribute))) is not None
    )
    return MetricScore(correct=sum(values), count=len(values))


def _hash_prompt(prompt: GroundedPrompt) -> str:
    serialized = json.dumps(
        prompt.to_messages(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(serialized).hexdigest()


def _serialize_comparisons(comparisons: Iterable[RAGCaseComparison]) -> bytes:
    return b"".join(
        (
            json.dumps(
                comparison.to_record(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for comparison in comparisons
    )


def _serialize_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _parse_case(payload: object, *, index: int) -> FrozenRAGCase:
    record = _require_mapping(payload, label=f"cases[{index}]")
    category_text = _require_str(record, "category")
    context_mode = _require_str(record, "context_mode")

    if category_text not in _CASE_CATEGORIES:
        raise RAGEvaluationError(f"cases[{index}].category is unsupported: {category_text!r}.")

    expectation = _parse_expectation(
        _require_mapping(record.get("expectation"), label="expectation")
    )

    if context_mode not in _CONTEXT_MODES:
        raise RAGEvaluationError(f"cases[{index}].context_mode is unsupported: {context_mode!r}.")
    retrieved_payloads = _require_list(record, "retrieved")
    retrieved = tuple(
        _parse_retrieval_result(result_payload, index=result_index)
        for result_index, result_payload in enumerate(retrieved_payloads)
    )
    return FrozenRAGCase(
        case_id=_require_str(record, "case_id"),
        category=category_text,
        context_mode=context_mode,
        production_gold_rank=_optional_int(record, "production_gold_rank"),
        question=_require_str(record, "question"),
        expectation=expectation,
        retrieved=retrieved,
    )


def _parse_expectation(record: Mapping[str, object]) -> RAGCaseExpectation:
    return RAGCaseExpectation(
        answerable=_require_bool(record, "answerable"),
        expected_values=_optional_str_tuple(record, "expected_values"),
        expected_units=_optional_str_tuple(record, "expected_units"),
        expected_phrases=_optional_str_tuple(record, "expected_phrases"),
        required_source_numbers=_optional_int_tuple(record, "required_source_numbers"),
    )


def _parse_retrieval_result(payload: object, *, index: int) -> RetrievalResult:
    record = _require_mapping(payload, label=f"retrieved[{index}]")
    chunk_record = _require_mapping(record.get("chunk"), label=f"retrieved[{index}].chunk")
    return RetrievalResult(
        chunk=_parse_chunk(chunk_record),
        score=_require_number(record, "score"),
        rank=_require_int(record, "rank"),
    )


def _parse_chunk(record: Mapping[str, object]) -> DocumentChunk:
    kind_text = _optional_str(record, "retrieval_unit_kind") or "chunk"

    if kind_text not in {"chunk", "narrative", "table_row"}:
        raise RAGEvaluationError(f"Unsupported retrieval_unit_kind: {kind_text!r}.")

    return DocumentChunk(
        chunk_id=_require_str(record, "chunk_id"),
        document_id=_require_str(record, "document_id"),
        document_sha256=_require_str(record, "document_sha256"),
        filename=_require_str(record, "filename"),
        chunk_index=_require_int(record, "chunk_index"),
        page_start=_require_int(record, "page_start"),
        page_end=_require_int(record, "page_end"),
        source_element_ids=_required_str_tuple(record, "source_element_ids"),
        text=_require_str(record, "text"),
        char_count=_require_int(record, "char_count"),
        token_count=_require_int(record, "token_count"),
        block_count=_require_int(record, "block_count"),
        retrieval_unit_kind=cast(RetrievalUnitKind, kind_text),
        parent_chunk_id=_optional_str(record, "parent_chunk_id"),
        parent_source_element_ids=_optional_str_tuple(record, "parent_source_element_ids"),
    )


def _parse_decimal(value: str, *, label: str) -> Decimal:
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation as exc:
        raise RAGEvaluationError(f"{label} is not a valid decimal: {value!r}.") from exc


def _normalize_text(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", value).strip().casefold()


def _answer_numbers(value: str) -> tuple[Decimal, ...]:
    """Extract numbers from the model's explicit or trailing calculation result.

    When the model writes ``the answer is``, operands before that marker cannot satisfy
    the expected value. An equals sign receives the same treatment for worked arithmetic.
    Free-form scalar answers without either marker retain all of their numbers so ordinary
    sentences such as ``Revenue was $14.1 million`` remain scoreable.
    """

    marker_matches = tuple(_ANSWER_MARKER_PATTERN.finditer(value))

    if marker_matches:
        candidate = value[marker_matches[-1].end() :]
    elif "=" in value:
        candidate = value.rsplit("=", 1)[1]
    else:
        candidate = value

    return tuple(
        _parse_decimal(match.group(0), label="Generated numeric value")
        for match in _NUMBER_PATTERN.finditer(candidate)
    )


def _contains_normalized_phrase(normalized_text: str, phrase: str) -> bool:
    """Match a required phrase without accepting substrings inside larger words."""

    normalized_phrase = _normalize_text(phrase)
    return (
        re.search(
            rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)",
            normalized_text,
        )
        is not None
    )


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RAGEvaluationError(f"{label} must be a JSON object.")

    return cast(dict[str, object], value)


def _require_list(record: Mapping[str, object], key: str) -> list[object]:
    value = record.get(key)

    if not isinstance(value, list):
        raise RAGEvaluationError(f"{key} must be a JSON array.")

    return cast(list[object], value)


def _require_str(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)

    if not isinstance(value, str) or not value.strip():
        raise RAGEvaluationError(f"{key} must be a non-empty string.")

    return value


def _optional_str(record: Mapping[str, object], key: str) -> str | None:
    value = record.get(key)

    if value is None:
        return None

    if not isinstance(value, str) or not value.strip():
        raise RAGEvaluationError(f"{key} must be a non-empty string when provided.")

    return value


def _require_bool(record: Mapping[str, object], key: str) -> bool:
    value = record.get(key)

    if not isinstance(value, bool):
        raise RAGEvaluationError(f"{key} must be a boolean.")

    return value


def _require_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)

    if not isinstance(value, int) or isinstance(value, bool):
        raise RAGEvaluationError(f"{key} must be an integer.")

    return value


def _optional_int(record: Mapping[str, object], key: str) -> int | None:
    value = record.get(key)

    if value is None:
        return None

    if not isinstance(value, int) or isinstance(value, bool):
        raise RAGEvaluationError(f"{key} must be an integer when provided.")

    return value


def _require_number(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RAGEvaluationError(f"{key} must be numeric.")

    return float(value)


def _required_str_tuple(record: Mapping[str, object], key: str) -> tuple[str, ...]:
    values = _optional_str_tuple(record, key)

    if not values:
        raise RAGEvaluationError(f"{key} must contain at least one string.")

    return values


def _optional_str_tuple(record: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = record.get(key)

    if value is None:
        return ()

    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise RAGEvaluationError(f"{key} must be an array of non-empty strings.")

    return tuple(cast(list[str], value))


def _optional_int_tuple(record: Mapping[str, object], key: str) -> tuple[int, ...]:
    value = record.get(key)

    if value is None:
        return ()

    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise RAGEvaluationError(f"{key} must be an array of integers.")

    return tuple(cast(list[int], value))
