"""Offline tests for production-shaped RAG adapter training export."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from document_rag.datasets.models import (
    DatasetExample,
    DatasetName,
    DatasetSplit,
    ReferenceAnswer,
    SupportingFact,
)
from document_rag.domain.documents import (
    Document,
    DocumentElement,
    DocumentElementType,
    TableCoordinates,
)
from document_rag.domain.questions import Question
from document_rag.ingestion.chunking import DocumentChunk
from document_rag.rag.config import RAGConfig
from document_rag.rag.prompting import SYSTEM_INSTRUCTION, UNSUPPORTED_ANSWER
from document_rag.retrieval.models import RetrievalResult
from document_rag.training.rag_export import (
    HuggingFaceTrainingTokenCounter,
    RAGTrainingExportConfig,
    _assigned_document_split,
    export_rag_training_data,
)


@dataclass(frozen=True)
class _FakeTokenCounter:
    """Count characters deterministically without loading a tokenizer."""

    def count(self, messages: Sequence[Mapping[str, str]]) -> int:
        return sum(len(message["content"]) for message in messages)


@dataclass(frozen=True)
class _SourceTokenCounter:
    """Treat each rendered source or citation label as one synthetic token."""

    overflow_phrase: str | None = None

    def count(self, messages: Sequence[Mapping[str, str]]) -> int:
        if self.overflow_phrase is not None and self.overflow_phrase in messages[1]["content"]:
            return 4

        return sum(message["content"].count("[Source ") for message in messages)


@dataclass(frozen=True)
class _FakeFactory:
    """Build deterministic rankings without embeddings, Internet, or GPU."""

    def build(self, chunks: tuple[DocumentChunk, ...]) -> _FakeRetriever:
        return _FakeRetriever(chunks)


@dataclass(frozen=True)
class _FakeRetriever:
    chunks: tuple[DocumentChunk, ...]

    def search(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        del query
        ordered = sorted(
            self.chunks,
            key=lambda chunk: (
                0 if "high-ranking distractor" in chunk.text.casefold() else 1,
                chunk.chunk_id,
            ),
        )[:top_k]
        return tuple(
            RetrievalResult(chunk=chunk, score=1.0 / rank, rank=rank)
            for rank, chunk in enumerate(ordered, start=1)
        )


def test_export_builds_supported_oracle_and_refusal_examples(tmp_path: Path) -> None:
    """Targets should contain units/citations and insufficient contexts should refuse."""

    finqa, docfinqa = _write_training_fixtures(tmp_path)
    result = export_rag_training_data(
        finqa_directory=finqa,
        docfinqa_directory=docfinqa,
        output_directory=tmp_path / "output",
        splits=(DatasetSplit.TRAIN,),
        rag_config=RAGConfig(top_k=1, candidate_k=2),
        export_config=RAGTrainingExportConfig(
            refusal_ratio=0.2,
            document_resplit=False,
        ),
        retriever_factory=_FakeFactory(),
        token_counter=_FakeTokenCounter(),
    )

    artifact = result.artifacts[0]
    records = _read_jsonl(artifact.path)

    assert artifact.record_count == 5
    assert artifact.supported_count == 4
    assert artifact.refusal_count == 1
    assert artifact.calculation_supervised_count == 1
    assert artifact.oracle_augmented_count == 1
    assert artifact.ambiguous_unit_exclusion_count == 1
    assert all(record["messages"][0]["content"] == SYSTEM_INSTRUCTION for record in records)

    calculation = _record(records, "finqa:calculation:supported")
    assert calculation["category"] == "reasoning"
    assert calculation["context_mode"] == "oracle_augmented"
    assert calculation["expected_units"] == ["$", "million"]
    assert calculation["calculation_supervised"] is True
    assert calculation["reasoning_program"] == "subtract(120, 100)"
    assert calculation["messages"][2]["content"] == (
        "Calculation:\nsubtract(120, 100)\nThe answer is $20 million. [Source 1]"
    )
    assert "Retrieved context:" in calculation["messages"][1]["content"]
    assert "chunk_id training-chunk:" in calculation["messages"][1]["content"]

    percentage = _record(records, "finqa:percentage:supported")
    assert percentage["calculation_supervised"] is False
    assert percentage["unit_status"] == "preserved"
    assert percentage["expected_units"] == ["%"]
    assert "25%" in percentage["messages"][2]["content"]

    table_lookup = _record(records, "finqa:table-lookup:supported")
    assert table_lookup["category"] == "table_lookup"
    assert table_lookup["expected_units"] == ["$"]
    assert "Table: Financial table" in table_lookup["messages"][1]["content"]
    assert "Quarter: 2024 Q1" in table_lookup["messages"][1]["content"]
    assert "$220,000" in table_lookup["messages"][2]["content"]

    refusal = _record(records, "finqa:calculation:unsupported")
    assert refusal["answerable"] is False
    assert refusal["calculation_supervised"] is False
    assert refusal["messages"][2]["content"] == UNSUPPORTED_ANSWER
    assert refusal["gold_source_numbers"] == []
    assert "high-ranking distractor" in refusal["messages"][1]["content"].casefold()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 4
    assert manifest["policies"]["reasoning_supervision"] == (
        "visible_normalized_calculation_program_then_final_answer"
    )
    assert manifest["policies"]["unsupported_target"] == UNSUPPORTED_ANSWER
    assert manifest["retrieval"]["top_k"] == 1
    assert manifest["artifacts"]["train"]["calculation_supervised_count"] == 1
    assert manifest["artifacts"]["train"]["context_trimmed_count"] == 0
    assert manifest["artifacts"]["train"]["sequence_overflow_exclusion_count"] == 0
    assert manifest["tokenization"] == {
        "add_generation_prompt": False,
        "enable_thinking": False,
        "max_sequence_tokens": 4096,
        "model_id": "Qwen/Qwen3-1.7B",
        "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    }
    assert manifest["artifacts"]["train"]["category_counts"] == {
        "reasoning": 1,
        "simple_lookup": 2,
        "table_lookup": 1,
        "unsupported": 1,
    }
    assert manifest["artifacts"]["train"]["context_mode_counts"] == {
        "oracle_augmented": 1,
        "retrieval_insufficient": 1,
        "retrieved": 3,
    }


def test_rag_export_is_deterministic(tmp_path: Path) -> None:
    """Identical normalized inputs and policy must produce byte-identical artifacts."""

    finqa, docfinqa = _write_training_fixtures(tmp_path)
    rag_config = RAGConfig(top_k=1, candidate_k=2)
    export_config = RAGTrainingExportConfig(
        refusal_ratio=0.2,
        document_resplit=False,
    )
    first = export_rag_training_data(
        finqa_directory=finqa,
        docfinqa_directory=docfinqa,
        output_directory=tmp_path / "first",
        splits=(DatasetSplit.TRAIN,),
        rag_config=rag_config,
        export_config=export_config,
        retriever_factory=_FakeFactory(),
        token_counter=_FakeTokenCounter(),
    )
    second = export_rag_training_data(
        finqa_directory=finqa,
        docfinqa_directory=docfinqa,
        output_directory=tmp_path / "second",
        splits=(DatasetSplit.TRAIN,),
        rag_config=rag_config,
        export_config=export_config,
        retriever_factory=_FakeFactory(),
        token_counter=_FakeTokenCounter(),
    )

    assert first.artifacts[0].sha256 == second.artifacts[0].sha256
    assert first.artifacts[0].path.read_bytes() == second.artifacts[0].path.read_bytes()
    assert first.manifest_sha256 == second.manifest_sha256


def test_source_report_groups_cannot_cross_splits(tmp_path: Path) -> None:
    """Questions from one report must not leak between training and evaluation splits."""

    finqa = tmp_path / "finqa"
    docfinqa = tmp_path / "docfinqa"

    for root, dataset in (
        (finqa, DatasetName.FINQA),
        (docfinqa, DatasetName.DOCFINQA),
    ):
        _write_manifest(root, dataset)

        for split in (DatasetSplit.TRAIN, DatasetSplit.TEST):
            document_id = f"{dataset.value}:{split.value}"
            document = Document(
                document_id=document_id,
                file_name="page_1.pdf",
                source_uri="ACME/2025/page_1.pdf",
            )
            element = _paragraph(
                document_id,
                "evidence",
                "Revenue was $10 million.",
            )
            example = _example(
                dataset=dataset,
                split=split,
                example_id=f"{dataset.value}:{split.value}",
                document_id=document_id,
                question="What was revenue in millions of dollars?",
                answer="10",
                supporting_element_id=element.element_id,
            )
            _write_split(root, split, (document,), (element,), (example,))

    with pytest.raises(ValueError, match="Source-report groups overlap"):
        export_rag_training_data(
            finqa_directory=finqa,
            docfinqa_directory=docfinqa,
            output_directory=tmp_path / "output",
            splits=(DatasetSplit.TRAIN, DatasetSplit.TEST),
            rag_config=RAGConfig(top_k=1, candidate_k=1),
            export_config=RAGTrainingExportConfig(document_resplit=False),
            retriever_factory=_FakeFactory(),
            token_counter=_FakeTokenCounter(),
        )


def test_document_resplit_is_deterministic_and_uses_all_output_splits() -> None:
    """Stable report identities should receive one reproducible 80/10/10 assignment."""

    group_ids = tuple(f"source-group:report-{index}" for index in range(100))
    first = tuple(_assigned_document_split(group_id) for group_id in group_ids)
    second = tuple(_assigned_document_split(group_id) for group_id in group_ids)

    assert first == second
    assert set(first) == set(DatasetSplit)


def test_export_trims_only_non_gold_context_to_fit_sequence_budget(tmp_path: Path) -> None:
    """Oversized context should lose a low-ranked distractor, never its gold row."""

    finqa, docfinqa = _write_training_fixtures(tmp_path)
    result = export_rag_training_data(
        finqa_directory=finqa,
        docfinqa_directory=docfinqa,
        output_directory=tmp_path / "output",
        splits=(DatasetSplit.TRAIN,),
        rag_config=RAGConfig(top_k=2, candidate_k=2),
        export_config=RAGTrainingExportConfig(
            refusal_ratio=0.2,
            document_resplit=False,
            max_sequence_tokens=3,
        ),
        retriever_factory=_FakeFactory(),
        token_counter=_SourceTokenCounter(),
    )

    records = _read_jsonl(result.artifacts[0].path)
    calculation = _record(records, "finqa:calculation:supported")

    assert calculation["context_trimmed"] is True
    assert calculation["sequence_token_count"] == 3
    assert calculation["gold_source_element_ids"] == ["finqa:calculation:gold"]
    assert calculation["gold_source_numbers"] == [1]
    assert len(calculation["source_chunk_ids"]) == 1
    assert "high-ranking distractor" not in calculation["messages"][1]["content"].casefold()
    assert result.artifacts[0].context_trimmed_count == 1
    assert result.artifacts[0].sequence_overflow_exclusion_count == 0


def test_export_audits_gold_only_sequence_overflow(tmp_path: Path) -> None:
    """A gold context that cannot fit must be counted instead of silently filtered later."""

    finqa, docfinqa = _write_training_fixtures(tmp_path)
    result = export_rag_training_data(
        finqa_directory=finqa,
        docfinqa_directory=docfinqa,
        output_directory=tmp_path / "output",
        splits=(DatasetSplit.TRAIN,),
        rag_config=RAGConfig(top_k=1, candidate_k=2),
        export_config=RAGTrainingExportConfig(
            refusal_ratio=0.2,
            document_resplit=False,
            max_sequence_tokens=3,
        ),
        retriever_factory=_FakeFactory(),
        token_counter=_SourceTokenCounter(overflow_phrase="Gross margin"),
    )

    records = _read_jsonl(result.artifacts[0].path)

    assert not any(record["example_id"] == "finqa:percentage:supported" for record in records)
    assert result.artifacts[0].sequence_overflow_exclusion_count == 1
    assert result.artifacts[0].max_sequence_token_count <= 3


def test_hugging_face_counter_uses_training_chat_template() -> None:
    """Token counting must match the non-thinking, target-inclusive Kaggle sequence."""

    class _FakeTokenizer:
        def apply_chat_template(self, messages: object, **options: object) -> tuple[int, ...]:
            assert messages == [{"role": "assistant", "content": "answer"}]
            assert options == {
                "tokenize": True,
                "add_generation_prompt": False,
                "enable_thinking": False,
                "return_dict": False,
            }
            return (1, 2, 3)

    counter = HuggingFaceTrainingTokenCounter(tokenizer=_FakeTokenizer())

    assert counter.count(({"role": "assistant", "content": "answer"},)) == 3


@pytest.mark.parametrize("ratio", [-0.1, 0.5, 1.0])
def test_refusal_ratio_is_bounded(ratio: float) -> None:
    """Refusal supervision must remain a minority of generated records."""

    with pytest.raises(ValueError, match="refusal_ratio"):
        RAGTrainingExportConfig(refusal_ratio=ratio)


def test_max_sequence_tokens_must_be_positive() -> None:
    """The exporter cannot enforce an empty or negative sequence budget."""

    with pytest.raises(ValueError, match="max_sequence_tokens"):
        RAGTrainingExportConfig(max_sequence_tokens=0)


def _write_training_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    finqa = tmp_path / "finqa"
    docfinqa = tmp_path / "docfinqa"
    _write_manifest(finqa, DatasetName.FINQA)
    _write_manifest(docfinqa, DatasetName.DOCFINQA)

    finqa_documents = (
        Document(
            document_id="finqa:calculation-document",
            file_name="page_1.pdf",
            source_uri="ACME/2024/page_1.pdf",
        ),
        Document(
            document_id="finqa:percentage-document",
            file_name="page_2.pdf",
            source_uri="BETA/2024/page_2.pdf",
        ),
        Document(
            document_id="finqa:ambiguous-document",
            file_name="page_3.pdf",
            source_uri="GAMMA/2024/page_3.pdf",
        ),
        Document(
            document_id="finqa:table-document",
            file_name="page_4.pdf",
            source_uri="OMEGA/2024/page_4.pdf",
        ),
    )
    table_id = "finqa:calculation:table"
    finqa_elements = (
        _paragraph(
            finqa_documents[0].document_id,
            "distractor",
            "High-ranking distractor about employee count.",
        ),
        DocumentElement(
            element_id=table_id,
            document_id=finqa_documents[0].document_id,
            element_type=DocumentElementType.TABLE,
            source_text="Metric | 2024 | 2025\nRevenue | $100 million | $120 million",
            page_number=1,
            table_coordinates=TableCoordinates(table_id=table_id),
        ),
        DocumentElement(
            element_id="finqa:calculation:gold",
            document_id=finqa_documents[0].document_id,
            element_type=DocumentElementType.TABLE_ROW,
            source_text="Revenue | $100 million | $120 million",
            page_number=1,
            parent_element_id=table_id,
            table_coordinates=TableCoordinates(table_id=table_id, row_index=1),
            metadata={"column_headers": ["Metric", "2024", "2025"]},
        ),
        _paragraph(
            finqa_documents[1].document_id,
            "gold",
            "Gross margin was 25%.",
        ),
        _paragraph(
            finqa_documents[2].document_id,
            "gold",
            "Interest expense was 3.8%.",
        ),
        DocumentElement(
            element_id="finqa:table-lookup:table",
            document_id=finqa_documents[3].document_id,
            element_type=DocumentElementType.TABLE,
            source_text="Quarter | Region | Revenue\n2024 Q1 | North America | $220,000",
            page_number=1,
            table_coordinates=TableCoordinates(table_id="finqa:table-lookup:table"),
        ),
        DocumentElement(
            element_id="finqa:table-lookup:gold",
            document_id=finqa_documents[3].document_id,
            element_type=DocumentElementType.TABLE_ROW,
            source_text="2024 Q1 | North America | $220,000",
            page_number=1,
            parent_element_id="finqa:table-lookup:table",
            table_coordinates=TableCoordinates(
                table_id="finqa:table-lookup:table",
                row_index=1,
            ),
            metadata={"column_headers": ["Quarter", "Region", "Revenue"]},
        ),
    )
    finqa_examples = (
        _example(
            dataset=DatasetName.FINQA,
            split=DatasetSplit.TRAIN,
            example_id="finqa:calculation",
            document_id=finqa_documents[0].document_id,
            question="By how many million dollars did revenue increase?",
            answer="20",
            supporting_element_id="finqa:calculation:gold",
            program="subtract(120, 100)",
        ),
        _example(
            dataset=DatasetName.FINQA,
            split=DatasetSplit.TRAIN,
            example_id="finqa:percentage",
            document_id=finqa_documents[1].document_id,
            question="What was gross margin?",
            answer="25%",
            supporting_element_id="finqa:percentage-document:gold",
        ),
        _example(
            dataset=DatasetName.FINQA,
            split=DatasetSplit.TRAIN,
            example_id="finqa:ambiguous",
            document_id=finqa_documents[2].document_id,
            question="What was interest expense?",
            answer="380",
            supporting_element_id="finqa:ambiguous-document:gold",
            program="answer = 380",
        ),
        _example(
            dataset=DatasetName.FINQA,
            split=DatasetSplit.TRAIN,
            example_id="finqa:table-lookup",
            document_id=finqa_documents[3].document_id,
            question="What was North America revenue in 2024 Q1?",
            answer="$220,000",
            supporting_element_id="finqa:table-lookup:gold",
        ),
    )
    _write_split(
        finqa,
        DatasetSplit.TRAIN,
        finqa_documents,
        finqa_elements,
        finqa_examples,
    )

    docfinqa_document = Document(
        document_id="docfinqa:cash-document",
        file_name="cash.txt",
        source_uri="DELTA/2024/report.txt",
    )
    docfinqa_element = _paragraph(
        docfinqa_document.document_id,
        "gold",
        "Cash at year end was $3.8 million.",
    )
    docfinqa_example = _example(
        dataset=DatasetName.DOCFINQA,
        split=DatasetSplit.TRAIN,
        example_id="docfinqa:cash",
        document_id=docfinqa_document.document_id,
        question="What was cash at year end?",
        answer="$3.8 million",
        supporting_element_id=docfinqa_element.element_id,
    )
    _write_split(
        docfinqa,
        DatasetSplit.TRAIN,
        (docfinqa_document,),
        (docfinqa_element,),
        (docfinqa_example,),
    )
    return finqa, docfinqa


def _example(
    *,
    dataset: DatasetName,
    split: DatasetSplit,
    example_id: str,
    document_id: str,
    question: str,
    answer: str,
    supporting_element_id: str,
    program: str | None = None,
) -> DatasetExample:
    return DatasetExample(
        dataset=dataset,
        split=split,
        example_id=example_id,
        question=Question(
            question_id=example_id,
            document_id=document_id,
            text=question,
        ),
        reference_answer=ReferenceAnswer(text=answer, program=program),
        supporting_facts=(
            SupportingFact(
                source_key="gold",
                element_id=supporting_element_id,
            ),
        ),
    )


def _paragraph(document_id: str, suffix: str, text: str) -> DocumentElement:
    return DocumentElement(
        element_id=f"{document_id}:{suffix}",
        document_id=document_id,
        element_type=DocumentElementType.PARAGRAPH,
        source_text=text,
        page_number=1,
    )


def _write_manifest(root: Path, dataset: DatasetName) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": dataset.value,
                "schema_version": "1",
                "source": {
                    "revision": "a" * 40,
                    "url": f"https://example.com/{dataset.value}",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_split(
    root: Path,
    split: DatasetSplit,
    documents: tuple[Document, ...],
    elements: tuple[DocumentElement, ...],
    examples: tuple[DatasetExample, ...],
) -> None:
    split_root = root / split.value
    _write_models(split_root / "documents.jsonl", documents)
    _write_models(split_root / "elements.jsonl", elements)
    _write_models(split_root / "examples.jsonl", examples)


def _write_models(path: Path, models: tuple[Any, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                model.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for model in models
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _record(records: list[dict[str, Any]], example_id: str) -> dict[str, Any]:
    return next(record for record in records if record["example_id"] == example_id)
