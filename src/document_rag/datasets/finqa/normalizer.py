import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import JsonValue

from document_rag.datasets.finqa.raw_models import FinQARawRecord
from document_rag.datasets.models import (
    DatasetExample,
    DatasetName,
    DatasetSplit,
    ReasoningStep,
    ReferenceAnswer,
    SupportingFact,
)
from document_rag.domain import (
    Document,
    DocumentElement,
    DocumentElementType,
    Question,
    TableCoordinates,
)

_PAGE_PATTERN = re.compile(r"page_(\d+)\.pdf$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class NormalizedFinQARecord:
    """Normalized representation of one FinQA example."""

    document: Document
    elements: tuple[DocumentElement, ...]
    example: DatasetExample


def _reference_answer_text(record: FinQARawRecord) -> str:
    answer = _optional_stringify(record.qa.answer)

    if answer is not None:
        return answer

    return _stringify(
        record.qa.exe_ans,
        field_name="exe_ans",
    )


def _optional_stringify(value: JsonValue) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def normalize_finqa_record(
    record: FinQARawRecord,
    *,
    split: DatasetSplit,
) -> NormalizedFinQARecord:
    """Convert one raw FinQA record into internal project models."""

    document_id = f"finqa:{record.filename}"
    question_id = f"finqa:{record.id}"
    page_number = _extract_page_number(record.filename)

    document = _build_document(
        record,
        document_id=document_id,
    )

    elements, source_key_to_element_id = _build_elements(
        record,
        document_id=document_id,
        page_number=page_number,
    )

    supporting_facts = _build_supporting_facts(
        record,
        source_key_to_element_id=source_key_to_element_id,
    )

    question = Question(
        question_id=question_id,
        document_id=document_id,
        text=record.qa.question,
        metadata={
            "dataset": DatasetName.FINQA.value,
            "source_example_id": record.id,
        },
    )

    reference_answer = ReferenceAnswer(
        text=_reference_answer_text(record),
        executable_answer=record.qa.exe_ans,
        explanation=_optional_string(record.qa.explanation),
        program=_optional_string(record.qa.program),
        normalized_program=_optional_string(record.qa.program_re),
        steps=tuple(
            ReasoningStep(
                operation=step.op,
                arguments=(
                    _stringify(step.arg1, field_name="step.arg1"),
                    _stringify(step.arg2, field_name="step.arg2"),
                ),
                result=_stringify(step.res, field_name="step.res"),
            )
            for step in record.qa.steps
        ),
    )

    example = DatasetExample(
        dataset=DatasetName.FINQA,
        split=split,
        example_id=question_id,
        question=question,
        reference_answer=reference_answer,
        supporting_facts=supporting_facts,
    )

    return NormalizedFinQARecord(
        document=document,
        elements=elements,
        example=example,
    )


def _build_document(
    record: FinQARawRecord,
    *,
    document_id: str,
) -> Document:
    source_path = PurePosixPath(record.filename)
    metadata: dict[str, JsonValue] = {
        "dataset": DatasetName.FINQA.value,
    }

    if len(source_path.parts) >= 3:
        metadata["company"] = source_path.parts[0]
        metadata["report_year"] = source_path.parts[1]

    return Document(
        document_id=document_id,
        file_name=source_path.name,
        source_uri=record.filename,
        page_count=1,
        metadata=metadata,
    )


def _build_elements(
    record: FinQARawRecord,
    *,
    document_id: str,
    page_number: int,
) -> tuple[tuple[DocumentElement, ...], dict[str, str]]:
    elements: list[DocumentElement] = []
    source_key_to_element_id: dict[str, str] = {}

    combined_text = (*record.pre_text, *record.post_text)

    for index, source_text in enumerate(combined_text):
        normalized_text = source_text.strip()

        if not normalized_text:
            continue

        source_key = f"text_{index}"
        element_id = _element_id(document_id, source_key)
        source_region = "pre_text" if index < len(record.pre_text) else "post_text"

        elements.append(
            DocumentElement(
                element_id=element_id,
                document_id=document_id,
                element_type=DocumentElementType.PARAGRAPH,
                source_text=normalized_text,
                page_number=page_number,
                metadata={
                    "dataset": DatasetName.FINQA.value,
                    "source_key": source_key,
                    "source_region": source_region,
                },
            )
        )
        source_key_to_element_id[source_key] = element_id

    table_id = _element_id(document_id, "table")

    elements.append(
        DocumentElement(
            element_id=table_id,
            document_id=document_id,
            element_type=DocumentElementType.TABLE,
            source_text=_format_table(record.table),
            page_number=page_number,
            table_coordinates=TableCoordinates(table_id=table_id),
            metadata={
                "dataset": DatasetName.FINQA.value,
                "original_table": [list(row) for row in record.table_ori],
            },
        )
    )

    column_headers: list[JsonValue] = list(record.table[0]) if record.table else []

    for row_index, row in enumerate(record.table):
        source_key = f"table_{row_index}"
        element_id = _element_id(document_id, source_key)

        elements.append(
            DocumentElement(
                element_id=element_id,
                document_id=document_id,
                element_type=DocumentElementType.TABLE_ROW,
                source_text=_format_table_row(row),
                page_number=page_number,
                parent_element_id=table_id,
                table_coordinates=TableCoordinates(
                    table_id=table_id,
                    row_index=row_index,
                ),
                metadata={
                    "dataset": DatasetName.FINQA.value,
                    "source_key": source_key,
                    "column_headers": column_headers,
                },
            )
        )
        source_key_to_element_id[source_key] = element_id

    return tuple(elements), source_key_to_element_id


def _build_supporting_facts(
    record: FinQARawRecord,
    *,
    source_key_to_element_id: dict[str, str],
) -> tuple[SupportingFact, ...]:
    supporting_facts: list[SupportingFact] = []

    for source_key in record.qa.gold_inds:
        element_id = source_key_to_element_id.get(source_key)

        if element_id is None:
            raise ValueError(
                f"Supporting fact {source_key!r} does not reference "
                f"a document element for example {record.id!r}"
            )

        supporting_facts.append(
            SupportingFact(
                source_key=source_key,
                element_id=element_id,
            )
        )

    return tuple(supporting_facts)


def _extract_page_number(filename: str) -> int:
    match = _PAGE_PATTERN.search(filename)

    if match is None:
        raise ValueError(f"Cannot extract page number from {filename!r}")

    return int(match.group(1))


def _element_id(document_id: str, source_key: str) -> str:
    return f"{document_id}:{source_key}"


def _format_table(table: tuple[tuple[str, ...], ...]) -> str:
    return "\n".join(_format_table_row(row) for row in table)


def _format_table_row(row: tuple[str, ...]) -> str:
    return " | ".join(cell.strip() for cell in row)


def _stringify(value: JsonValue, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} cannot be null")

    text = str(value).strip()

    if not text:
        raise ValueError(f"{field_name} cannot be empty")

    return text


def _optional_string(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None
