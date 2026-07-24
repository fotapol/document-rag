from collections.abc import Sequence

import pytest

from document_rag.datasets import DatasetSplit
from document_rag.datasets.finqa import (
    FinQARawRecord,
    prepare_finqa_split,
)


class StubFinQASource:
    def __init__(self, records: Sequence[FinQARawRecord]) -> None:
        self._records = records

    def read_split(
        self,
        split: DatasetSplit,
    ) -> Sequence[FinQARawRecord]:
        return self._records


def make_raw_record(
    *,
    example_id: str = "ABC/2025/page_10.pdf-1",
    question: str = "How much did revenue increase?",
    first_paragraph: str = "Revenue increased during the year.",
) -> FinQARawRecord:
    return FinQARawRecord.model_validate(
        {
            "pre_text": [
                first_paragraph,
                "Operating income was $120 million.",
            ],
            "post_text": ["Additional information follows."],
            "filename": "ABC/2025/page_10.pdf",
            "table_ori": [
                ["", "2024", "2025"],
                ["Revenue", "$100", "$120"],
            ],
            "table": [
                ["", "2024", "2025"],
                ["revenue", "$ 100", "$ 120"],
            ],
            "qa": {
                "question": question,
                "answer": "20",
                "explanation": "",
                "steps": [
                    {
                        "op": "subtract",
                        "arg1": "120",
                        "arg2": "100",
                        "res": "20",
                    }
                ],
                "program": "subtract(120, 100)",
                "gold_inds": {
                    "table_1": "revenue ; $ 100 ; $ 120",
                },
                "exe_ans": 20,
                "program_re": "subtract(120, 100)",
            },
            "id": example_id,
        }
    )


def test_preparer_deduplicates_shared_documents_and_elements() -> None:
    first = make_raw_record()
    second = make_raw_record(
        example_id="ABC/2025/page_10.pdf-2",
        question="What was revenue in 2025?",
    )

    result = prepare_finqa_split(
        StubFinQASource([first, second]),
        split=DatasetSplit.TRAIN,
    )

    assert len(result.documents) == 1
    assert len(result.examples) == 2

    element_ids = [element.element_id for element in result.elements]
    assert len(element_ids) == len(set(element_ids))


def test_preparer_preserves_split() -> None:
    result = prepare_finqa_split(
        StubFinQASource([make_raw_record()]),
        split=DatasetSplit.VALIDATION,
    )

    assert result.split is DatasetSplit.VALIDATION
    assert result.examples[0].split is DatasetSplit.VALIDATION


def test_preparer_rejects_duplicate_example_ids() -> None:
    record = make_raw_record()

    with pytest.raises(ValueError, match="Duplicate dataset example ID"):
        prepare_finqa_split(
            StubFinQASource([record, record]),
            split=DatasetSplit.TRAIN,
        )


def test_preparer_rejects_conflicting_document_elements() -> None:
    first = make_raw_record()
    second = make_raw_record(
        example_id="ABC/2025/page_10.pdf-2",
        first_paragraph="A conflicting paragraph.",
    )

    with pytest.raises(ValueError, match="Conflicting document element"):
        prepare_finqa_split(
            StubFinQASource([first, second]),
            split=DatasetSplit.TRAIN,
        )


def test_preparer_produces_stable_order() -> None:
    first = make_raw_record(
        example_id="ABC/2025/page_10.pdf-2",
    )
    second = make_raw_record(
        example_id="ABC/2025/page_10.pdf-1",
    )

    result = prepare_finqa_split(
        StubFinQASource([first, second]),
        split=DatasetSplit.TRAIN,
    )

    example_ids = [example.example_id for example in result.examples]

    assert example_ids == sorted(example_ids)
