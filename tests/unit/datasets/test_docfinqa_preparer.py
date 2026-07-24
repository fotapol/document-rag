import pytest

from document_rag.datasets.docfinqa import (
    DocFinQAChunker,
    DocFinQAEvidenceSelector,
    DocFinQANormalizer,
    DocFinQARawRecord,
    DocFinQASplitPreparer,
)
from document_rag.datasets.finqa.raw_models import (
    FinQARawRecord,
)
from document_rag.datasets.models import (
    DatasetName,
    DatasetSplit,
)


def make_finqa_record(
    *,
    record_id: str = "ABC/2020/page_1.pdf-1",
    filename: str = "ABC/2020/page_1.pdf",
    question: str = "What was the revenue?",
    answer: object = "100",
    evidence: str = "revenue was 100",
) -> FinQARawRecord:
    return FinQARawRecord.model_validate(
        {
            "id": record_id,
            "filename": filename,
            "pre_text": ["Introduction."],
            "post_text": ["Conclusion."],
            "table_ori": [
                ["Metric", "2020"],
                ["Revenue", "100"],
            ],
            "table": [
                ["Metric", "2020"],
                ["Revenue", "100"],
            ],
            "qa": {
                "id": record_id,
                "question": question,
                "answer": answer,
                "exe_ans": answer,
                "explanation": "Financial calculation.",
                "program": "divide(200, const_2)",
                "program_re": "divide(200, const_2)",
                "gold_inds": {
                    "text_1": evidence,
                },
                "steps": [],
            },
        }
    )


def make_docfinqa_record(
    *,
    context: str = ("Annual report. Revenue was 100. End of report."),
    question: str = "What was the revenue?",
    answer: str = "100",
) -> DocFinQARawRecord:
    return DocFinQARawRecord.model_validate(
        {
            "Context": context,
            "Question": question,
            "Program": "answer = 100",
            "Answer": answer,
        }
    )


def make_normalizer() -> DocFinQANormalizer:
    return DocFinQANormalizer(
        chunker=DocFinQAChunker(
            chunk_size=100,
            overlap=20,
        ),
        evidence_selector=DocFinQAEvidenceSelector(
            minimum_score=0.8,
        ),
    )


def make_preparer(
    finqa_records: list[FinQARawRecord],
) -> DocFinQASplitPreparer:
    return DocFinQASplitPreparer(
        split=DatasetSplit.TRAIN,
        finqa_records=finqa_records,
        normalizer=make_normalizer(),
    )


def test_preparer_emits_document_elements_and_example() -> None:
    preparer = make_preparer([make_finqa_record()])

    items = list(preparer.iter_prepare([make_docfinqa_record()]))

    assert len(items) == 1

    item = items[0]

    assert item.document is not None
    assert item.document.metadata["dataset"] == DatasetName.DOCFINQA.value
    assert item.document.metadata["split"] == DatasetSplit.TRAIN.value
    assert item.elements
    assert item.example.question.text == ("What was the revenue?")
    assert item.example.reference_answer.text == "100"

    assert preparer.stats.total_records == 1
    assert preparer.stats.normalized_records == 1
    assert preparer.stats.unique_documents == 1
    assert preparer.stats.exact_links == 1
    assert preparer.stats.skipped_records == 0


def test_preparer_emits_repeated_document_once() -> None:
    context = "Annual report. Revenue was 100. Operating income was 40."

    finqa_records = [
        make_finqa_record(),
        make_finqa_record(
            record_id="ABC/2020/page_2.pdf-1",
            filename="ABC/2020/page_2.pdf",
            question="What was operating income?",
            answer="40",
            evidence="operating income was 40",
        ),
    ]

    raw_records = [
        make_docfinqa_record(
            context=context,
        ),
        make_docfinqa_record(
            context=context,
            question="What was operating income?",
            answer="40",
        ),
    ]

    preparer = make_preparer(finqa_records)
    items = list(preparer.iter_prepare(raw_records))

    assert len(items) == 2

    assert items[0].document is not None
    assert items[0].elements

    assert items[1].document is None
    assert items[1].elements == ()

    assert items[0].example.question.document_id == items[1].example.question.document_id
    assert preparer.stats.unique_documents == 1
    assert preparer.stats.normalized_records == 2


def test_preparer_skips_ambiguous_link() -> None:
    finqa_records = [
        make_finqa_record(
            record_id="ABC/2020/page_1.pdf-1",
            filename="ABC/2020/page_1.pdf",
        ),
        make_finqa_record(
            record_id="ABC/2021/page_1.pdf-1",
            filename="ABC/2021/page_1.pdf",
        ),
    ]

    preparer = make_preparer(finqa_records)

    items = list(preparer.iter_prepare([make_docfinqa_record()]))

    assert items == []
    assert preparer.stats.total_records == 1
    assert preparer.stats.normalized_records == 0
    assert preparer.stats.skipped_ambiguous == 1


def test_preparer_skips_answer_mismatch() -> None:
    preparer = make_preparer(
        [
            make_finqa_record(
                answer="$ 100 million",
            )
        ]
    )

    items = list(
        preparer.iter_prepare(
            [
                make_docfinqa_record(
                    answer="100",
                )
            ]
        )
    )

    assert items == []
    assert preparer.stats.skipped_answer_mismatch == 1


def test_preparer_skips_duplicate_examples() -> None:
    preparer = make_preparer([make_finqa_record()])
    raw_record = make_docfinqa_record()

    items = list(
        preparer.iter_prepare(
            [
                raw_record,
                raw_record,
            ]
        )
    )

    assert len(items) == 1
    assert preparer.stats.total_records == 2
    assert preparer.stats.normalized_records == 1
    assert preparer.stats.skipped_duplicate == 1
    assert preparer.stats.skipped_records == 1


def test_preparer_skips_incomplete_evidence() -> None:
    preparer = make_preparer(
        [
            make_finqa_record(
                evidence=("unrelated debt maturity information"),
            )
        ]
    )

    items = list(preparer.iter_prepare([make_docfinqa_record()]))

    assert items == []
    assert preparer.stats.skipped_evidence_incomplete == 1
    assert preparer.stats.unmatched_evidence_facts == 1


def test_preparer_rejects_empty_split() -> None:
    preparer = make_preparer([make_finqa_record()])

    with pytest.raises(
        ValueError,
        match="DocFinQA split is empty",
    ):
        list(preparer.iter_prepare([]))


def test_preparer_is_single_use() -> None:
    preparer = make_preparer([make_finqa_record()])

    list(preparer.iter_prepare([make_docfinqa_record()]))

    with pytest.raises(
        RuntimeError,
        match="single-use",
    ):
        list(preparer.iter_prepare([make_docfinqa_record()]))


def test_preparation_is_deterministic() -> None:
    finqa_records = [make_finqa_record()]
    raw_records = [make_docfinqa_record()]

    first_preparer = make_preparer(finqa_records)
    second_preparer = make_preparer(finqa_records)

    first_items = list(first_preparer.iter_prepare(raw_records))
    second_items = list(second_preparer.iter_prepare(raw_records))

    assert first_items == second_items
    assert first_preparer.stats == second_preparer.stats
