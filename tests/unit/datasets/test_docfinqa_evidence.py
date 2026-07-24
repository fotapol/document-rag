import pytest

from document_rag.datasets.docfinqa.chunking import (
    DocFinQAChunk,
)
from document_rag.datasets.docfinqa.evidence import (
    DocFinQAEvidenceMatch,
    DocFinQAEvidenceSelector,
)
from document_rag.datasets.finqa.raw_models import (
    FinQARawRecord,
)


def make_chunk(
    *,
    index: int,
    text: str,
) -> DocFinQAChunk:
    start_char = index * 1_000

    return DocFinQAChunk(
        index=index,
        start_char=start_char,
        end_char=start_char + len(text),
        text=text,
    )


def make_finqa_record(
    *,
    gold_inds: dict[str, str],
) -> FinQARawRecord:
    return FinQARawRecord.model_validate(
        {
            "id": "ABC/2020/page_1.pdf-1",
            "filename": "ABC/2020/page_1.pdf",
            "pre_text": ["Annual report introduction."],
            "post_text": ["Annual report conclusion."],
            "table_ori": [
                ["Metric", "2020"],
                ["Revenue", "100"],
            ],
            "table": [
                ["Metric", "2020"],
                ["Revenue", "100"],
            ],
            "qa": {
                "id": "ABC/2020/page_1.pdf-1",
                "question": "What was the revenue?",
                "answer": "100",
                "exe_ans": 100,
                "explanation": "Revenue was 100.",
                "program": "divide(200, const_2)",
                "program_re": "divide(200, const_2)",
                "gold_inds": gold_inds,
                "steps": [],
            },
        }
    )


def test_selector_finds_exact_normalized_match() -> None:
    chunks = [
        make_chunk(
            index=0,
            text="General annual report information.",
        ),
        make_chunk(
            index=1,
            text=("Revenue increased from 90 to 100 during the fiscal year."),
        ),
    ]
    record = make_finqa_record(
        gold_inds={"text_1": ("revenue increased from 90 to 100 during the fiscal year .")}
    )

    result = DocFinQAEvidenceSelector().select(
        chunks=chunks,
        finqa_record=record,
    )

    assert result == (
        DocFinQAEvidenceMatch(
            source_key="text_1",
            chunk_index=1,
            score=1.0,
        ),
    )


def test_selector_matches_table_evidence_by_tokens() -> None:
    chunks = [
        make_chunk(
            index=0,
            text=("Property and equipment amounted to $300. Total assets acquired were $11,010."),
        ),
        make_chunk(
            index=1,
            text=("The company reported operating income of $900."),
        ),
    ]
    record = make_finqa_record(
        gold_inds={
            "table_4": ("the property and equipment is 300 ; the total assets acquired is 11010 ;")
        }
    )

    result = DocFinQAEvidenceSelector(
        minimum_score=0.5,
    ).select(
        chunks=chunks,
        finqa_record=record,
    )

    assert len(result) == 1
    assert result[0].source_key == "table_4"
    assert result[0].chunk_index == 0
    assert result[0].score >= 0.5


def test_selector_matches_multiple_supporting_facts() -> None:
    chunks = [
        make_chunk(
            index=0,
            text=("Net revenue for 2020 was $100 million."),
        ),
        make_chunk(
            index=1,
            text=("Net revenue for 2021 was $120 million."),
        ),
    ]
    record = make_finqa_record(
        gold_inds={
            "table_1": ("net revenue for 2020 was 100 million"),
            "table_2": ("net revenue for 2021 was 120 million"),
        }
    )

    result = DocFinQAEvidenceSelector().select(
        chunks=chunks,
        finqa_record=record,
    )

    assert [(match.source_key, match.chunk_index) for match in result] == [
        ("table_1", 0),
        ("table_2", 1),
    ]


def test_selector_uses_lowest_chunk_index_on_tie() -> None:
    chunks = [
        make_chunk(
            index=2,
            text="Revenue was 100.",
        ),
        make_chunk(
            index=1,
            text="Revenue was 100.",
        ),
    ]
    record = make_finqa_record(gold_inds={"text_1": "revenue was 100"})

    result = DocFinQAEvidenceSelector().select(
        chunks=chunks,
        finqa_record=record,
    )

    assert result[0].chunk_index == 1


def test_selector_omits_low_score_evidence() -> None:
    chunks = [
        make_chunk(
            index=0,
            text="General annual report information.",
        )
    ]
    record = make_finqa_record(gold_inds={"text_1": ("completely unrelated debt maturity data")})

    result = DocFinQAEvidenceSelector(
        minimum_score=0.8,
    ).select(
        chunks=chunks,
        finqa_record=record,
    )

    assert result == ()


def test_selector_returns_empty_for_empty_gold_inds() -> None:
    record = make_finqa_record(gold_inds={})

    result = DocFinQAEvidenceSelector().select(
        chunks=[
            make_chunk(
                index=0,
                text="Annual report.",
            )
        ],
        finqa_record=record,
    )

    assert result == ()


def test_selector_rejects_empty_chunks() -> None:
    record = make_finqa_record(gold_inds={"text_1": "revenue was 100"})

    with pytest.raises(
        ValueError,
        match="At least one DocFinQA chunk",
    ):
        DocFinQAEvidenceSelector().select(
            chunks=[],
            finqa_record=record,
        )


def test_selector_rejects_duplicate_chunk_indexes() -> None:
    record = make_finqa_record(gold_inds={"text_1": "revenue was 100"})

    chunks = [
        make_chunk(
            index=0,
            text="Revenue was 100.",
        ),
        make_chunk(
            index=0,
            text="Another revenue statement.",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="chunk indexes must be unique",
    ):
        DocFinQAEvidenceSelector().select(
            chunks=chunks,
            finqa_record=record,
        )


@pytest.mark.parametrize(
    "minimum_score",
    [
        0.0,
        -0.1,
        1.1,
    ],
)
def test_selector_rejects_invalid_minimum_score(
    minimum_score: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Minimum score",
    ):
        DocFinQAEvidenceSelector(
            minimum_score=minimum_score,
        )


def test_evidence_match_rejects_invalid_score() -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        DocFinQAEvidenceMatch(
            source_key="text_1",
            chunk_index=0,
            score=1.1,
        )
