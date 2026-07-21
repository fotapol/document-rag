from document_rag.datasets.docfinqa import (
    DocFinQALinker,
    DocFinQALinkStatus,
    DocFinQARawRecord,
)
from document_rag.datasets.finqa.raw_models import (
    FinQARawRecord,
)


def make_finqa_record(
    *,
    record_id: str = "ABC/2020/page_1.pdf-1",
    filename: str = "ABC/2020/page_1.pdf",
    question: str = "What was the revenue?",
    answer: object = "100",
    executable_answer: object = 100,
    program: str = "divide(200, const_2)",
    normalized_program: str = "divide(200, const_2)",
    gold_inds: dict[str, str] | None = None,
) -> FinQARawRecord:
    return FinQARawRecord.model_validate(
        {
            "id": record_id,
            "filename": filename,
            "pre_text": ["Introductory text."],
            "post_text": ["Closing text."],
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
                "exe_ans": executable_answer,
                "explanation": "Revenue was 100.",
                "program": program,
                "program_re": normalized_program,
                "gold_inds": gold_inds or {"table_1": ("the revenue of 2020 is 100 ;")},
                "steps": [],
            },
        }
    )


def make_docfinqa_record(
    *,
    question: str = "What was the revenue?",
    answer: str = "100",
) -> DocFinQARawRecord:
    return DocFinQARawRecord.model_validate(
        {
            "Context": "Complete annual report.",
            "Question": question,
            "Program": "answer = 100",
            "Answer": answer,
        }
    )


def test_linker_returns_unique_exact_match() -> None:
    finqa_record = make_finqa_record()
    linker = DocFinQALinker([finqa_record])

    result = linker.link(make_docfinqa_record())

    assert result.status is DocFinQALinkStatus.EXACT
    assert result.finqa_record is finqa_record


def test_linker_normalizes_case_and_whitespace() -> None:
    finqa_record = make_finqa_record(
        question="What   was the Revenue?",
        answer=" 100 ",
    )
    linker = DocFinQALinker([finqa_record])

    result = linker.link(
        make_docfinqa_record(
            question=" what was the revenue? ",
            answer="100",
        )
    )

    assert result.status is DocFinQALinkStatus.EXACT
    assert result.finqa_record is finqa_record


def test_linker_uses_executable_answer_as_fallback() -> None:
    finqa_record = make_finqa_record(
        answer="",
        executable_answer=100,
    )
    linker = DocFinQALinker([finqa_record])

    result = linker.link(make_docfinqa_record(answer="100"))

    assert result.status is DocFinQALinkStatus.EXACT
    assert result.finqa_record is finqa_record


def test_linker_selects_lowest_equivalent_id() -> None:
    higher_id_record = make_finqa_record(
        record_id="ABC/2020/page_1.pdf-3",
    )
    lower_id_record = make_finqa_record(
        record_id="ABC/2020/page_1.pdf-2",
    )

    linker = DocFinQALinker(
        [
            higher_id_record,
            lower_id_record,
        ]
    )

    result = linker.link(make_docfinqa_record())

    assert result.status is DocFinQALinkStatus.EQUIVALENT
    assert result.finqa_record is lower_id_record


def test_linker_rejects_candidates_from_different_pages() -> None:
    first_record = make_finqa_record(
        record_id="ABC/2020/page_1.pdf-1",
        filename="ABC/2020/page_1.pdf",
    )
    second_record = make_finqa_record(
        record_id="ABC/2021/page_2.pdf-1",
        filename="ABC/2021/page_2.pdf",
    )

    linker = DocFinQALinker(
        [
            first_record,
            second_record,
        ]
    )

    result = linker.link(make_docfinqa_record())

    assert result.status is DocFinQALinkStatus.AMBIGUOUS
    assert result.finqa_record is None


def test_linker_rejects_different_annotations() -> None:
    first_record = make_finqa_record(
        record_id="ABC/2020/page_1.pdf-1",
        gold_inds={"table_1": "the revenue is 100 ;"},
    )
    second_record = make_finqa_record(
        record_id="ABC/2020/page_1.pdf-2",
        gold_inds={"text_1": "revenue increased to 100 ;"},
    )

    linker = DocFinQALinker(
        [
            first_record,
            second_record,
        ]
    )

    result = linker.link(make_docfinqa_record())

    assert result.status is DocFinQALinkStatus.AMBIGUOUS
    assert result.finqa_record is None


def test_linker_reports_answer_mismatch() -> None:
    finqa_record = make_finqa_record(
        answer="$ 100 million",
    )
    linker = DocFinQALinker([finqa_record])

    result = linker.link(make_docfinqa_record(answer="100"))

    assert result.status is DocFinQALinkStatus.ANSWER_MISMATCH
    assert result.finqa_record is None
