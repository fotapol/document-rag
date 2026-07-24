from document_rag.datasets.docfinqa import (
    DocFinQAChunker,
    DocFinQAEvidenceSelector,
    DocFinQALinkResult,
    DocFinQALinkStatus,
    DocFinQANormalizationStatus,
    DocFinQANormalizer,
    DocFinQARawRecord,
)
from document_rag.datasets.finqa.raw_models import (
    FinQARawRecord,
)
from document_rag.datasets.models import DatasetSplit


def make_finqa_record(
    *,
    record_id: str = "ABC/2020/page_1.pdf-1",
    filename: str = "ABC/2020/page_1.pdf",
    question: str = "What was the revenue?",
    answer: object = "100",
    gold_inds: dict[str, str] | None = None,
) -> FinQARawRecord:
    if gold_inds is None:
        gold_inds = {"text_1": "revenue was 100"}

    return FinQARawRecord.model_validate(
        {
            "id": record_id,
            "filename": filename,
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
                "id": record_id,
                "question": question,
                "answer": answer,
                "exe_ans": 100,
                "explanation": "Revenue was 100.",
                "program": "divide(200, const_2)",
                "program_re": ("divide(200, const_2)"),
                "gold_inds": gold_inds,
                "steps": [],
            },
        }
    )


def make_docfinqa_record(
    *,
    context: str = ("Annual report. Revenue was 100. End of report."),
    question: str = "What was the revenue?",
    program: str = "answer = 100",
    answer: str = "100",
) -> DocFinQARawRecord:
    return DocFinQARawRecord.model_validate(
        {
            "Context": context,
            "Question": question,
            "Program": program,
            "Answer": answer,
        }
    )


def make_link_result(
    finqa_record: FinQARawRecord,
    *,
    status: DocFinQALinkStatus = (DocFinQALinkStatus.EXACT),
) -> DocFinQALinkResult:
    return DocFinQALinkResult(
        status=status,
        finqa_record=finqa_record,
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


def test_normalizer_builds_complete_record() -> None:
    finqa_record = make_finqa_record()

    result = make_normalizer().normalize(
        raw_record=make_docfinqa_record(),
        split=DatasetSplit.TRAIN,
        link_result=make_link_result(finqa_record),
    )

    assert result.status is DocFinQANormalizationStatus.NORMALIZED
    assert result.record is not None
    assert result.expected_evidence_count == 1
    assert result.matched_evidence_count == 1

    record = result.record

    assert record.example.split is DatasetSplit.TRAIN
    assert record.example.question.metadata["source_example_id"] == finqa_record.id
    assert record.example.question.text == "What was the revenue?"
    assert record.example.reference_answer.text == "100"
    assert record.example.reference_answer.program == "answer = 100"

    assert record.document.document_id.startswith("docfinqa:document:")
    assert record.example.example_id.startswith("docfinqa:example:")
    assert record.example.question.question_id == record.example.example_id
    assert record.example.question.document_id == record.document.document_id

    assert len(record.elements) == 1
    assert len(record.example.supporting_facts) == 1

    assert record.example.supporting_facts[0].element_id == record.elements[0].element_id
    assert record.example.supporting_facts[0].source_key == "text_1"


def test_normalizer_preserves_exact_chunk_text() -> None:
    context = "Heading\n\nRevenue\twas 100.\nClosing section."

    result = make_normalizer().normalize(
        raw_record=make_docfinqa_record(context=context),
        split=DatasetSplit.TRAIN,
        link_result=make_link_result(make_finqa_record()),
    )

    assert result.record is not None
    assert result.record.elements[0].source_text == context


def test_normalizer_converts_blank_program_to_none() -> None:
    result = make_normalizer().normalize(
        raw_record=make_docfinqa_record(program="   "),
        split=DatasetSplit.VALIDATION,
        link_result=make_link_result(make_finqa_record()),
    )

    assert result.record is not None
    assert result.record.example.reference_answer.program is None


def test_normalizer_returns_unlinked_result() -> None:
    link_result = DocFinQALinkResult(
        status=DocFinQALinkStatus.AMBIGUOUS,
        finqa_record=None,
    )

    result = make_normalizer().normalize(
        raw_record=make_docfinqa_record(),
        split=DatasetSplit.TEST,
        link_result=link_result,
    )

    assert result.status is DocFinQANormalizationStatus.UNLINKED
    assert result.record is None
    assert result.expected_evidence_count == 0
    assert result.matched_evidence_count == 0


def test_normalizer_rejects_incomplete_evidence() -> None:
    finqa_record = make_finqa_record(
        gold_inds={
            "text_1": "revenue was 100",
            "text_2": ("unrelated debt maturity evidence"),
        }
    )

    result = make_normalizer().normalize(
        raw_record=make_docfinqa_record(),
        split=DatasetSplit.TRAIN,
        link_result=make_link_result(finqa_record),
    )

    assert result.status is (DocFinQANormalizationStatus.EVIDENCE_INCOMPLETE)
    assert result.record is None
    assert result.expected_evidence_count == 2
    assert result.matched_evidence_count == 1


def test_normalizer_rejects_missing_gold_evidence() -> None:
    result = make_normalizer().normalize(
        raw_record=make_docfinqa_record(),
        split=DatasetSplit.TRAIN,
        link_result=make_link_result(make_finqa_record(gold_inds={})),
    )

    assert result.status is (DocFinQANormalizationStatus.EVIDENCE_INCOMPLETE)
    assert result.record is None
    assert result.expected_evidence_count == 0
    assert result.matched_evidence_count == 0


def test_normalization_is_deterministic() -> None:
    normalizer = make_normalizer()
    raw_record = make_docfinqa_record()
    link_result = make_link_result(make_finqa_record())

    first_result = normalizer.normalize(
        raw_record=raw_record,
        split=DatasetSplit.TRAIN,
        link_result=link_result,
    )
    second_result = normalizer.normalize(
        raw_record=raw_record,
        split=DatasetSplit.TRAIN,
        link_result=link_result,
    )

    assert first_result == second_result


def test_same_document_has_stable_document_id() -> None:
    context = "Annual report. Revenue was 100. End of report."
    normalizer = make_normalizer()

    first_result = normalizer.normalize(
        raw_record=make_docfinqa_record(
            context=context,
            question="What was the revenue?",
        ),
        split=DatasetSplit.TRAIN,
        link_result=make_link_result(make_finqa_record(question="What was the revenue?")),
    )

    second_result = normalizer.normalize(
        raw_record=make_docfinqa_record(
            context=context,
            question="Was revenue equal to 100?",
        ),
        split=DatasetSplit.TRAIN,
        link_result=make_link_result(
            make_finqa_record(
                record_id=("ABC/2020/page_1.pdf-2"),
                question=("Was revenue equal to 100?"),
            )
        ),
    )

    assert first_result.record is not None
    assert second_result.record is not None

    assert first_result.record.document.document_id == second_result.record.document.document_id
    assert first_result.record.example.example_id != second_result.record.example.example_id
