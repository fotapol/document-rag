import pytest

from document_rag.datasets import DatasetSplit
from document_rag.datasets.finqa import FinQARawRecord, normalize_finqa_record
from document_rag.domain import DocumentElementType


def make_raw_record() -> FinQARawRecord:
    return FinQARawRecord.model_validate(
        {
            "pre_text": [
                "Revenue increased during the year.",
                "Operating income was $120 million.",
            ],
            "post_text": [
                "Additional information follows.",
            ],
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
                "question": "How much did revenue increase?",
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
                    "text_1": "Operating income was $120 million.",
                    "table_1": "revenue ; $ 100 ; $ 120",
                },
                "exe_ans": 20,
                "program_re": "subtract(120, 100)",
            },
            "id": "ABC/2025/page_10.pdf-1",
        }
    )


def test_normalizer_creates_internal_models() -> None:
    result = normalize_finqa_record(
        make_raw_record(),
        split=DatasetSplit.TRAIN,
    )

    assert result.document.document_id == "finqa:ABC/2025/page_10.pdf"
    assert result.example.question.document_id == result.document.document_id
    assert result.example.reference_answer.text == "20"


def test_normalizer_creates_text_and_table_elements() -> None:
    result = normalize_finqa_record(
        make_raw_record(),
        split=DatasetSplit.TRAIN,
    )

    element_types = {element.element_type for element in result.elements}

    assert DocumentElementType.PARAGRAPH in element_types
    assert DocumentElementType.TABLE in element_types
    assert DocumentElementType.TABLE_ROW in element_types


def test_normalizer_maps_supporting_facts_to_elements() -> None:
    result = normalize_finqa_record(
        make_raw_record(),
        split=DatasetSplit.TRAIN,
    )

    element_ids = {element.element_id for element in result.elements}
    supporting_element_ids = {fact.element_id for fact in result.example.supporting_facts}

    assert supporting_element_ids <= element_ids
    assert len(supporting_element_ids) == 2


def test_normalization_is_deterministic() -> None:
    record = make_raw_record()

    first = normalize_finqa_record(record, split=DatasetSplit.TRAIN)
    second = normalize_finqa_record(record, split=DatasetSplit.TRAIN)

    assert first == second


def test_normalizer_rejects_unknown_supporting_fact() -> None:
    record = make_raw_record()
    invalid_record = record.model_copy(
        update={"qa": record.qa.model_copy(update={"gold_inds": {"text_999": "Missing paragraph"}})}
    )

    with pytest.raises(ValueError, match="does not reference"):
        normalize_finqa_record(
            invalid_record,
            split=DatasetSplit.TRAIN,
        )


def test_normalizer_uses_executable_answer_when_answer_is_empty() -> None:
    record = make_raw_record()
    record_with_empty_answer = record.model_copy(
        update={
            "qa": record.qa.model_copy(
                update={
                    "answer": "",
                    "exe_ans": 20,
                }
            )
        }
    )

    result = normalize_finqa_record(
        record_with_empty_answer,
        split=DatasetSplit.VALIDATION,
    )

    assert result.example.reference_answer.text == "20"
    assert result.example.reference_answer.executable_answer == 20


def test_normalizer_rejects_missing_answer_and_executable_answer() -> None:
    record = make_raw_record()
    invalid_record = record.model_copy(
        update={
            "qa": record.qa.model_copy(
                update={
                    "answer": "",
                    "exe_ans": None,
                }
            )
        }
    )

    with pytest.raises(ValueError, match="exe_ans cannot be null"):
        normalize_finqa_record(
            invalid_record,
            split=DatasetSplit.VALIDATION,
        )


def test_normalizer_accepts_table_operation_without_second_argument() -> None:
    record = make_raw_record()
    record_with_unary_step = record.model_copy(
        update={
            "qa": record.qa.model_copy(
                update={
                    "steps": (
                        record.qa.steps[0].model_copy(
                            update={
                                "op": "min1-1",
                                "arg1": "expected volatility",
                                "arg2": "",
                                "res": "39.8",
                            }
                        ),
                    ),
                    "program": (
                        "table_min(expected volatility, none), table_max(expected volatility, none)"
                    ),
                }
            )
        }
    )

    result = normalize_finqa_record(
        record_with_unary_step,
        split=DatasetSplit.TRAIN,
    )

    step = result.example.reference_answer.steps[0]

    assert step.operation == "min1-1"
    assert step.arguments == ("expected volatility",)
    assert step.result == "39.8"


def test_normalizer_rejects_reasoning_step_without_arguments() -> None:
    record = make_raw_record()
    invalid_record = record.model_copy(
        update={
            "qa": record.qa.model_copy(
                update={
                    "steps": (
                        record.qa.steps[0].model_copy(
                            update={
                                "arg1": "",
                                "arg2": "",
                            }
                        ),
                    )
                }
            )
        }
    )

    with pytest.raises(ValueError, match="at least one argument"):
        normalize_finqa_record(
            invalid_record,
            split=DatasetSplit.TRAIN,
        )


def test_normalizer_accepts_reasoning_step_without_result() -> None:
    record = make_raw_record()
    record_without_step_result = record.model_copy(
        update={
            "qa": record.qa.model_copy(
                update={
                    "steps": (
                        record.qa.steps[0].model_copy(
                            update={
                                "op": "divide1-1",
                                "arg1": "27.1",
                                "arg2": "82.7",
                                "res": "",
                            }
                        ),
                    )
                }
            )
        }
    )

    result = normalize_finqa_record(
        record_without_step_result,
        split=DatasetSplit.TRAIN,
    )

    step = result.example.reference_answer.steps[0]

    assert step.arguments == ("27.1", "82.7")
    assert step.result is None


def test_normalizer_resolves_negative_text_index() -> None:
    record = make_raw_record()
    record_with_negative_index = record.model_copy(
        update={
            "qa": record.qa.model_copy(
                update={
                    "gold_inds": {
                        "text_-1": "Additional information follows.",
                    }
                }
            )
        }
    )

    result = normalize_finqa_record(
        record_with_negative_index,
        split=DatasetSplit.TRAIN,
    )

    supporting_fact = result.example.supporting_facts[0]

    assert supporting_fact.source_key == "text_-1"
    assert supporting_fact.element_id.endswith(":text_2")
