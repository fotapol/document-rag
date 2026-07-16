from decimal import Decimal

import pytest
from pydantic import ValidationError

from document_rag.domain import (
    Calculation,
    CalculationOperand,
    CalculationOperation,
)


def make_operand(
    *,
    label: str,
    value: str,
    evidence_id: str = "evidence-1",
) -> CalculationOperand:
    return CalculationOperand(
        label=label,
        value=Decimal(value),
        source_evidence_ids=(evidence_id,),
    )


def test_calculation_accepts_percentage_change() -> None:
    calculation = Calculation(
        calculation_id="calculation-1",
        question_id="question-1",
        operation=CalculationOperation.PERCENTAGE_CHANGE,
        operands=(
            make_operand(label="Revenue 2024", value="100"),
            make_operand(label="Revenue 2025", value="120"),
        ),
        result=Decimal("20"),
        result_unit="percent",
        display_formula="(120 - 100) / 100 x 100",
    )

    assert calculation.result == Decimal("20")
    assert calculation.operation is CalculationOperation.PERCENTAGE_CHANGE


def test_operand_requires_source() -> None:
    with pytest.raises(ValidationError, match="requires a source"):
        CalculationOperand(
            label="Revenue",
            value=Decimal("120"),
        )


def test_binary_operation_requires_two_operands() -> None:
    with pytest.raises(ValidationError, match="exactly two operands"):
        Calculation(
            calculation_id="calculation-1",
            question_id="question-1",
            operation=CalculationOperation.RATIO,
            operands=(
                make_operand(label="Revenue", value="120"),
                make_operand(label="Assets", value="300"),
                make_operand(label="Employees", value="50"),
            ),
            result=Decimal("0.4"),
            display_formula="120 / 300",
        )


def test_ratio_rejects_zero_denominator() -> None:
    with pytest.raises(ValidationError, match="denominator cannot be zero"):
        Calculation(
            calculation_id="calculation-1",
            question_id="question-1",
            operation=CalculationOperation.RATIO,
            operands=(
                make_operand(label="Revenue", value="120"),
                make_operand(label="Assets", value="0"),
            ),
            result=Decimal("0"),
            display_formula="120 / 0",
        )


def test_decimal_values_serialize_as_strings() -> None:
    calculation = Calculation(
        calculation_id="calculation-1",
        question_id="question-1",
        operation=CalculationOperation.DIFFERENCE,
        operands=(
            make_operand(label="Revenue 2025", value="120.50"),
            make_operand(label="Revenue 2024", value="100.25"),
        ),
        result=Decimal("20.25"),
        result_unit="USD million",
        display_formula="120.50 - 100.25",
    )

    result = calculation.model_dump(mode="json")

    assert result["result"] == "20.25"
    assert result["operands"][0]["value"] == "120.50"


def test_percentage_change_rejects_zero_baseline() -> None:
    with pytest.raises(ValidationError, match="Percentage change baseline cannot be zero"):
        Calculation(
            calculation_id="calculation-1",
            question_id="question-1",
            operation=CalculationOperation.PERCENTAGE_CHANGE,
            operands=(
                make_operand(label="Revenue 2024", value="0"),
                make_operand(label="Revenue 2025", value="120"),
            ),
            result=Decimal("0"),
            display_formula="(120 - 0) / 0 x 100",
        )


def test_ratio_accepts_non_zero_denominator() -> None:
    calculation = Calculation(
        calculation_id="calculation-1",
        question_id="question-1",
        operation=CalculationOperation.RATIO,
        operands=(
            make_operand(label="Revenue", value="120"),
            make_operand(label="Assets", value="300"),
        ),
        result=Decimal("0.4"),
        display_formula="120 / 300",
    )
    assert calculation.result == Decimal("0.4")


def test_operand_accepts_calculation_source() -> None:
    operand = CalculationOperand(
        label="Derived Value",
        value=Decimal("50"),
        source_calculation_ids=("calc-parent",),
    )
    assert operand.source_calculation_ids == ("calc-parent",)
    assert not operand.source_evidence_ids
