from decimal import Decimal
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from document_rag.domain.base import BaseDomainModel
from document_rag.domain.types import NonEmptyString


class CalculationOperation(StrEnum):
    """Supported deterministic financial operations."""

    SUM = "sum"
    DIFFERENCE = "difference"
    PRODUCT = "product"
    RATIO = "ratio"
    PERCENTAGE = "percentage"
    PERCENTAGE_CHANGE = "percentage_change"
    AVERAGE = "average"


class CalculationOperand(BaseDomainModel):
    """Value used as an input to a calculation."""

    label: NonEmptyString
    value: Decimal
    unit: str | None = Field(default=None, min_length=1)
    period: str | None = Field(default=None, min_length=1)

    source_evidence_ids: tuple[NonEmptyString, ...] = ()
    source_calculation_ids: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if not self.source_evidence_ids and not self.source_calculation_ids:
            raise ValueError("Calculation operand requires a source")

        return self


class Calculation(BaseDomainModel):
    """Auditable result produced by the numerical reasoning engine."""

    calculation_id: NonEmptyString
    question_id: NonEmptyString
    operation: CalculationOperation
    operands: tuple[CalculationOperand, ...] = Field(min_length=2)

    result: Decimal
    result_unit: str | None = Field(default=None, min_length=1)
    display_formula: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_operation(self) -> Self:
        binary_operations = {
            CalculationOperation.DIFFERENCE,
            CalculationOperation.RATIO,
            CalculationOperation.PERCENTAGE,
            CalculationOperation.PERCENTAGE_CHANGE,
        }

        if self.operation in binary_operations and len(self.operands) != 2:
            raise ValueError(f"{self.operation.value} requires exactly two operands")

        if self.operation in {
            CalculationOperation.RATIO,
            CalculationOperation.PERCENTAGE,
        }:
            denominator = self.operands[1].value

            if denominator == 0:
                raise ValueError(f"{self.operation.value} denominator cannot be zero")

        if self.operation is CalculationOperation.PERCENTAGE_CHANGE:
            baseline = self.operands[0].value

            if baseline == 0:
                raise ValueError("Percentage change baseline cannot be zero")

        return self
