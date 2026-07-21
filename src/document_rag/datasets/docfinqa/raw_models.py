from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)


class DocFinQARawRecord(BaseModel):
    """One record in the external DocFinQA JSON format."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    context: str = Field(alias="Context")
    question: str = Field(alias="Question")
    program: str = Field(alias="Program")
    answer: str = Field(alias="Answer")

    @field_validator("context", "question", "answer")
    @classmethod
    def validate_required_text(
        cls,
        value: str,
        info: ValidationInfo,
    ) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} cannot be empty")

        return value
