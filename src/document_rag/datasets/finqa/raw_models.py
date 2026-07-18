from pydantic import BaseModel, ConfigDict, JsonValue


class FinQARawModel(BaseModel):
    """Base model for data received from the external FinQA dataset."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )


class FinQARawStep(FinQARawModel):
    op: str
    arg1: JsonValue
    arg2: JsonValue
    res: JsonValue


class FinQARawQuestionAnswer(FinQARawModel):
    question: str
    answer: JsonValue = None
    explanation: str
    steps: tuple[FinQARawStep, ...]
    program: str
    gold_inds: dict[str, str]
    exe_ans: JsonValue
    program_re: str


class FinQARawRecord(FinQARawModel):
    pre_text: tuple[str, ...]
    post_text: tuple[str, ...]
    filename: str
    table_ori: tuple[tuple[str, ...], ...]
    table: tuple[tuple[str, ...], ...]
    qa: FinQARawQuestionAnswer
    id: str
