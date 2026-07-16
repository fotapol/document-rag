from pydantic import BaseModel, ConfigDict


class BaseDomainModel(BaseModel):
    """Base class for immutable domain objects."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )
