from typing import Annotated

from pydantic import Field

type NonEmptyString = Annotated[str, Field(min_length=1)]
type PositivePageNumber = Annotated[int, Field(ge=1)]
