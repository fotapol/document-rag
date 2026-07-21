import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from document_rag.datasets.docfinqa.raw_models import (
    DocFinQARawRecord,
)
from document_rag.datasets.finqa.raw_models import (
    FinQARawRecord,
)

_WHITESPACE_PATTERN = re.compile(r"\s+")


class DocFinQALinkStatus(StrEnum):
    EXACT = "exact"
    EQUIVALENT = "equivalent"
    AMBIGUOUS = "ambiguous"
    ANSWER_MISMATCH = "answer_mismatch"


@dataclass(frozen=True, slots=True)
class DocFinQALinkResult:
    status: DocFinQALinkStatus
    finqa_record: FinQARawRecord | None


class DocFinQALinker:
    """Link DocFinQA records to FinQA annotations."""

    def __init__(
        self,
        finqa_records: Iterable[FinQARawRecord],
    ) -> None:
        self._question_answer_index: dict[
            tuple[str, str],
            list[FinQARawRecord],
        ] = defaultdict(list)

        self._question_index: dict[
            str,
            list[FinQARawRecord],
        ] = defaultdict(list)

        for record in finqa_records:
            question = _normalize(record.qa.question)
            answer = _normalize(_resolved_answer(record))

            self._question_answer_index[(question, answer)].append(record)

            self._question_index[question].append(record)

    def link(
        self,
        record: DocFinQARawRecord,
    ) -> DocFinQALinkResult:
        question = _normalize(record.question)
        answer = _normalize(record.answer)

        candidates = self._question_answer_index.get(
            (question, answer),
            [],
        )

        if len(candidates) == 1:
            return DocFinQALinkResult(
                status=DocFinQALinkStatus.EXACT,
                finqa_record=candidates[0],
            )

        if len(candidates) > 1:
            if _are_equivalent(candidates):
                selected = min(
                    candidates,
                    key=lambda candidate: candidate.id,
                )

                return DocFinQALinkResult(
                    status=DocFinQALinkStatus.EQUIVALENT,
                    finqa_record=selected,
                )

            return DocFinQALinkResult(
                status=DocFinQALinkStatus.AMBIGUOUS,
                finqa_record=None,
            )

        if question in self._question_index:
            return DocFinQALinkResult(
                status=DocFinQALinkStatus.ANSWER_MISMATCH,
                finqa_record=None,
            )

        return DocFinQALinkResult(
            status=DocFinQALinkStatus.AMBIGUOUS,
            finqa_record=None,
        )


def _normalize(value: object) -> str:
    return (
        _WHITESPACE_PATTERN.sub(
            " ",
            str(value),
        )
        .strip()
        .casefold()
    )


def _resolved_answer(record: FinQARawRecord) -> object:
    answer = record.qa.answer

    if answer is None or not str(answer).strip():
        return record.qa.exe_ans

    return answer


def _are_equivalent(
    candidates: list[FinQARawRecord],
) -> bool:
    signatures = {
        (
            candidate.filename,
            _normalize(candidate.qa.program_re or candidate.qa.program or ""),
            json.dumps(
                candidate.qa.gold_inds,
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        for candidate in candidates
    }

    return len(signatures) == 1
