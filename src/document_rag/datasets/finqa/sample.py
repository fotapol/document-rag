from hashlib import sha256

from document_rag.datasets.finqa.preparer import PreparedFinQASplit
from document_rag.datasets.models import DatasetExample
from document_rag.domain import Document, DocumentElement


def create_finqa_sample(
    prepared_split: PreparedFinQASplit,
    *,
    example_count: int,
    seed: str = "document-rag-finqa-v1",
) -> PreparedFinQASplit:
    """Create a deterministic subset of one prepared FinQA split."""

    if example_count < 1:
        raise ValueError("Sample example count must be positive")

    if example_count > len(prepared_split.examples):
        raise ValueError("Sample example count cannot exceed the split size")

    normalized_seed = seed.strip()

    if not normalized_seed:
        raise ValueError("Sample seed cannot be empty")

    ranked_examples = sorted(
        prepared_split.examples,
        key=lambda example: (
            _stable_rank(
                example_id=example.example_id,
                seed=normalized_seed,
            ),
            example.example_id,
        ),
    )

    selected_examples = tuple(
        sorted(
            ranked_examples[:example_count],
            key=lambda example: example.example_id,
        )
    )

    selected_document_ids = frozenset(example.question.document_id for example in selected_examples)

    selected_documents = tuple(
        sorted(
            (
                document
                for document in prepared_split.documents
                if document.document_id in selected_document_ids
            ),
            key=lambda document: document.document_id,
        )
    )

    selected_elements = tuple(
        sorted(
            (
                element
                for element in prepared_split.elements
                if element.document_id in selected_document_ids
            ),
            key=lambda element: element.element_id,
        )
    )

    _validate_sample_references(
        selected_document_ids=selected_document_ids,
        selected_documents=selected_documents,
        selected_elements=selected_elements,
        selected_examples=selected_examples,
    )

    return PreparedFinQASplit(
        split=prepared_split.split,
        documents=selected_documents,
        elements=selected_elements,
        examples=selected_examples,
    )


def _stable_rank(
    *,
    example_id: str,
    seed: str,
) -> str:
    payload = f"{seed}\0{example_id}".encode()
    return sha256(payload).hexdigest()


def _validate_sample_references(
    *,
    selected_document_ids: frozenset[str],
    selected_documents: tuple[Document, ...],
    selected_elements: tuple[DocumentElement, ...],
    selected_examples: tuple[DatasetExample, ...],
) -> None:
    available_document_ids = {document.document_id for document in selected_documents}

    missing_document_ids = selected_document_ids - available_document_ids

    if missing_document_ids:
        raise ValueError("Sample references documents missing from the prepared split")

    available_element_ids = {element.element_id for element in selected_elements}

    for example in selected_examples:
        for supporting_fact in example.supporting_facts:
            if supporting_fact.element_id not in available_element_ids:
                raise ValueError(
                    f"Sample example {example.example_id!r} references "
                    f"missing element {supporting_fact.element_id!r}"
                )
