"""Deterministic local cache for normalized document embeddings."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np

from document_rag.ingestion.chunking import DocumentChunk
from document_rag.retrieval.dense import (
    DenseEmbedder,
    FloatMatrix,
    normalize_embedding_matrix,
)
from document_rag.retrieval.embeddings import DOCUMENT_ENCODING_STRATEGY

EMBEDDINGS_FILENAME = "embeddings.npy"
EMBEDDING_MANIFEST_FILENAME = "embedding_manifest.json"
_CACHE_SCHEMA_VERSION = "1"


@dataclass(frozen=True, slots=True)
class CachedEmbeddings:
    """Normalized embedding matrix and its cache provenance."""

    embeddings: FloatMatrix
    embeddings_path: Path
    manifest_path: Path
    reused: bool


class DocumentEmbeddingCache:
    """Persist float32 unit vectors with strict provenance validation."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory.resolve()

    def load_or_encode(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        embedder: DenseEmbedder,
        batch_size: int,
        corpus_sha256: str,
    ) -> CachedEmbeddings:
        """Reuse a matching cache or encode and atomically replace it."""

        if not chunks:
            raise ValueError("At least one chunk is required for embedding cache.")

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if not corpus_sha256.strip():
            raise ValueError("corpus_sha256 must not be empty.")

        self._directory.mkdir(parents=True, exist_ok=True)
        embeddings_path = self._directory / EMBEDDINGS_FILENAME
        manifest_path = self._directory / EMBEDDING_MANIFEST_FILENAME
        expected_manifest = self._build_manifest(
            chunks=chunks,
            embedder=embedder,
            batch_size=batch_size,
            corpus_sha256=corpus_sha256,
        )
        cached = self._load_matching_cache(
            embeddings_path=embeddings_path,
            manifest_path=manifest_path,
            expected_manifest=expected_manifest,
        )

        if cached is not None:
            return CachedEmbeddings(
                embeddings=cached,
                embeddings_path=embeddings_path,
                manifest_path=manifest_path,
                reused=True,
            )

        embeddings = normalize_embedding_matrix(
            embedder.embed_documents(
                tuple(chunk.text for chunk in chunks),
                batch_size=batch_size,
            ),
            expected_rows=len(chunks),
            expected_dimension=embedder.dimension,
            label="Document",
        )
        self._write_cache(
            embeddings_path=embeddings_path,
            manifest_path=manifest_path,
            embeddings=embeddings,
            manifest=expected_manifest,
        )
        return CachedEmbeddings(
            embeddings=embeddings,
            embeddings_path=embeddings_path,
            manifest_path=manifest_path,
            reused=False,
        )

    def _build_manifest(
        self,
        *,
        chunks: Sequence[DocumentChunk],
        embedder: DenseEmbedder,
        batch_size: int,
        corpus_sha256: str,
    ) -> dict[str, object]:
        return {
            "batch_size": batch_size,
            "chunk_count": len(chunks),
            "chunk_ids_sha256": _hash_chunk_ids(chunks),
            "corpus_sha256": corpus_sha256,
            "device": embedder.device,
            "document_encoding_strategy": DOCUMENT_ENCODING_STRATEGY,
            "dtype": "float32",
            "embedding_dimension": embedder.dimension,
            "model_id": embedder.model_id,
            "model_revision": embedder.model_revision,
            "normalization": "l2_unit",
            "schema_version": _CACHE_SCHEMA_VERSION,
        }

    def _load_matching_cache(
        self,
        *,
        embeddings_path: Path,
        manifest_path: Path,
        expected_manifest: dict[str, object],
    ) -> FloatMatrix | None:
        if not embeddings_path.is_file() or not manifest_path.is_file():
            return None

        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            if raw_manifest != expected_manifest:
                return None

            matrix = np.load(embeddings_path, allow_pickle=False)
        except (OSError, ValueError, EOFError, json.JSONDecodeError, UnicodeDecodeError):
            return None

        if matrix.dtype != np.float32:
            return None

        expected_shape = (
            cast(int, expected_manifest["chunk_count"]),
            cast(int, expected_manifest["embedding_dimension"]),
        )

        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            return None

        norms = np.linalg.norm(matrix, axis=1)

        if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
            return None

        return np.ascontiguousarray(matrix, dtype=np.float32)

    def _write_cache(
        self,
        *,
        embeddings_path: Path,
        manifest_path: Path,
        embeddings: FloatMatrix,
        manifest: dict[str, object],
    ) -> None:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{embeddings_path.name}.",
            suffix=".tmp",
            dir=self._directory,
            delete=False,
        ) as temporary_embeddings_file:
            temporary_embeddings_path = Path(temporary_embeddings_file.name)
            np.save(
                temporary_embeddings_file,
                embeddings,
                allow_pickle=False,
            )
            temporary_embeddings_file.flush()

        manifest_bytes = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=self._directory,
            delete=False,
        ) as temporary_manifest_file:
            temporary_manifest_path = Path(temporary_manifest_file.name)
            temporary_manifest_file.write(manifest_bytes)
            temporary_manifest_file.flush()

        try:
            os.replace(temporary_embeddings_path, embeddings_path)
            os.replace(temporary_manifest_path, manifest_path)
        finally:
            temporary_embeddings_path.unlink(missing_ok=True)
            temporary_manifest_path.unlink(missing_ok=True)


def _hash_chunk_ids(chunks: Sequence[DocumentChunk]) -> str:
    digest = sha256()

    for chunk in chunks:
        encoded = chunk.chunk_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(length=8, byteorder="big"))
        digest.update(encoded)

    return digest.hexdigest()
