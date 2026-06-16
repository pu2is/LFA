"""Tests for rag.service.embed_file() (Job2).

All tests use FakeEmbeddings — Ollama is never called.
"""
from datetime import datetime, timezone

import pytest
from sqlalchemy import delete, select

from app.modules.files.models import File, RegisteredPath
from app.modules.rag.models import EMBEDDING_DIMENSIONS, FileChunk
from app.modules.rag.service import embed_file


# --------------------------------------------------------------------------- #
# Fake embedding clients
# --------------------------------------------------------------------------- #

class FakeEmbeddings:
    """Returns deterministic 1024-dim zero vectors without calling Ollama."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]


class FailingEmbeddings:
    """Always raises, simulating Ollama being unreachable."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Ollama unreachable")


# --------------------------------------------------------------------------- #
# Fixture
# --------------------------------------------------------------------------- #

@pytest.fixture
def file_with_chunks(db):
    path = RegisteredPath(path="/tmp/lfa_embed_test")
    db.add(path)
    db.flush()

    f = File(
        path_id=path.id,
        filename="doc.pdf",
        full_path="/tmp/lfa_embed_test/doc.pdf",
        file_type="pdf",
        file_size=1000,
        file_hash="embed_test_hash",
        file_modified_at=datetime.now(timezone.utc),
    )
    db.add(f)
    db.flush()

    chunks = [
        FileChunk(file_id=f.id, chunk_index=0, content="First chunk content."),
        FileChunk(file_id=f.id, chunk_index=1, content="Second chunk content."),
    ]
    db.add_all(chunks)
    db.commit()
    db.refresh(f)

    yield f, chunks

    db.execute(delete(FileChunk).where(FileChunk.file_id == f.id))
    db.delete(f)
    db.delete(path)
    db.commit()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_embed_file_backfills_null_chunks(db, file_with_chunks):
    file, _ = file_with_chunks

    embed_file(db, file.id, embeddings=FakeEmbeddings())

    db.refresh(file)
    assert file.embedding_status == "done"

    stored = db.scalars(select(FileChunk).where(FileChunk.file_id == file.id)).all()
    assert len(stored) == 2
    for chunk in stored:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == EMBEDDING_DIMENSIONS


def test_embed_file_skips_already_embedded_chunks(db, file_with_chunks):
    file, chunks = file_with_chunks

    # Pre-embed the first chunk so it already has a vector.
    pre_vector = [1.0] * EMBEDDING_DIMENSIONS
    chunks[0].embedding = pre_vector
    db.commit()

    embed_file(db, file.id, embeddings=FakeEmbeddings())

    db.expire_all()
    stored = {
        c.chunk_index: c
        for c in db.scalars(select(FileChunk).where(FileChunk.file_id == file.id)).all()
    }
    # Pre-embedded chunk must be unchanged (list() converts the numpy array pgvector returns).
    assert list(stored[0].embedding) == pre_vector
    # Previously NULL chunk must now be filled.
    assert stored[1].embedding is not None
    assert list(stored[1].embedding) != pre_vector


def test_embed_file_marks_failed_on_error(db, file_with_chunks):
    file, _ = file_with_chunks

    with pytest.raises(RuntimeError, match="Ollama unreachable"):
        embed_file(db, file.id, embeddings=FailingEmbeddings())

    db.refresh(file)
    assert file.embedding_status == "failed"

    stored = db.scalars(select(FileChunk).where(FileChunk.file_id == file.id)).all()
    for chunk in stored:
        assert chunk.embedding is None
