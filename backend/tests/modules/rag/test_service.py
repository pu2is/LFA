from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.rag.models import FileChunk
from app.modules.rag.service import chunk_and_store, get_chunk_texts

# Long enough to produce multiple chunks (>CHUNK_SIZE chars).
SAMPLE_TEXT = "This is a sample sentence about document management. " * 30


@pytest.fixture
def test_file(db):
    path = RegisteredPath(path="/rag-test-fixture")
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="sample.pdf",
        full_path="/rag-test-fixture/sample.pdf",
        file_type="pdf",
        file_size=1000,
        file_hash="a" * 64,
        file_modified_at=datetime.now(timezone.utc),
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    yield file


def test_chunk_and_store_creates_chunks(db, test_file):
    chunks = chunk_and_store(db, test_file.id, SAMPLE_TEXT)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.content
        assert chunk.embedding is None


def test_chunk_and_store_assigns_sequential_index(db, test_file):
    chunks = chunk_and_store(db, test_file.id, SAMPLE_TEXT)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_and_store_is_idempotent(db, test_file):
    first_chunks = chunk_and_store(db, test_file.id, SAMPLE_TEXT)
    db.commit()
    first_ids = {c.id for c in first_chunks}

    second_chunks = chunk_and_store(db, test_file.id, SAMPLE_TEXT)
    db.commit()
    second_ids = {c.id for c in second_chunks}

    # Second run issued new rows -- no overlap with first run's IDs.
    assert not (first_ids & second_ids)

    # Only the second run's rows remain in the database.
    stored = db.scalars(select(FileChunk).where(FileChunk.file_id == test_file.id)).all()
    assert {c.id for c in stored} == second_ids


def test_get_chunk_texts_returns_content_in_index_order_and_honors_limit(db, test_file):
    db.add_all(
        [
            FileChunk(file_id=test_file.id, chunk_index=2, content="third"),
            FileChunk(file_id=test_file.id, chunk_index=0, content="first"),
            FileChunk(file_id=test_file.id, chunk_index=1, content="second"),
        ]
    )
    db.commit()

    assert get_chunk_texts(db, test_file.id) == ["first", "second", "third"]
    assert get_chunk_texts(db, test_file.id, limit=2) == ["first", "second"]
