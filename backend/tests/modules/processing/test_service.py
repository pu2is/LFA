import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.processing.models import ProcessingJob
from app.modules.processing.service import create_processing_job, process_file
from app.modules.rag.models import FileChunk

TEST_DIR = Path("E:/lfa_test")
TEST_PDF = TEST_DIR / "onfilmlab_rechnung_62123151.pdf"

pytestmark = pytest.mark.skipif(
    not TEST_PDF.exists(),
    reason="E:\\lfa_test fixture PDF not present on this machine",
)


@pytest.fixture
def registered_file(db):
    path = RegisteredPath(path=str(TEST_DIR.resolve()))
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename=TEST_PDF.name,
        full_path=str(TEST_PDF.resolve()),
        file_type="pdf",
        file_size=TEST_PDF.stat().st_size,
        file_hash=hashlib.sha256(TEST_PDF.read_bytes()).hexdigest(),
        file_modified_at=datetime.fromtimestamp(TEST_PDF.stat().st_mtime, tz=timezone.utc),
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    yield file


def test_process_file_creates_chunks_from_text_layer_pdf(db, registered_file):
    job = create_processing_job(db, registered_file.id, triggered_by="manual")

    result = process_file(db, job.id)

    # No llm injected → uses real ChatOllama; requires Ollama to be running.
    assert result.status == "succeeded"
    assert result.started_at is not None
    assert result.completed_at is not None

    chunks = db.scalars(select(FileChunk).where(FileChunk.file_id == registered_file.id)).all()
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.content
        assert chunk.embedding is None

    db.refresh(registered_file)
    assert registered_file.status == "ready"
