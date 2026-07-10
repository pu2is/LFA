"""run_embed's job-lifecycle bookkeeping (separate from embed_file's own logic,
covered in test_embedding.py)."""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.rag.service import run_embed


@pytest.fixture
def embed_job(db):
    path = RegisteredPath(path="/rag-run-embed-fixture")
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="sample.pdf",
        full_path="/rag-run-embed-fixture/sample.pdf",
        file_type="pdf",
        file_size=1000,
        file_hash="b" * 64,
        file_modified_at=datetime.now(timezone.utc),
    )
    db.add(file)
    db.flush()

    job = Job(
        type="embed",
        file_id=file.id,
        trigger="scan",
        # Simulates a prior failed attempt that RQ is now retrying (#33).
        status="failed",
        error_message="Ollama unreachable",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    yield job


def test_run_embed_clears_stale_error_message_on_success(db, embed_job):
    result = run_embed(db, embed_job.id, embeddings=MagicMock())

    assert result.status == "succeeded"
    assert result.error_message is None
