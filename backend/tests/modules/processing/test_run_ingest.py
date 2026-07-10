"""run_ingest's job-lifecycle bookkeeping (extraction/cleaning/chunking
themselves are covered in their own module tests)."""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.processing.extraction import ExtractionResult
from app.modules.processing.service import run_ingest


@pytest.fixture
def ingest_job(db):
    path = RegisteredPath(path="/processing-run-ingest-fixture")
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="sample.pdf",
        full_path="/processing-run-ingest-fixture/sample.pdf",
        file_type="pdf",
        file_size=1000,
        file_hash="c" * 64,
        file_modified_at=datetime.now(timezone.utc),
        status="processing",
    )
    db.add(file)
    db.flush()

    job = Job(
        type="ingest",
        file_id=file.id,
        trigger="scan",
        # Simulates a prior failed attempt that RQ is now retrying (#33).
        status="failed",
        error_message="file was locked",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    yield job


@patch("app.modules.processing.service.extraction.extract_text")
def test_run_ingest_clears_stale_error_message_on_success(mock_extract, db, ingest_job):
    mock_extract.return_value = ExtractionResult(text="Some extracted document text.", ocr_applied=False)

    ingest_job_result, embed_job = run_ingest(db, ingest_job.id)

    assert ingest_job_result.status == "succeeded"
    assert ingest_job_result.error_message is None
    assert embed_job is not None
