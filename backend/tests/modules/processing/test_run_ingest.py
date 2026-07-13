"""run_ingest's job-lifecycle bookkeeping (extraction/cleaning/chunking
themselves are covered in their own module tests)."""
from unittest.mock import patch

import pytest

from app.modules.processing.extraction import ExtractionResult
from app.modules.processing.service import run_ingest
from tests.factories import make_failed_job


@pytest.fixture
def ingest_job(db):
    return make_failed_job(db, job_type="ingest", file_status="processing", error_message="file was locked")


@patch("app.modules.processing.service.rag_service.chunk_and_store")
@patch("app.modules.processing.service.extraction.extract_text")
def test_run_ingest_clears_stale_error_message_on_success(mock_extract, mock_chunk_and_store, db, ingest_job):
    # Chunking is unrelated to what this test checks (job-lifecycle bookkeeping)
    # and would otherwise delete+re-insert real file_chunks rows for nothing.
    mock_extract.return_value = ExtractionResult(text="Some extracted document text.", ocr_applied=False)
    mock_chunk_and_store.return_value = []

    ingest_job_result, embed_job = run_ingest(db, ingest_job.id)

    assert ingest_job_result.status == "succeeded"
    assert ingest_job_result.error_message is None
    assert embed_job is not None
    mock_chunk_and_store.assert_called_once()
