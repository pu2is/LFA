"""run_ingest_job's rq_job_id bookkeeping for the embed fan-out (#33).

service.run_ingest is mocked entirely (no real extraction/DB work) -- see
tests/modules/scans/test_tasks.py for why the mocked return value is a plain,
session-less Job instance rather than something fetched from a real session.
"""
import uuid
from unittest.mock import MagicMock, patch

from app.modules.jobs.models import Job
from app.modules.processing.tasks import run_ingest_job
from tests.conftest import mock_rq_job


@patch("app.modules.processing.tasks.embed_queue")
@patch("app.modules.processing.tasks.service")
def test_run_ingest_job_records_rq_job_id_on_embed_job(mock_service, mock_queue):
    job_id = uuid.uuid4()
    embed_job = Job(id=uuid.uuid4(), type="embed", file_id=uuid.uuid4(), trigger="scan")
    mock_service.run_ingest.return_value = (MagicMock(), embed_job)
    mock_queue.enqueue.return_value = mock_rq_job("rq-embed-1")

    run_ingest_job(job_id)

    mock_service.run_ingest.assert_called_once()
    assert embed_job.rq_job_id == "rq-embed-1"


@patch("app.modules.processing.tasks.embed_queue")
@patch("app.modules.processing.tasks.service")
def test_run_ingest_job_skips_enqueue_when_ingest_failed(mock_service, mock_queue):
    job_id = uuid.uuid4()
    mock_service.run_ingest.return_value = (MagicMock(), None)

    run_ingest_job(job_id)

    mock_queue.enqueue.assert_not_called()
