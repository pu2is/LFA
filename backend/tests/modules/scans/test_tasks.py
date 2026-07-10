"""run_scan_job's rq_job_id bookkeeping for the ingest fan-out (#33).

service.run_scan is mocked entirely (no real DB/filesystem work) so this only
exercises the enqueue-and-record loop in run_scan_job itself. The mock
returns plain, session-less Job instances -- run_scan_job's own SessionLocal()
session never has them added to it, so its db.commit() is a no-op for them;
the assertions read the same in-memory objects back directly rather than
re-querying, avoiding any cross-session/DetachedInstanceError pitfalls.
"""
import uuid
from unittest.mock import MagicMock, patch

from app.modules.jobs.models import Job
from app.modules.scans.tasks import run_scan_job


def _mock_rq_job(job_id: str) -> MagicMock:
    rq_job = MagicMock()
    rq_job.id = job_id
    return rq_job


@patch("app.modules.scans.tasks.ingest_queue")
@patch("app.modules.scans.tasks.service")
def test_run_scan_job_records_rq_job_id_for_each_fanned_out_ingest_job(mock_service, mock_queue):
    scan_id = uuid.uuid4()
    ingest_jobs = [
        Job(id=uuid.uuid4(), type="ingest", file_id=uuid.uuid4(), trigger="scan"),
        Job(id=uuid.uuid4(), type="ingest", file_id=uuid.uuid4(), trigger="scan"),
    ]
    mock_service.run_scan.return_value = (MagicMock(), ingest_jobs)
    mock_queue.enqueue.side_effect = [_mock_rq_job("rq-1"), _mock_rq_job("rq-2")]

    run_scan_job(scan_id)

    mock_service.run_scan.assert_called_once()
    assert ingest_jobs[0].rq_job_id == "rq-1"
    assert ingest_jobs[1].rq_job_id == "rq-2"
