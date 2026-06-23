import uuid

from app.modules.processing.tasks import run_ingest_job
from app.modules.scans import service
from app.shared.database import SessionLocal
from app.shared.queue import ingest_queue


def run_scan_job(scan_id: uuid.UUID) -> None:
    """RQ entrypoint for a scan.

    After file discovery, enqueues one ingest job per discovered file.
    """
    db = SessionLocal()
    try:
        _scan_job, ingest_jobs = service.run_scan(db, scan_id)
        for ingest_job in ingest_jobs:
            ingest_queue.enqueue(run_ingest_job, ingest_job.id)
    finally:
        db.close()
