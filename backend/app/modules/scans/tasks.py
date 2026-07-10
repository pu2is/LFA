import uuid

from app.modules.processing.tasks import run_ingest_job
from app.modules.scans import service
from app.shared.database import SessionLocal
from app.shared.queue import JOB_RETRY, ingest_queue


def run_scan_job(scan_id: uuid.UUID) -> None:
    """RQ entrypoint for a scan.

    After file discovery, enqueues one ingest job per discovered file. The
    ingest Job rows are already committed by service.run_scan; this only
    records each rq_job_id, in one commit after the whole fan-out so the
    bookkeeping update is atomic across the batch (see #33).
    """
    db = SessionLocal()
    try:
        _scan_job, ingest_jobs = service.run_scan(db, scan_id)
        for ingest_job in ingest_jobs:
            rq_job = ingest_queue.enqueue(run_ingest_job, ingest_job.id, retry=JOB_RETRY)
            ingest_job.rq_job_id = str(rq_job.id)
        db.commit()
    finally:
        db.close()
