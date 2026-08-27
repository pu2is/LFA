import uuid

from app.modules.jobs.models import Job
from app.modules.jobs.service import mark_failed, mark_running, mark_succeeded
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


def run_rescan_job(scan_id: uuid.UUID) -> None:
    """RQ entrypoint for a global Rescan (WF1b, ADR-0001b D6).

    Serves both the initial run and a resume: if the job hasn't reached
    stage='fan_out' yet, runs inventory/diff/apply first (service.run_rescan
    owns that failure handling internally and returns either way); once at
    'fan_out' -- either just now, or because this call *is* the resume, see
    routes.py's POST /rescans/{job_id}/resume -- only outstanding children
    get enqueued.

    Unlike run_scan_job's ingest fan-out (#33, one commit at the end),
    each child's rq_job_id is committed immediately after its enqueue call
    succeeds: if a later child's enqueue then fails, the children already
    queued must not show rq_job_id IS NULL, or a retry/resume would enqueue
    them a second time (ADR-0001b D6's idempotent-resume requirement).
    """
    db = SessionLocal()
    try:
        scan_job = db.get(Job, scan_id)
        if scan_job.stage != "fan_out":
            scan_job = service.run_rescan(db, scan_id)
            if scan_job.status == "failed":
                return
        else:
            mark_running(db, scan_job)

        for child in service.get_pending_fan_out_jobs(db, scan_job):
            try:
                rq_job = ingest_queue.enqueue(run_ingest_job, child.id, retry=JOB_RETRY)
            except Exception as exc:
                mark_failed(db, scan_job, exc)
                return
            child.rq_job_id = str(rq_job.id)
            db.commit()

        mark_succeeded(db, scan_job)
    finally:
        db.close()
