import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.modules.files import service as files_service
from app.modules.jobs.models import Job
from app.modules.jobs.schemas import JobRead
from app.modules.scans import service
from app.modules.scans.schemas import ScanCreate
from app.modules.scans.tasks import run_rescan_job, run_scan_job
from app.shared.database import get_db
from app.shared.queue import scan_queue

router = APIRouter(tags=["scans"])


@router.post("/scans", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
def create_scan(payload: ScanCreate, db: Session = Depends(get_db)) -> Job:
    if files_service.get_path(db, payload.path_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Path not found")

    scan = service.create_scan(db, payload.path_id)
    rq_job = scan_queue.enqueue(run_scan_job, scan.id)
    scan.rq_job_id = str(rq_job.id)
    db.commit()
    return scan


@router.get("/scans/{scan_id}", response_model=JobRead)
def get_scan(scan_id: uuid.UUID, db: Session = Depends(get_db)) -> Job:
    scan = service.get_scan(db, scan_id)
    if scan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")
    return scan


@router.post("/rescans/{job_id}/resume", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
def resume_rescan(job_id: uuid.UUID, db: Session = Depends(get_db)) -> Job:
    """ADR-0001b D6: only a Rescan stuck at status=failed, stage=fan_out is
    resumable -- apply already committed (the manifest is in sync), only
    child-job enqueueing didn't finish. Re-enqueues the same run_rescan_job
    entrypoint used at creation time, which skips straight to fan-out once
    job.stage is already 'fan_out'.
    """
    job = service.get_rescan(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rescan not found")
    if job.status != "failed" or job.stage != "fan_out":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Rescan is not resumable")

    job.status = "queued"
    rq_job = scan_queue.enqueue(run_rescan_job, job.id)
    job.rq_job_id = str(rq_job.id)
    db.commit()
    return job
