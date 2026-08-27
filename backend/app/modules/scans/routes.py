import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.modules.files import service as files_service
from app.modules.files.schemas import FileRead
from app.modules.jobs.models import Job
from app.modules.jobs.schemas import JobRead
from app.modules.processing.tasks import run_ingest_job
from app.modules.scans import rescan, service
from app.modules.scans.schemas import (
    LabelReviewRequest,
    LabelReviewResult,
    RescanCandidateResolve,
    RescanCandidateResolveResult,
    RescanRead,
    ScanCreate,
)
from app.modules.scans.tasks import run_rescan_job, run_scan_job
from app.shared.database import get_db
from app.shared.locking import acquire_mutation_lock
from app.shared.queue import JOB_RETRY, ingest_queue, scan_queue

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


@router.post("/rescans", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
def create_rescan(db: Session = Depends(get_db)) -> Job:
    """ADR-0001b D1: full precondition gate, checked and inserted under the
    same advisory lock path/job mutations already use (acquire_mutation_lock)
    so a concurrent path change or job creation can't slip between the check
    and this job's insert.
    """
    acquire_mutation_lock(db)

    registered_paths = files_service.list_paths(db)
    if not registered_paths:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="No registered paths; register a path before running Rescan"
        )
    unscanned = service.list_unscanned_paths(db)
    if unscanned:
        names = ", ".join(p.path for p in unscanned)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"The following paths have not completed an initial scan yet: {names}",
        )
    if service.has_pending_candidates(db):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Resolve all pending recovery candidates before starting a new Rescan",
        )
    if service.has_active_job(db):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another scan, ingest, label, or embed job is already active",
        )

    job = service.create_rescan(db)
    rq_job = scan_queue.enqueue(run_rescan_job, job.id)
    job.rq_job_id = str(rq_job.id)
    db.commit()
    return job


@router.get("/rescans/{job_id}", response_model=RescanRead)
def get_rescan(job_id: uuid.UUID, db: Session = Depends(get_db)) -> RescanRead:
    job = service.get_rescan(db, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rescan not found")

    job_data = JobRead.model_validate(job).model_dump()
    return RescanRead(
        **job_data,
        event_counts=service.get_rescan_event_counts(db, job.id),
        pending_candidate_count=service.count_pending_candidates(db, job.id),
    )


@router.post("/rescan-candidates/{candidate_id}/resolve", response_model=RescanCandidateResolveResult)
def resolve_rescan_candidate(
    candidate_id: uuid.UUID, payload: RescanCandidateResolve, db: Session = Depends(get_db)
) -> RescanCandidateResolveResult:
    candidate = service.get_candidate(db, candidate_id)
    if candidate is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recovery candidate not found")
    if candidate.status != "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Recovery candidate already resolved")

    try:
        file, ingest_job = rescan.resolve_candidate(db, candidate, payload.action)
    except rescan.SnapshotMismatch as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from None

    rq_job = ingest_queue.enqueue(run_ingest_job, ingest_job.id, retry=JOB_RETRY)
    ingest_job.rq_job_id = str(rq_job.id)
    db.commit()

    return RescanCandidateResolveResult(
        candidate_id=candidate.id, action=payload.action, file=FileRead.model_validate(file)
    )


@router.post("/files/{file_id}/label-review", response_model=LabelReviewResult)
def resolve_label_review(
    file_id: uuid.UUID, payload: LabelReviewRequest, db: Session = Depends(get_db)
) -> LabelReviewResult:
    file = files_service.get_file(db, file_id)
    if file is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    if not file.labels_need_review:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="File has no pending label review")

    event = service.get_last_content_change_event(db, file_id)
    file = service.resolve_label_review(db, file, payload.action)

    return LabelReviewResult(
        file=FileRead.model_validate(file),
        from_hash=event.from_hash if event is not None else None,
        to_hash=event.to_hash if event is not None else None,
    )


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
