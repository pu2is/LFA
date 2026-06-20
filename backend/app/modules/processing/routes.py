import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from rq.job import Retry
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.files import service as files_service
from app.modules.files.models import File
from app.modules.processing import service
from app.modules.processing.schemas import (
    ProcessByPathIdsRequest,
    ProcessFilesByIdsRequest,
    ProcessingJobRead,
    RelabelRequest,
)
from app.modules.processing.tasks import run_processing_job, run_relabel_job
from app.shared.database import get_db
from app.shared.queue import labeling_queue

# Retry for resource errors (Ollama unreachable): 3 attempts, back-off 1 min / 2 min / 5 min.
_RETRY = Retry(max=3, interval=[60, 120, 300])

router = APIRouter(prefix="/process", tags=["processing"])


@router.post(
    "/files",
    response_model=list[ProcessingJobRead],
    status_code=status.HTTP_202_ACCEPTED,
)
def process_files(payload: ProcessFilesByIdsRequest, db: Session = Depends(get_db)):
    """Enqueue processing jobs for the given file IDs.

    Only creates jobs; the actual extract→chunk pipeline runs in the RQ worker.
    Returns 404 if any file_id is not found (validated before any job is created).
    Files that are not in 'discovered' status are silently skipped.
    """
    files = []
    for file_id in payload.file_ids:
        file = db.get(File, file_id)
        if file is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {file_id} not found",
            )
        if file.status == "discovered":
            files.append(file)

    jobs = []
    for file in files:
        job = service.create_processing_job(db, file.id, triggered_by="manual")
        rq_job = labeling_queue.enqueue(run_processing_job, job.id, retry=_RETRY)
        job.rq_job_id = str(rq_job.id)
        db.commit()
        db.refresh(job)
        jobs.append(job)

    return jobs


@router.post(
    "/paths",
    response_model=list[ProcessingJobRead],
    status_code=status.HTTP_202_ACCEPTED,
)
def process_paths(payload: ProcessByPathIdsRequest, db: Session = Depends(get_db)):
    """Enqueue processing jobs for all discovered files under the given path IDs.

    Only files with status='discovered' are picked up; already-processed files
    are skipped. Returns 404 if any path_id is not found.
    """
    for path_id in payload.path_ids:
        if files_service.get_path(db, path_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Path {path_id} not found",
            )

    files = list(
        db.scalars(
            select(File)
            .where(File.path_id.in_(payload.path_ids))
            .where(File.status == "discovered")
        )
    )

    jobs = []
    for file in files:
        job = service.create_processing_job(db, file.id, triggered_by="manual")
        rq_job = labeling_queue.enqueue(run_processing_job, job.id, retry=_RETRY)
        job.rq_job_id = str(rq_job.id)
        db.commit()
        db.refresh(job)
        jobs.append(job)

    return jobs


@router.post(
    "/relabel",
    response_model=list[ProcessingJobRead],
    status_code=status.HTTP_202_ACCEPTED,
)
def relabel_files(payload: RelabelRequest, db: Session = Depends(get_db)):
    """Re-run label suggestion on already-processed files, merging with existing labels.

    Only files with status='ready' are enqueued; others are silently skipped.
    Returns 404 if any file_id is not found.
    Existing confirmed labels are preserved; rejected labels are reset to suggested.
    """
    files = []
    for file_id in payload.file_ids:
        file = db.get(File, file_id)
        if file is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {file_id} not found",
            )
        if file.status == "ready":
            files.append(file)

    jobs = []
    for file in files:
        job = service.create_processing_job(db, file.id, triggered_by="manual")
        rq_job = labeling_queue.enqueue(run_relabel_job, job.id, retry=_RETRY)
        job.rq_job_id = str(rq_job.id)
        db.commit()
        db.refresh(job)
        jobs.append(job)

    return jobs


@router.get("/{job_id}", response_model=ProcessingJobRead)
def get_processing_job(job_id: uuid.UUID, db: Session = Depends(get_db)):
    """Fetch the current status of a processing job."""
    job = service.get_processing_job(db, job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processing job not found",
        )
    return job
