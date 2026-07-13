import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.files import service as files_service
from app.modules.files.models import File
from app.modules.jobs.models import Job
from app.modules.scans import discovery
from app.shared.events import publish_job_status


def create_scan(db: Session, path_id: uuid.UUID) -> Job:
    job = Job(type="scan", path_id=path_id, trigger="scan", mode="initial")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_scan(db: Session, scan_id: uuid.UUID) -> Job | None:
    job = db.get(Job, scan_id)
    if job is not None and job.type != "scan":
        return None
    return job


def run_scan(db: Session, scan_id: uuid.UUID) -> tuple[Job, list[Job]]:
    """Walk the scan's registered path, upsert File rows, and fan-out ingest jobs.

    Returns the scan job and a list of ingest jobs created for discovered files.
    The caller (RQ task) is responsible for enqueuing the ingest jobs.
    """
    scan_job = db.get(Job, scan_id)
    if scan_job is None:
        raise ValueError(f"Job {scan_id} not found")

    registered_path = files_service.get_path(db, scan_job.path_id)
    if registered_path is None:
        raise ValueError(f"Registered path {scan_job.path_id} not found")

    scan_job.status = "running"
    scan_job.started_at = datetime.now(timezone.utc)
    db.commit()
    publish_job_status(scan_job)

    child_paths = files_service.get_child_paths(db, registered_path.id)
    exclude_roots = frozenset(Path(child.path) for child in child_paths)

    try:
        for doc in discovery.iter_documents(Path(registered_path.path), exclude_roots=exclude_roots):
            files_service.upsert_file(
                db,
                path_id=registered_path.id,
                full_path=str(doc.path),
                filename=doc.path.name,
                file_type=doc.file_type,
                file_size=doc.file_size,
                file_hash=doc.file_hash,
                file_created_at=doc.file_created_at,
                file_modified_at=doc.file_modified_at,
            )
    except OSError as exc:
        scan_job.status = "failed"
        scan_job.error_message = str(exc)
        scan_job.completed_at = datetime.now(timezone.utc)
        db.commit()
        publish_job_status(scan_job)
        return scan_job, []

    file_count = files_service.count_files_by_path(db, scan_job.path_id)
    scan_job.status = "succeeded"
    scan_job.completed_at = datetime.now(timezone.utc)
    registered_path.last_scanned_at = scan_job.completed_at
    db.commit()
    publish_job_status(scan_job, file_count=file_count)

    # Fan-out: create one ingest job per discovered file under this path.
    discovered_files = list(db.scalars(
        select(File)
        .where(File.path_id == scan_job.path_id, File.status == "discovered")
    ))
    ingest_jobs: list[Job] = []
    for file in discovered_files:
        ingest_job = Job(
            type="ingest",
            file_id=file.id,
            parent_job_id=scan_job.id,
            trigger="scan",
        )
        db.add(ingest_job)
        ingest_jobs.append(ingest_job)

    if ingest_jobs:
        db.commit()
        for job in ingest_jobs:
            db.refresh(job)

    return scan_job, ingest_jobs
