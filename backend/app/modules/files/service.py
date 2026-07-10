import uuid
from datetime import datetime

from sqlalchemy import func, select, true
from sqlalchemy.orm import Session

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job

# Job types surfaced in the files list as "processing status": ingest and
# label are the two file-level pipeline steps with no other visibility (embed
# has its own files.embedding_status column; scan is path-level -- Job.file_id
# is NULL for scans, see ck_jobs_target -- so it can't join to a file anyway).
_FILE_STATUS_JOB_TYPES = ("ingest", "label")


def get_path_by_value(db: Session, path: str) -> RegisteredPath | None:
    return db.scalar(select(RegisteredPath).where(RegisteredPath.path == path))


def get_path(db: Session, path_id: uuid.UUID) -> RegisteredPath | None:
    return db.get(RegisteredPath, path_id)


def list_paths(db: Session) -> list[RegisteredPath]:
    return list(db.scalars(select(RegisteredPath)))


def create_path(db: Session, path: str) -> RegisteredPath:
    registered_path = RegisteredPath(path=path)
    db.add(registered_path)
    db.commit()
    db.refresh(registered_path)
    return registered_path


def delete_path(db: Session, registered_path: RegisteredPath) -> None:
    db.delete(registered_path)
    db.commit()


def get_file_by_full_path(db: Session, full_path: str) -> File | None:
    return db.scalar(select(File).where(File.full_path == full_path))


def list_files(
    db: Session, path_id: uuid.UUID | None = None
) -> list[tuple[File, str | None, str | None, str | None]]:
    """List files, each joined with its most recent ingest-or-label job.

    One LATERAL join per file (not one scalar subquery per projected column)
    so Postgres runs the "latest job" lookup once and reuses the
    ix_jobs_file_type_created index, instead of evaluating the same
    correlated subquery twice (once for status, once for error_message).
    """
    latest_job = (
        select(Job.status, Job.error_message, Job.type)
        .where(Job.file_id == File.id, Job.type.in_(_FILE_STATUS_JOB_TYPES))
        .order_by(Job.created_at.desc())
        .limit(1)
        .correlate(File)
        .lateral("latest_job")
    )
    stmt = select(
        File,
        latest_job.c.status.label("processing_job_status"),
        latest_job.c.error_message.label("processing_error_message"),
        latest_job.c.type.label("processing_job_type"),
    ).outerjoin(latest_job, true())
    if path_id is not None:
        stmt = stmt.where(File.path_id == path_id)
    return list(db.execute(stmt))


def count_files_by_path(db: Session, path_id: uuid.UUID) -> int:
    return db.scalar(
        select(func.count()).select_from(File).where(File.path_id == path_id)
    )


def upsert_file(
    db: Session,
    *,
    path_id: uuid.UUID,
    full_path: str,
    filename: str,
    file_type: str,
    file_size: int,
    file_hash: str,
    file_created_at: datetime | None,
    file_modified_at: datetime,
) -> File:
    """Insert a new file row, or refresh filesystem metadata on an existing one.

    Matched by `full_path` (unique). Deliberately leaves `status` and
    `ocr_applied` untouched on existing rows: those reflect later pipeline
    stages (OCR, labeling) that a scan must not silently reset, even if the
    file's content changed (see 03_er-diagram.md "modified" handling).
    """
    file = get_file_by_full_path(db, full_path)
    if file is not None:
        file.path_id = path_id
        file.filename = filename
        file.file_type = file_type
        file.file_size = file_size
        file.file_hash = file_hash
        file.file_created_at = file_created_at
        file.file_modified_at = file_modified_at
        return file

    file = File(
        path_id=path_id,
        filename=filename,
        full_path=full_path,
        file_type=file_type,
        file_size=file_size,
        file_hash=file_hash,
        file_created_at=file_created_at,
        file_modified_at=file_modified_at,
    )
    db.add(file)
    return file
