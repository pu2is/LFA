import uuid
from datetime import datetime
from pathlib import Path as FsPath

from sqlalchemy import func, select, true
from sqlalchemy.orm import Session

from app.modules.files.models import File, RegisteredPath
from app.modules.files.schemas import FileRead
from app.modules.jobs.models import Job

# Job types surfaced in the files list as "processing status": ingest and
# label are the two file-level pipeline steps with no other visibility (embed
# has its own files.embedding_status column; scan is path-level -- Job.file_id
# is NULL for scans, see ck_jobs_target -- so it can't join to a file anyway).
_FILE_STATUS_JOB_TYPES = ("ingest", "label")

# WF2a (#57): columns POST /search/files may sort by. file_created_at is the
# only nullable one (see File model) -- driving list_files_by_ids's NULLS LAST.
_SORTABLE_COLUMNS = {
    "filename": File.filename,
    "file_type": File.file_type,
    "file_size": File.file_size,
    "file_created_at": File.file_created_at,
    "file_modified_at": File.file_modified_at,
}


def get_path_by_value(db: Session, path: str) -> RegisteredPath | None:
    return db.scalar(select(RegisteredPath).where(RegisteredPath.path == path))


def get_path(db: Session, path_id: uuid.UUID) -> RegisteredPath | None:
    return db.get(RegisteredPath, path_id)


def list_paths(db: Session) -> list[RegisteredPath]:
    return list(db.scalars(select(RegisteredPath)))


def get_child_paths(db: Session, parent_id: uuid.UUID) -> list[RegisteredPath]:
    """Direct registered children of `parent_id` (one level -- sufficient for
    scan-time pruning: any deeper registered descendant lies inside one of
    these children's subtree, see docs/workflow/00a-path-register.md).
    """
    return list(db.scalars(select(RegisteredPath).where(RegisteredPath.parent_path_id == parent_id)))


def find_ancestor_conflict(db: Session, resolved_path: str) -> RegisteredPath | None:
    """Return an existing registered path that `resolved_path` is nested under, if any.

    Segment-based comparison (`Path.is_relative_to`), not SQL LIKE prefix
    matching: "/home/coll" is not a parent of "/home/collection" but a
    string-prefix check would wrongly say it is.
    """
    candidate = FsPath(resolved_path)
    for existing in list_paths(db):
        if candidate != FsPath(existing.path) and candidate.is_relative_to(existing.path):
            return existing
    return None


def _adopt_orphans(db: Session, new_path: RegisteredPath) -> None:
    """Re-point existing orphan descendants of `new_path` to it.

    Paths that already have a parent are left untouched -- they already
    point at a nearer registered ancestor (see the nearest-ancestor
    invariant in docs/workflow/00a-path-register.md).
    """
    candidate = FsPath(new_path.path)
    for existing in list_paths(db):
        if (
            existing.id != new_path.id
            and existing.parent_path_id is None
            and FsPath(existing.path).is_relative_to(candidate)
        ):
            existing.parent_path_id = new_path.id


def create_path(db: Session, path: str) -> RegisteredPath:
    registered_path = RegisteredPath(path=path)
    db.add(registered_path)
    db.flush()
    _adopt_orphans(db, registered_path)
    db.commit()
    db.refresh(registered_path)
    return registered_path


def delete_path(db: Session, registered_path: RegisteredPath) -> None:
    db.delete(registered_path)
    db.commit()


def get_file_by_full_path(db: Session, full_path: str) -> File | None:
    return db.scalar(select(File).where(File.full_path == full_path))


def _files_with_processing_status_stmt():
    """Base SELECT: every File row joined with its most recent ingest-or-label
    job's status/error_message/type. Shared by list_files and
    list_files_by_ids so both get the same processing-status enrichment.

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
    return select(
        File,
        latest_job.c.status.label("processing_job_status"),
        latest_job.c.error_message.label("processing_error_message"),
        latest_job.c.type.label("processing_job_type"),
    ).outerjoin(latest_job, true())


def list_files(
    db: Session, path_id: uuid.UUID | None = None
) -> list[tuple[File, str | None, str | None, str | None]]:
    stmt = _files_with_processing_status_stmt()
    if path_id is not None:
        stmt = stmt.where(File.path_id == path_id)
    return list(db.execute(stmt))


def to_file_reads(rows: list[tuple[File, str | None, str | None, str | None]]) -> list[FileRead]:
    """Shape (File, processing_job_status, processing_error_message,
    processing_job_type) rows -- the shared shape produced by
    _files_with_processing_status_stmt -- into FileRead objects."""
    return [
        FileRead.model_validate(file).model_copy(update={
            "processing_job_status": job_status,
            "processing_error_message": error_msg,
            "processing_job_type": job_type,
        })
        for file, job_status, error_msg, job_type in rows
    ]


def list_files_by_ids(
    db: Session,
    file_ids: set[uuid.UUID] | None,
    *,
    sort_by: str = "file_modified_at",
    sort_order: str = "desc",
) -> list[FileRead]:
    """WF2a (ADR-0002a D3 / #57): fetch files narrowed to file_ids (None = no
    restriction, i.e. an unfiltered search) with the same processing-status
    enrichment as list_files, sorted by a caller-chosen column.

    NULLS LAST + id tiebreak: file_created_at is the only nullable sortable
    column, so without NULLS LAST an ascending sort would put never-populated
    dates first (Postgres's default), burying real ones behind them. The id
    tiebreak makes ordering deterministic across repeated calls when the sort
    column has duplicate values, instead of an arbitrary/unstable order.
    """
    if file_ids is not None and not file_ids:
        return []

    column = _SORTABLE_COLUMNS[sort_by]
    order = column.asc() if sort_order == "asc" else column.desc()

    stmt = _files_with_processing_status_stmt().order_by(order.nulls_last(), File.id)
    if file_ids is not None:
        stmt = stmt.where(File.id.in_(file_ids))
    return to_file_reads(list(db.execute(stmt)))


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
