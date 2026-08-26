import uuid
from datetime import datetime
from pathlib import Path as FsPath
from typing import Any

from sqlalchemy import Select, and_, func, or_, select, text, true
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
# only nullable one (see File model) -- driving list_files_page's NULLS LAST
# and the extra branch in _keyset_condition.
_SORTABLE_COLUMNS = {
    "filename": File.filename,
    "file_type": File.file_type,
    "file_size": File.file_size,
    "file_created_at": File.file_created_at,
    "file_modified_at": File.file_modified_at,
}


def sortable_column_python_type(sort_by: str) -> type:
    """Python type of a sort_by column's values (str/int/datetime) -- lets
    search/routes.py's cursor decode validate a cursor's 'v' against the
    column it claims to sort by, without hand-maintaining a second list of
    which sortable columns are which type (#58 fix: that second list is what
    let a mismatched cursor value reach the DB unvalidated)."""
    return _SORTABLE_COLUMNS[sort_by].type.python_type


def acquire_path_mutation_lock(db: Session) -> None:
    """Block until this transaction holds the exclusive gate for path
    mutations (#62). Every path-mutating endpoint must call this before any
    check-then-write (duplicate/ancestor check then insert, or delete), so
    two concurrent requests can never both pass a check before either
    commits -- e.g. `/a` and `/a/b` both seeing "no conflict" and registering
    as unrelated roots (see docs/workflow/00a-path-register.md "已知邊界").

    Uses `pg_advisory_xact_lock`, not the session-scoped `pg_advisory_lock`,
    so the lock is released automatically on this transaction's commit or
    rollback -- no manual unlock, and no leak on an exception path.
    """
    db.execute(text("SELECT pg_advisory_xact_lock(hashtext('lfa:paths:mutation'))"))


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
    list_files_page so both get the same processing-status enrichment.

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


def _apply_id_filters(
    stmt: Select,
    *,
    type_ids_subquery: Select | None,
    tag_ids_subquery: Select | None,
) -> Select:
    """AND in the (optional) type/tag candidate-set subqueries from
    labeling.service. Each is an unexecuted SELECT of file_id -- folding
    them in here means Postgres evaluates the whole filter+sort+page in one
    query, instead of files.service receiving pre-materialized id sets (see
    docs/workflow/02a1_smart-search-pagination.md, #58)."""
    if type_ids_subquery is not None:
        stmt = stmt.where(File.id.in_(type_ids_subquery))
    if tag_ids_subquery is not None:
        stmt = stmt.where(File.id.in_(tag_ids_subquery))
    return stmt


def count_files(
    db: Session,
    *,
    type_ids_subquery: Select | None = None,
    tag_ids_subquery: Select | None = None,
) -> int:
    """WF2a (ADR-0002a D3 / #57, pagination #58): count of files matching the
    type/tag candidate-set subqueries (both None = every file). Independent
    of any page/cursor -- always reflects the full filtered set."""
    stmt = _apply_id_filters(
        select(func.count()).select_from(File),
        type_ids_subquery=type_ids_subquery,
        tag_ids_subquery=tag_ids_subquery,
    )
    return db.scalar(stmt)


def _keyset_condition(column, cursor_value: Any, cursor_id: uuid.UUID, ascending: bool):
    """WHERE condition for 'rows strictly after (cursor_value, cursor_id)' in
    this column's NULLS-LAST order, for seek/keyset pagination.

    file_created_at is the only nullable sortable column, and NULLS LAST
    means null rows sort after every non-null value regardless of asc/desc --
    so the condition has to branch on whether the cursor itself already sits
    in that trailing null group (cursor_value is None: only later nulls,
    tie-broken by id) or is still among non-null values (cursor_value is not
    None: rows that progressed past it, OR the same value tie-broken by id,
    OR the null group entirely -- since once non-null values are exhausted,
    NULLS LAST puts every null row next regardless of value).
    """
    if cursor_value is None:
        return and_(column.is_(None), File.id > cursor_id)

    same_value_tiebreak = and_(column == cursor_value, File.id > cursor_id)
    progressed = column > cursor_value if ascending else column < cursor_value
    return or_(progressed, same_value_tiebreak, column.is_(None))


def list_files_page(
    db: Session,
    *,
    type_ids_subquery: Select | None = None,
    tag_ids_subquery: Select | None = None,
    sort_by: str = "file_modified_at",
    sort_order: str = "desc",
    cursor: tuple[Any, uuid.UUID] | None = None,
    limit: int = 50,
) -> list[FileRead]:
    """WF2a (ADR-0002a D3 / #57, keyset pagination #58): fetch up to
    limit + 1 rows (the extra row lets the caller detect a next page exists
    without a separate query) narrowed to the type/tag candidate-set
    subqueries (both None = every file), sorted by a caller-chosen column,
    resuming after `cursor` -- the (sort_value, id) of the last item on the
    previous page; None = first page, i.e. just take the first rows.

    Seek/keyset pagination, not OFFSET: OFFSET makes Postgres scan and
    discard every preceding row on each call, so cost grows with page depth.
    `WHERE (sort_col, id) > (cursor_val, cursor_id)` lets it seek directly
    via the sort column's index instead, regardless of how deep the page is.

    NULLS LAST: file_created_at is the only nullable sortable column, so
    without it an ascending sort would put never-populated dates first
    (Postgres's default), burying real ones behind them. The id tiebreak
    (both here and in _keyset_condition) makes ordering deterministic when
    the sort column has duplicate values -- required for pages to stay
    disjoint, not just for single-query ordering.
    """
    column = _SORTABLE_COLUMNS[sort_by]
    ascending = sort_order == "asc"
    order = column.asc() if ascending else column.desc()

    stmt = _apply_id_filters(
        _files_with_processing_status_stmt(),
        type_ids_subquery=type_ids_subquery,
        tag_ids_subquery=tag_ids_subquery,
    )
    if cursor is not None:
        cursor_value, cursor_id = cursor
        stmt = stmt.where(_keyset_condition(column, cursor_value, cursor_id, ascending))

    stmt = stmt.order_by(order.nulls_last(), File.id).limit(limit + 1)
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
