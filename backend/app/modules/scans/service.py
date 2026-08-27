import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.modules.files import service as files_service
from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.jobs.service import mark_failed, mark_progress, mark_running
from app.modules.labeling.service import clear_all_labels
from app.modules.scans import discovery, rescan
from app.modules.scans.models import FileEvent, FileMatchCandidate
from app.shared.events import publish_job_status

# Same active-status pair jobs/routes.py uses for the processing-table snapshot.
_ACTIVE_JOB_STATUSES = ("queued", "running")


def create_scan(db: Session, path_id: uuid.UUID) -> Job:
    job = Job(type="scan", path_id=path_id, trigger="scan", mode="initial")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def list_unscanned_paths(db: Session) -> list[RegisteredPath]:
    """Registered paths that have never completed an initial scan (ADR-0001b
    D1 precondition 1) -- POST /rescans rejects until every path has one."""
    return [p for p in files_service.list_paths(db) if p.last_scanned_at is None]


def has_pending_candidates(db: Session) -> bool:
    """ADR-0001b D1 precondition 2: any unresolved fuzzy recovery candidate,
    from any past Rescan, blocks a new one."""
    return db.scalar(select(FileMatchCandidate.id).where(FileMatchCandidate.status == "pending").limit(1)) is not None


def has_active_job(db: Session) -> bool:
    """ADR-0001b D1 precondition 3: any queued/running scan/ingest/label/embed
    job blocks a new Rescan -- this also covers "no other active Rescan"
    since that is itself a Job row (the partial unique index on
    ix_jobs_active_rescan is the DB-level backstop for the same rule)."""
    return db.scalar(select(Job.id).where(Job.status.in_(_ACTIVE_JOB_STATUSES)).limit(1)) is not None


def create_rescan(db: Session) -> Job:
    job = Job(type="scan", mode="rescan", trigger="manual")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_candidate(db: Session, candidate_id: uuid.UUID) -> FileMatchCandidate | None:
    return db.get(FileMatchCandidate, candidate_id)


def get_rescan_event_counts(db: Session, scan_id: uuid.UUID) -> dict[str, int]:
    stmt = (
        select(FileEvent.event_type, func.count())
        .where(FileEvent.scan_id == scan_id)
        .group_by(FileEvent.event_type)
    )
    return {event_type: count for event_type, count in db.execute(stmt)}


def count_pending_candidates(db: Session, scan_id: uuid.UUID) -> int:
    return db.scalar(
        select(func.count()).select_from(FileMatchCandidate).where(
            FileMatchCandidate.scan_id == scan_id, FileMatchCandidate.status == "pending",
        )
    )


def get_last_content_change_event(db: Session, file_id: uuid.UUID) -> FileEvent | None:
    """Most recent modified/moved_modified event for a file -- the one that
    set labels_need_review=true -- so label-review can surface its
    from_hash/to_hash (ADR-0001b D4's "no automatic drift judgment, just show
    the user what changed")."""
    return db.scalar(
        select(FileEvent)
        .where(FileEvent.file_id == file_id, FileEvent.event_type.in_(("modified", "moved_modified")))
        .order_by(FileEvent.created_at.desc())
        .limit(1)
    )


def resolve_label_review(db: Session, file: File, action: str) -> File:
    if action == "drop":
        clear_all_labels(db, file.id)
    file.labels_need_review = False
    db.commit()
    db.refresh(file)
    return file


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

    mark_running(db, scan_job)

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
        mark_failed(db, scan_job, exc)
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


def get_rescan(db: Session, scan_id: uuid.UUID) -> Job | None:
    job = db.get(Job, scan_id)
    if job is not None and (job.type != "scan" or job.mode != "rescan"):
        return None
    return job


def run_rescan(db: Session, scan_id: uuid.UUID) -> Job:
    """Run one global Rescan (WF1b, ADR-0001b): inventory -> diff -> apply,
    each phase's start recorded as job.stage before it runs. An inventory or
    diff failure (an unreadable root, a file that kept changing mid-hash)
    fails the job with zero manifest changes (D2); an apply failure rolls
    back the whole transaction (D6). Neither is resumable -- a fresh Rescan
    is the recovery path for both, same as a plain scan failure.

    Returns before fan-out: enqueueing the child ingest jobs this creates is
    tasks.py's job, since that's D6's separate, resumable stage.
    """
    scan_job = db.get(Job, scan_id)
    if scan_job is None:
        raise ValueError(f"Job {scan_id} not found")

    # ADR-0001b D1: the registered path set is fixed once, here, at the start
    # of the run -- not re-queried per root during the walk.
    registered_paths = files_service.list_paths(db)
    mark_running(db, scan_job)

    scan_job.stage = "inventory"
    mark_progress(db, scan_job)
    try:
        inventory = rescan.build_inventory(registered_paths)
    except OSError as exc:
        mark_failed(db, scan_job, exc)
        return scan_job

    scan_job.stage = "diff"
    mark_progress(db, scan_job)
    current_files = list(db.scalars(select(File)))
    try:
        diff = rescan.diff_inventory(inventory, current_files)
    except OSError as exc:
        mark_failed(db, scan_job, exc)
        return scan_job

    scan_job.stage = "apply"
    mark_progress(db, scan_job)
    try:
        rescan.apply_diff(db, scan_job, diff, registered_paths)
        scan_job.stage = "fan_out"
        db.commit()
    except Exception as exc:
        db.rollback()
        mark_failed(db, scan_job, exc)
        return scan_job

    publish_job_status(scan_job)
    return scan_job


def get_pending_fan_out_jobs(db: Session, scan_job: Job) -> list[Job]:
    """Child ingest jobs for this Rescan that still need enqueueing (ADR-0001b
    D6). rq_job_id IS NULL covers both the first fan-out attempt and a
    resumed one uniformly: apply_diff creates every child job row up front
    but never enqueues them itself, so "pending" always just means "not
    enqueued yet", regardless of which attempt is asking.
    """
    return list(db.scalars(
        select(Job).where(
            Job.parent_job_id == scan_job.id,
            Job.type == "ingest",
            Job.rq_job_id.is_(None),
        )
    ))
