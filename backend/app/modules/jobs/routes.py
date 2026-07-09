"""Read-only jobs API.

Pairs with the SSE stream (GET /events/stream): the client loads this snapshot
once, then applies incremental job_status deltas. The DB is the source of truth
(see the events module's snapshot+stream contract), so on a reconnect gap the
client just re-fetches this endpoint.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.jobs.schemas import JobListRead
from app.shared.database import get_db

router = APIRouter(tags=["jobs"])

_ACTIVE = ("queued", "running")
_TERMINAL = ("succeeded", "failed")


def _query_jobs(
    db: Session, status_filter: ColumnElement[bool], limit: int | None
) -> list[JobListRead]:
    # coalesce: scan jobs have a path (no file), everything else has a file.
    target_name = func.coalesce(File.filename, RegisteredPath.path).label("target_name")
    stmt = (
        select(Job, target_name)
        .outerjoin(File, File.id == Job.file_id)
        .outerjoin(RegisteredPath, RegisteredPath.id == Job.path_id)
        .where(status_filter)
        .order_by(Job.created_at.desc())
    )
    if limit is not None:
        stmt = stmt.limit(limit)

    items: list[JobListRead] = []
    for job, name in db.execute(stmt):
        item = JobListRead.model_validate(job)
        item.target_name = name
        items.append(item)
    return items


@router.get("/jobs", response_model=list[JobListRead])
def list_jobs(limit: int = 30, db: Session = Depends(get_db)) -> list[JobListRead]:
    """Processing-table snapshot: all active jobs + the most recent terminal jobs.

    Active jobs (queued/running) are returned in full so nothing in flight is
    hidden; terminal jobs (succeeded/failed) are capped at `limit` for history.
    """
    active = _query_jobs(db, Job.status.in_(_ACTIVE), limit=None)
    recent = _query_jobs(db, Job.status.in_(_TERMINAL), limit=limit)
    return active + recent
