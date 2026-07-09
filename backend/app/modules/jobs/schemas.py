import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    type: str
    path_id: uuid.UUID | None
    file_id: uuid.UUID | None
    parent_job_id: uuid.UUID | None
    status: str
    stage: str | None
    trigger: str
    mode: str
    error_message: str | None
    rq_job_id: str | None
    attempts: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class JobListRead(JobRead):
    """JobRead plus a human-readable target (filename, or path for scans).

    Used by the processing-table snapshot so the UI can show what each job is
    working on without a second round-trip per job.
    """

    target_name: str | None = None
