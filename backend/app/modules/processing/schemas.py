import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ProcessFilesByIdsRequest(BaseModel):
    file_ids: list[uuid.UUID]


class RelabelRequest(BaseModel):
    file_ids: list[uuid.UUID]


class ProcessByPathIdsRequest(BaseModel):
    path_ids: list[uuid.UUID]


class ProcessingJobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    file_id: uuid.UUID
    status: str
    rq_job_id: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
