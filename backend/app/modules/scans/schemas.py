import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ScanCreate(BaseModel):
    path_id: uuid.UUID


class ScanRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    path_id: uuid.UUID
    status: str
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
