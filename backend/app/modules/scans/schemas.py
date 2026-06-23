import uuid

from pydantic import BaseModel


class ScanCreate(BaseModel):
    path_id: uuid.UUID
