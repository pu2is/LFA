import uuid
from datetime import datetime
from pathlib import Path as FsPath

from pydantic import BaseModel, ConfigDict, field_validator


class PathCreate(BaseModel):
    path: str

    @field_validator("path")
    @classmethod
    def validate_and_normalize_path(cls, value: str) -> str:
        fs_path = FsPath(value)
        if not fs_path.is_dir():
            raise ValueError(f"Path does not exist or is not a directory: {value}")
        return str(fs_path.resolve())


class PathRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    path: str
    created_at: datetime
    last_scanned_at: datetime | None
