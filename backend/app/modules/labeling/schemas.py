import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LabelBulkCreate(BaseModel):
    names: list[str] = Field(min_length=1)

    @field_validator("names")
    @classmethod
    def normalize_names(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            cleaned = value.strip().lower()
            if not cleaned:
                raise ValueError("Label name must not be empty or whitespace")
            normalized.append(cleaned)
        return normalized


class LabelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    created_at: datetime


class LabelBulkCreateResult(BaseModel):
    created: list[LabelRead]
    skipped: list[str]


class PresetRead(BaseModel):
    name: str
    recommended: bool
