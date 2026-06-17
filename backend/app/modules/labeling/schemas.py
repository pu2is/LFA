import uuid
from datetime import datetime
from typing import Literal

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


# --- File-label review schemas ---

class FileLabelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    file_id: uuid.UUID
    label_id: uuid.UUID | None  # None for LLM free-text labels
    label_name: str             # always set; copied from labels.name for catalog picks
    source: str
    status: str
    confidence: float | None
    created_at: datetime
    updated_at: datetime


class FileLabelPatchOp(BaseModel):
    # Uses file_labels.id (the row's own PK) to address any file_label,
    # regardless of whether it is a catalog label or a free-text label.
    file_label_id: uuid.UUID
    action: Literal["confirm", "reject"]


class FileLabelBatchPatch(BaseModel):
    operations: list[FileLabelPatchOp] = Field(min_length=1)


class FileLabelAdd(BaseModel):
    label_id: uuid.UUID
