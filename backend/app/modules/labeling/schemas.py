import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.modules.labeling.service import normalize_label_name


# --- Label job request/response schemas ---

class LabelByFileIdsRequest(BaseModel):
    file_ids: list[uuid.UUID] = Field(min_length=1)


class LabelByPathIdsRequest(BaseModel):
    path_ids: list[uuid.UUID] = Field(min_length=1)


class LabelJobEnqueued(BaseModel):
    job_id: uuid.UUID
    file_id: uuid.UUID
    mode: str


class LabelJobSkipped(BaseModel):
    file_id: uuid.UUID
    reason: str


class LabelJobResult(BaseModel):
    enqueued: list[LabelJobEnqueued]
    skipped: list[LabelJobSkipped]


class LabelBulkCreate(BaseModel):
    names: list[str] = Field(min_length=1)

    @field_validator("names")
    @classmethod
    def normalize_names(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            cleaned = normalize_label_name(value)
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


# --- Type-label catalog CRUD schemas (ADR-0001) ---

class TypeLabelBulkCreate(BaseModel):
    names: list[str] = Field(min_length=1)

    @field_validator("names")
    @classmethod
    def normalize_names(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            cleaned = normalize_label_name(value)
            if not cleaned:
                raise ValueError("Type label name must not be empty or whitespace")
            normalized.append(cleaned)
        return normalized


class TypeLabelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    created_at: datetime


class TypeLabelBulkCreateResult(BaseModel):
    created: list[TypeLabelRead]
    skipped: list[str]


# --- Tag-kind catalog CRUD schemas (ADR-0001) ---

class TagKindBulkCreate(BaseModel):
    names: list[str] = Field(min_length=1)

    @field_validator("names")
    @classmethod
    def normalize_names(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            cleaned = normalize_label_name(value)
            if not cleaned:
                raise ValueError("Tag kind name must not be empty or whitespace")
            normalized.append(cleaned)
        return normalized


class TagKindRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    created_at: datetime


class TagKindBulkCreateResult(BaseModel):
    created: list[TagKindRead]
    skipped: list[str]


# --- Type-label file review schemas (ADR-0001 / 01x) ---

class TypeLabelFileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    file_id: uuid.UUID
    type_label_id: uuid.UUID
    source: str
    status: str
    created_at: datetime
    updated_at: datetime


class TypeLabelFilePatchOp(BaseModel):
    type_label_file_id: uuid.UUID
    action: Literal["confirm", "reject"]


class TypeLabelFileBatchPatch(BaseModel):
    operations: list[TypeLabelFilePatchOp] = Field(min_length=1)


class TypeLabelFileAdd(BaseModel):
    type_label_id: uuid.UUID


# --- Tag-label file review schemas (ADR-0001 / 01x) ---

class TagLabelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    file_id: uuid.UUID
    kind_id: uuid.UUID
    value: str
    source: str
    status: str
    created_at: datetime
    updated_at: datetime


class TagLabelPatchOp(BaseModel):
    tag_label_id: uuid.UUID
    action: Literal["confirm", "reject"]


class TagLabelBatchPatch(BaseModel):
    operations: list[TagLabelPatchOp] = Field(min_length=1)


class TagLabelAdd(BaseModel):
    kind_id: uuid.UUID
    value: str

    @field_validator("value")
    @classmethod
    def strip_value(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Tag value must not be empty or whitespace")
        return cleaned
