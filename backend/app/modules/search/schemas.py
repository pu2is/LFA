import uuid
from typing import Literal

from pydantic import BaseModel, Field


class TagFacetRead(BaseModel):
    kind_id: uuid.UUID
    kind_name: str
    values: list[str]


class TagSelector(BaseModel):
    kind_id: uuid.UUID
    value: str


class SearchFilesRequest(BaseModel):
    type_label_ids: list[uuid.UUID] = Field(default_factory=list)
    tags: list[TagSelector] = Field(default_factory=list)
    sort_by: Literal["filename", "file_type", "file_size", "file_created_at", "file_modified_at"] = "file_modified_at"
    sort_order: Literal["asc", "desc"] = "desc"
