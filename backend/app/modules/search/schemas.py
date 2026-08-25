import uuid
from typing import Literal

from pydantic import BaseModel, Field

from app.modules.files.schemas import FileRead


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
    limit: int = Field(default=50, ge=1, le=200)
    # Opaque token from a previous response's next_cursor. Omitted/None = first page.
    cursor: str | None = None


class SearchFilesResponse(BaseModel):
    items: list[FileRead]
    total: int
    limit: int
    # Pass back as `cursor` to fetch the next page; None = no more pages.
    next_cursor: str | None
