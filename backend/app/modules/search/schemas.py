import uuid

from pydantic import BaseModel


class TagFacetRead(BaseModel):
    kind_id: uuid.UUID
    kind_name: str
    values: list[str]
