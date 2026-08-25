import base64
import binascii
import json
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.modules.files import service as files_service
from app.modules.files.schemas import FileRead
from app.modules.labeling import service as labeling_service
from app.modules.search.schemas import SearchFilesRequest, SearchFilesResponse, TagFacetRead
from app.shared.database import get_db

router = APIRouter(tags=["search"])


def _encode_cursor(file: FileRead, sort_by: str, sort_order: str) -> str:
    """Opaque token for the keyset position after `file` in this sort. Embeds
    sort_by/sort_order so a cursor built under one sort can't silently be
    replayed against another (see _decode_cursor)."""
    value = getattr(file, sort_by)
    payload = {
        "sort_by": sort_by,
        "sort_order": sort_order,
        "v": value.isoformat() if isinstance(value, datetime) else value,
        "id": str(file.id),
    }
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def _decode_cursor(cursor: str, sort_by: str, sort_order: str) -> tuple[Any, uuid.UUID]:
    """Also validates the payload's shape (not just its base64/JSON framing):
    `id` must be a string (uuid.UUID raises AttributeError, not ValueError,
    on a non-string argument -- easy to miss in the except tuple below), and
    `v` must match sort_by's column type (files_service.sortable_column_python_type)
    so a type-mismatched cursor value 422s here instead of reaching the DB
    unvalidated and surfacing as a raw driver error (#58 fix)."""
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        if payload["sort_by"] != sort_by or payload["sort_order"] != sort_order:
            raise ValueError("cursor was issued for a different sort_by/sort_order")
        if not isinstance(payload["id"], str):
            raise ValueError("cursor id must be a string")

        value = payload["v"]
        expected_type = files_service.sortable_column_python_type(sort_by)
        if value is None:
            pass
        elif expected_type is datetime:
            value = datetime.fromisoformat(value)
        elif not isinstance(value, expected_type):
            raise ValueError("cursor value does not match sort_by column's type")

        return value, uuid.UUID(payload["id"])
    except (ValueError, KeyError, TypeError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Invalid cursor") from None


@router.get("/search/tag-facets", response_model=list[TagFacetRead])
def get_tag_facets(
    type_label_ids: list[uuid.UUID] = Query(default=[]),
    db: Session = Depends(get_db),
):
    """WF2a (ADR-0002a D3): candidate-set tag facets for the manual filter's
    type->tag narrowing step. Empty type_label_ids = full candidate set."""
    for type_label_id in type_label_ids:
        if labeling_service.get_type_label(db, type_label_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Type label {type_label_id} not found")

    facets = labeling_service.get_tag_facets(db, type_label_ids)
    return [TagFacetRead(kind_id=kind.id, kind_name=kind.name, values=values) for kind, values in facets]


@router.post("/search/files", response_model=SearchFilesResponse)
def search_files(payload: SearchFilesRequest, db: Session = Depends(get_db)):
    """WF2a (ADR-0002a D3 / #57, pagination #58): execute the final type/tag
    filter and return one page of matching files, sorted. An empty
    type_label_ids/tags list means that group is unrestricted (D2); both
    empty returns the full file list (paginated)."""
    for type_label_id in payload.type_label_ids:
        if labeling_service.get_type_label(db, type_label_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Type label {type_label_id} not found")
    for tag in payload.tags:
        if labeling_service.get_tag_kind(db, tag.kind_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Tag kind {tag.kind_id} not found")

    type_ids_subquery = (
        labeling_service.get_file_ids_by_confirmed_types_subquery(payload.type_label_ids)
        if payload.type_label_ids
        else None
    )
    tag_ids_subquery = (
        labeling_service.get_file_ids_by_confirmed_tags_subquery(
            [(tag.kind_id, tag.value) for tag in payload.tags]
        )
        if payload.tags
        else None
    )

    total = files_service.count_files(db, type_ids_subquery=type_ids_subquery, tag_ids_subquery=tag_ids_subquery)

    cursor = _decode_cursor(payload.cursor, payload.sort_by, payload.sort_order) if payload.cursor else None
    page = files_service.list_files_page(
        db,
        type_ids_subquery=type_ids_subquery,
        tag_ids_subquery=tag_ids_subquery,
        sort_by=payload.sort_by,
        sort_order=payload.sort_order,
        cursor=cursor,
        limit=payload.limit,
    )

    has_more = len(page) > payload.limit
    items = page[: payload.limit]
    next_cursor = _encode_cursor(items[-1], payload.sort_by, payload.sort_order) if has_more and items else None

    return SearchFilesResponse(items=items, total=total, limit=payload.limit, next_cursor=next_cursor)
