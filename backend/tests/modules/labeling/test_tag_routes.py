"""Tests for the tag-kind catalog CRUD and /files/{id}/tags review endpoints
(#44, ADR-0001 / 01x). Mirrors test_type_label_routes.py -- same confirm/
reject/manual-add/delete/error semantics, applied to the tag facet.
"""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.labeling.models import TagKind, TagLabel


@pytest.fixture
def file_id(db):
    path = RegisteredPath(path="/tmp/lfa_tag_routes_test")
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="sample.pdf",
        full_path="/tmp/lfa_tag_routes_test/sample.pdf",
        file_type="pdf",
        file_size=100,
        file_hash="tag-routes-test",
        file_modified_at=datetime.now(timezone.utc),
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file.id


@pytest.fixture
def tag_kind(db):
    row = TagKind(name="person")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# --------------------------------------------------------------------------- #
# Tag-kind catalog CRUD: GET/POST/DELETE /tag-kinds
# --------------------------------------------------------------------------- #

def test_list_tag_kinds_empty(client):
    resp = client.get("/tag-kinds")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_tag_kinds_bulk_and_normalizes_names(client, db):
    resp = client.post("/tag-kinds", json={"names": ["Person", "Organization"]})

    assert resp.status_code == 201
    data = resp.json()
    assert {k["name"] for k in data["created"]} == {"person", "organization"}
    assert data["skipped"] == []
    assert len(list(db.scalars(select(TagKind)))) == 2


def test_create_tag_kinds_skips_existing(client, tag_kind):
    resp = client.post("/tag-kinds", json={"names": ["person", "place"]})

    assert resp.status_code == 201
    data = resp.json()
    assert [k["name"] for k in data["created"]] == ["place"]
    assert data["skipped"] == ["person"]


def test_create_tag_kinds_rejects_blank_name(client):
    resp = client.post("/tag-kinds", json={"names": ["   "]})
    assert resp.status_code == 422


def test_delete_tag_kind_success(client, tag_kind):
    resp = client.delete(f"/tag-kinds/{tag_kind.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == str(tag_kind.id)

    assert client.get("/tag-kinds").json() == []


def test_delete_tag_kind_not_found(client):
    resp = client.delete(f"/tag-kinds/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_delete_tag_kind_in_use_returns_409(client, db, file_id, tag_kind):
    db.add(TagLabel(file_id=file_id, kind_id=tag_kind.id, value="Berlin", source="user", status="confirmed"))
    db.commit()

    resp = client.delete(f"/tag-kinds/{tag_kind.id}")

    assert resp.status_code == 409


# --------------------------------------------------------------------------- #
# Tag file review: GET/PATCH/POST/DELETE /files/{id}/tags
# --------------------------------------------------------------------------- #

def test_list_file_tags_requires_existing_file(client):
    resp = client.get(f"/files/{uuid.uuid4()}/tags")
    assert resp.status_code == 404


def test_list_file_tags_empty(client, file_id):
    resp = client.get(f"/files/{file_id}/tags")
    assert resp.status_code == 200
    assert resp.json() == []


def test_add_user_tag_success(client, file_id, tag_kind):
    resp = client.post(f"/files/{file_id}/tags", json={"kind_id": str(tag_kind.id), "value": "Berlin"})

    assert resp.status_code == 201
    data = resp.json()
    assert data["kind_id"] == str(tag_kind.id)
    assert data["value"] == "Berlin"
    assert data["source"] == "user"
    assert data["status"] == "confirmed"


def test_add_user_tag_unknown_kind_404(client, file_id):
    resp = client.post(f"/files/{file_id}/tags", json={"kind_id": str(uuid.uuid4()), "value": "Berlin"})
    assert resp.status_code == 404


def test_add_user_tag_duplicate_409(client, file_id, tag_kind):
    client.post(f"/files/{file_id}/tags", json={"kind_id": str(tag_kind.id), "value": "Berlin"})
    resp = client.post(f"/files/{file_id}/tags", json={"kind_id": str(tag_kind.id), "value": "Berlin"})
    assert resp.status_code == 409


def test_add_user_tag_rejects_blank_value(client, file_id, tag_kind):
    resp = client.post(f"/files/{file_id}/tags", json={"kind_id": str(tag_kind.id), "value": "   "})
    assert resp.status_code == 422


def test_patch_confirms_suggested_tag(client, db, file_id, tag_kind):
    row = TagLabel(file_id=file_id, kind_id=tag_kind.id, value="Berlin", source="llm", status="suggested")
    db.add(row)
    db.commit()
    db.refresh(row)

    resp = client.patch(
        f"/files/{file_id}/tags",
        json={"operations": [{"tag_label_id": str(row.id), "action": "confirm"}]},
    )

    assert resp.status_code == 200
    assert resp.json()[0]["status"] == "confirmed"


def test_patch_rejects_suggested_tag(client, db, file_id, tag_kind):
    row = TagLabel(file_id=file_id, kind_id=tag_kind.id, value="Berlin", source="llm", status="suggested")
    db.add(row)
    db.commit()
    db.refresh(row)

    resp = client.patch(
        f"/files/{file_id}/tags",
        json={"operations": [{"tag_label_id": str(row.id), "action": "reject"}]},
    )

    assert resp.status_code == 200
    assert resp.json()[0]["status"] == "rejected"


def test_patch_batch_is_all_or_nothing(client, db, file_id, tag_kind):
    row = TagLabel(file_id=file_id, kind_id=tag_kind.id, value="Berlin", source="llm", status="suggested")
    db.add(row)
    db.commit()
    db.refresh(row)

    resp = client.patch(
        f"/files/{file_id}/tags",
        json={
            "operations": [
                {"tag_label_id": str(row.id), "action": "confirm"},
                {"tag_label_id": str(uuid.uuid4()), "action": "confirm"},
            ]
        },
    )

    assert resp.status_code == 404
    db.refresh(row)
    assert row.status == "suggested"  # untouched -- the bad id rolled back the whole batch


def test_delete_file_tag_success(client, file_id, tag_kind):
    add_resp = client.post(f"/files/{file_id}/tags", json={"kind_id": str(tag_kind.id), "value": "Berlin"})
    row_id = add_resp.json()["id"]

    resp = client.delete(f"/files/{file_id}/tags/{row_id}")

    assert resp.status_code == 200
    assert client.get(f"/files/{file_id}/tags").json() == []


def test_delete_file_tag_not_found(client, file_id):
    resp = client.delete(f"/files/{file_id}/tags/{uuid.uuid4()}")
    assert resp.status_code == 404
