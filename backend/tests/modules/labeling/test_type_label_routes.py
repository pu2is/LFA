"""Tests for the type-label catalog CRUD and /files/{id}/type-labels review
endpoints (#44, ADR-0001 / 01x). Mirrors test coverage the old /labels and
/files/{id}/labels endpoints never had, for the new symmetric type facet.
"""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.labeling.models import TypeLabel, TypeLabelFile


@pytest.fixture
def file_id(db):
    path = RegisteredPath(path="/tmp/lfa_type_label_routes_test")
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="sample.pdf",
        full_path="/tmp/lfa_type_label_routes_test/sample.pdf",
        file_type="pdf",
        file_size=100,
        file_hash="type-label-routes-test",
        file_modified_at=datetime.now(timezone.utc),
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file.id


@pytest.fixture
def type_label(db):
    row = TypeLabel(name="invoice")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# --------------------------------------------------------------------------- #
# Type-label catalog CRUD: GET/POST/DELETE /type-labels
# --------------------------------------------------------------------------- #

def test_list_type_labels_empty(client):
    resp = client.get("/type-labels")
    assert resp.status_code == 200
    assert resp.json() == []


def test_create_type_labels_bulk_and_normalizes_names(client, db):
    resp = client.post("/type-labels", json={"names": ["Invoice", "Bank Statement"]})

    assert resp.status_code == 201
    data = resp.json()
    assert {t["name"] for t in data["created"]} == {"invoice", "bank_statement"}
    assert data["skipped"] == []
    assert len(list(db.scalars(select(TypeLabel)))) == 2


def test_create_type_labels_skips_existing(client, type_label):
    resp = client.post("/type-labels", json={"names": ["invoice", "contract"]})

    assert resp.status_code == 201
    data = resp.json()
    assert [t["name"] for t in data["created"]] == ["contract"]
    assert data["skipped"] == ["invoice"]


def test_create_type_labels_rejects_blank_name(client):
    resp = client.post("/type-labels", json={"names": ["   "]})
    assert resp.status_code == 422


def test_delete_type_label_success(client, type_label):
    resp = client.delete(f"/type-labels/{type_label.id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == str(type_label.id)

    assert client.get("/type-labels").json() == []


def test_delete_type_label_not_found(client):
    resp = client.delete(f"/type-labels/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_delete_type_label_in_use_returns_409(client, db, file_id, type_label):
    db.add(TypeLabelFile(file_id=file_id, type_label_id=type_label.id, source="user", status="confirmed"))
    db.commit()

    resp = client.delete(f"/type-labels/{type_label.id}")

    assert resp.status_code == 409


# --------------------------------------------------------------------------- #
# Type-label file review: GET/PATCH/POST/DELETE /files/{id}/type-labels
# --------------------------------------------------------------------------- #

def test_list_file_type_labels_requires_existing_file(client):
    resp = client.get(f"/files/{uuid.uuid4()}/type-labels")
    assert resp.status_code == 404


def test_list_file_type_labels_empty(client, file_id):
    resp = client.get(f"/files/{file_id}/type-labels")
    assert resp.status_code == 200
    assert resp.json() == []


def test_add_user_type_label_success(client, file_id, type_label):
    resp = client.post(f"/files/{file_id}/type-labels", json={"type_label_id": str(type_label.id)})

    assert resp.status_code == 201
    data = resp.json()
    assert data["type_label_id"] == str(type_label.id)
    assert data["source"] == "user"
    assert data["status"] == "confirmed"


def test_add_user_type_label_unknown_type_404(client, file_id):
    resp = client.post(f"/files/{file_id}/type-labels", json={"type_label_id": str(uuid.uuid4())})
    assert resp.status_code == 404


def test_add_user_type_label_duplicate_is_idempotent_200(client, db, file_id, type_label):
    """#50: a repeat manual add is not an error -- it's a no-op confirm."""
    first = client.post(f"/files/{file_id}/type-labels", json={"type_label_id": str(type_label.id)})
    assert first.status_code == 201

    resp = client.post(f"/files/{file_id}/type-labels", json={"type_label_id": str(type_label.id)})

    assert resp.status_code == 200
    assert resp.json()["id"] == first.json()["id"]  # same row, not a new one
    assert len(list(db.scalars(select(TypeLabelFile).where(TypeLabelFile.file_id == file_id)))) == 1


def test_add_user_type_label_flips_rejected_to_confirmed_200(client, db, file_id, type_label):
    """#50: the dead end -- LLM suggested, user rejected, user manually re-adds
    -- must now succeed (200), not 409 with no way forward."""
    rejected = TypeLabelFile(file_id=file_id, type_label_id=type_label.id, source="llm", status="rejected")
    db.add(rejected)
    db.commit()
    db.refresh(rejected)

    resp = client.post(f"/files/{file_id}/type-labels", json={"type_label_id": str(type_label.id)})

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == str(rejected.id)
    assert data["status"] == "confirmed"
    assert data["source"] == "llm"  # provenance untouched -- only status flips


def test_patch_confirms_suggested_type_label(client, db, file_id, type_label):
    row = TypeLabelFile(file_id=file_id, type_label_id=type_label.id, source="llm", status="suggested")
    db.add(row)
    db.commit()
    db.refresh(row)

    resp = client.patch(
        f"/files/{file_id}/type-labels",
        json={"operations": [{"type_label_file_id": str(row.id), "action": "confirm"}]},
    )

    assert resp.status_code == 200
    assert resp.json()[0]["status"] == "confirmed"


def test_patch_rejects_suggested_type_label(client, db, file_id, type_label):
    row = TypeLabelFile(file_id=file_id, type_label_id=type_label.id, source="llm", status="suggested")
    db.add(row)
    db.commit()
    db.refresh(row)

    resp = client.patch(
        f"/files/{file_id}/type-labels",
        json={"operations": [{"type_label_file_id": str(row.id), "action": "reject"}]},
    )

    assert resp.status_code == 200
    assert resp.json()[0]["status"] == "rejected"


def test_patch_batch_is_all_or_nothing(client, db, file_id, type_label):
    row = TypeLabelFile(file_id=file_id, type_label_id=type_label.id, source="llm", status="suggested")
    db.add(row)
    db.commit()
    db.refresh(row)

    resp = client.patch(
        f"/files/{file_id}/type-labels",
        json={
            "operations": [
                {"type_label_file_id": str(row.id), "action": "confirm"},
                {"type_label_file_id": str(uuid.uuid4()), "action": "confirm"},
            ]
        },
    )

    assert resp.status_code == 404
    db.refresh(row)
    assert row.status == "suggested"  # untouched -- the bad id rolled back the whole batch


def test_delete_file_type_label_success(client, file_id, type_label):
    add_resp = client.post(f"/files/{file_id}/type-labels", json={"type_label_id": str(type_label.id)})
    row_id = add_resp.json()["id"]

    resp = client.delete(f"/files/{file_id}/type-labels/{row_id}")

    assert resp.status_code == 200
    assert client.get(f"/files/{file_id}/type-labels").json() == []


def test_delete_file_type_label_not_found(client, file_id):
    resp = client.delete(f"/files/{file_id}/type-labels/{uuid.uuid4()}")
    assert resp.status_code == 404
