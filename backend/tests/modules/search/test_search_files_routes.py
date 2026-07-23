"""Tests for POST /search/files (#57, ADR-0002a D3, WF2a manual filter final
query: type/tag intersection, case-insensitive tag matching, sort + NULLS LAST)."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.modules.files.models import File, RegisteredPath
from app.modules.labeling.models import TagKind, TagLabel, TypeLabel, TypeLabelFile


@pytest.fixture
def make_file(db):
    def _make(name: str = "sample.pdf", *, file_created_at: datetime | None = None, file_size: int = 100):
        path = RegisteredPath(path=f"/tmp/lfa_search_files_test_{uuid.uuid4().hex}")
        db.add(path)
        db.flush()

        file = File(
            path_id=path.id,
            filename=name,
            full_path=f"{path.path}/{name}",
            file_type="pdf",
            file_size=file_size,
            file_hash=f"hash-{uuid.uuid4().hex}",
            file_created_at=file_created_at,
            file_modified_at=datetime.now(timezone.utc),
        )
        db.add(file)
        db.commit()
        db.refresh(file)
        return file

    return _make


@pytest.fixture
def tag_kind(db):
    row = TagKind(name="city")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@pytest.fixture
def type_label(db):
    row = TypeLabel(name="invoice")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _add_tag(db, file, kind, value, *, status="confirmed", source="user"):
    row = TagLabel(file_id=file.id, kind_id=kind.id, value=value, source=source, status=status)
    db.add(row)
    db.commit()
    return row


def _add_type(db, file, type_label_row, *, status="confirmed", source="user"):
    row = TypeLabelFile(file_id=file.id, type_label_id=type_label_row.id, source=source, status=status)
    db.add(row)
    db.commit()
    return row


def test_empty_request_returns_all_files(client, db, make_file):
    make_file("a.pdf")
    make_file("b.pdf")

    resp = client.post("/search/files", json={})

    assert resp.status_code == 200
    assert {f["filename"] for f in resp.json()} == {"a.pdf", "b.pdf"}


def test_filters_by_confirmed_type(client, db, make_file, type_label):
    other_type = TypeLabel(name="contract")
    db.add(other_type)
    db.commit()
    db.refresh(other_type)

    invoice_file = make_file("invoice.pdf")
    _add_type(db, invoice_file, type_label)
    contract_file = make_file("contract.pdf")
    _add_type(db, contract_file, other_type)

    resp = client.post("/search/files", json={"type_label_ids": [str(type_label.id)]})

    assert resp.status_code == 200
    assert [f["filename"] for f in resp.json()] == ["invoice.pdf"]


def test_excludes_non_confirmed_type_labels_file(client, db, make_file, type_label):
    file = make_file()
    _add_type(db, file, type_label, status="suggested", source="llm")

    resp = client.post("/search/files", json={"type_label_ids": [str(type_label.id)]})

    assert resp.status_code == 200
    assert resp.json() == []


def test_filters_by_tag_case_insensitive(client, db, make_file, tag_kind):
    berlin_file = make_file("berlin.pdf")
    _add_tag(db, berlin_file, tag_kind, "berlin")
    munich_file = make_file("munich.pdf")
    _add_tag(db, munich_file, tag_kind, "Munich")

    resp = client.post("/search/files", json={"tags": [{"kind_id": str(tag_kind.id), "value": "Berlin"}]})

    assert resp.status_code == 200
    assert [f["filename"] for f in resp.json()] == ["berlin.pdf"]


def test_excludes_non_confirmed_tags(client, db, make_file, tag_kind):
    file = make_file()
    _add_tag(db, file, tag_kind, "Berlin", status="rejected", source="llm")

    resp = client.post("/search/files", json={"tags": [{"kind_id": str(tag_kind.id), "value": "Berlin"}]})

    assert resp.status_code == 200
    assert resp.json() == []


def test_type_and_tag_are_intersected(client, db, make_file, tag_kind, type_label):
    matches_both = make_file("both.pdf")
    _add_type(db, matches_both, type_label)
    _add_tag(db, matches_both, tag_kind, "Berlin")

    matches_type_only = make_file("type_only.pdf")
    _add_type(db, matches_type_only, type_label)

    matches_tag_only = make_file("tag_only.pdf")
    _add_tag(db, matches_tag_only, tag_kind, "Berlin")

    resp = client.post(
        "/search/files",
        json={"type_label_ids": [str(type_label.id)], "tags": [{"kind_id": str(tag_kind.id), "value": "Berlin"}]},
    )

    assert resp.status_code == 200
    assert [f["filename"] for f in resp.json()] == ["both.pdf"]


def test_multiple_tags_are_or_within_group(client, db, make_file, tag_kind):
    berlin_file = make_file("berlin.pdf")
    _add_tag(db, berlin_file, tag_kind, "Berlin")
    munich_file = make_file("munich.pdf")
    _add_tag(db, munich_file, tag_kind, "Munich")
    paris_file = make_file("paris.pdf")
    _add_tag(db, paris_file, tag_kind, "Paris")

    resp = client.post(
        "/search/files",
        json={"tags": [{"kind_id": str(tag_kind.id), "value": "Berlin"}, {"kind_id": str(tag_kind.id), "value": "Munich"}]},
    )

    assert resp.status_code == 200
    assert {f["filename"] for f in resp.json()} == {"berlin.pdf", "munich.pdf"}


def test_file_matching_two_selected_types_appears_once(client, db, make_file, type_label):
    other_type = TypeLabel(name="contract")
    db.add(other_type)
    db.commit()
    db.refresh(other_type)

    file = make_file("both_types.pdf")
    _add_type(db, file, type_label)
    _add_type(db, file, other_type)

    resp = client.post("/search/files", json={"type_label_ids": [str(type_label.id), str(other_type.id)]})

    assert resp.status_code == 200
    assert [f["filename"] for f in resp.json()] == ["both_types.pdf"]


def test_unknown_type_label_id_returns_404(client):
    resp = client.post("/search/files", json={"type_label_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404


def test_unknown_tag_kind_id_returns_404(client):
    resp = client.post("/search/files", json={"tags": [{"kind_id": str(uuid.uuid4()), "value": "Berlin"}]})
    assert resp.status_code == 404


def test_unknown_tag_value_yields_empty_not_error(client, db, make_file, tag_kind):
    file = make_file()
    _add_tag(db, file, tag_kind, "Berlin")

    resp = client.post("/search/files", json={"tags": [{"kind_id": str(tag_kind.id), "value": "Nowhere"}]})

    assert resp.status_code == 200
    assert resp.json() == []


def test_invalid_sort_by_returns_422(client):
    resp = client.post("/search/files", json={"sort_by": "not_a_real_column"})
    assert resp.status_code == 422


def test_invalid_sort_order_returns_422(client):
    resp = client.post("/search/files", json={"sort_order": "sideways"})
    assert resp.status_code == 422


def test_sort_by_file_size_ascending(client, db, make_file):
    make_file("big.pdf", file_size=300)
    make_file("small.pdf", file_size=100)
    make_file("medium.pdf", file_size=200)

    resp = client.post("/search/files", json={"sort_by": "file_size", "sort_order": "asc"})

    assert resp.status_code == 200
    assert [f["filename"] for f in resp.json()] == ["small.pdf", "medium.pdf", "big.pdf"]


def test_sort_by_file_created_at_puts_nulls_last(client, db, make_file):
    now = datetime.now(timezone.utc)
    make_file("no_created_at.pdf", file_created_at=None)
    make_file("older.pdf", file_created_at=now - timedelta(days=2))
    make_file("newer.pdf", file_created_at=now - timedelta(days=1))

    resp = client.post("/search/files", json={"sort_by": "file_created_at", "sort_order": "asc"})

    assert resp.status_code == 200
    assert [f["filename"] for f in resp.json()] == ["older.pdf", "newer.pdf", "no_created_at.pdf"]
