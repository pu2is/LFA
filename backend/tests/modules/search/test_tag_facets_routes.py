"""Tests for GET /search/tag-facets (#56, ADR-0002a D3, WF2a manual filter
candidate-set facet query)."""
import uuid
from datetime import datetime, timezone

import pytest

from app.modules.files.models import File, RegisteredPath
from app.modules.labeling.models import TagKind, TagLabel, TypeLabel, TypeLabelFile


@pytest.fixture
def make_file(db):
    def _make(name: str = "sample.pdf"):
        path = RegisteredPath(path=f"/tmp/lfa_search_test_{uuid.uuid4().hex}")
        db.add(path)
        db.flush()

        file = File(
            path_id=path.id,
            filename=name,
            full_path=f"{path.path}/{name}",
            file_type="pdf",
            file_size=100,
            file_hash=f"hash-{uuid.uuid4().hex}",
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
    db.refresh(row)
    return row


def _add_type(db, file, type_label_row, *, status="confirmed", source="user"):
    row = TypeLabelFile(file_id=file.id, type_label_id=type_label_row.id, source=source, status=status)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def test_empty_catalog_returns_empty_list(client):
    resp = client.get("/search/tag-facets")
    assert resp.status_code == 200
    assert resp.json() == []


def test_no_filter_returns_all_confirmed_tags(client, db, make_file, tag_kind):
    file = make_file()
    _add_tag(db, file, tag_kind, "Berlin")

    resp = client.get("/search/tag-facets")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["kind_id"] == str(tag_kind.id)
    assert data[0]["kind_name"] == "city"
    assert data[0]["values"] == ["Berlin"]


def test_excludes_non_confirmed_tags(client, db, make_file, tag_kind):
    file = make_file()
    _add_tag(db, file, tag_kind, "Berlin", status="suggested", source="llm")
    _add_tag(db, file, tag_kind, "Munich", status="rejected", source="llm")

    resp = client.get("/search/tag-facets")

    assert resp.status_code == 200
    assert resp.json() == []


def test_empty_kinds_are_excluded(client, db, make_file):
    """A kind that exists in the catalog but has no confirmed tag_labels in
    the candidate set must not appear in the response."""
    file = make_file()
    unused_kind = TagKind(name="unused")
    db.add(unused_kind)
    db.commit()

    resp = client.get("/search/tag-facets")

    assert resp.status_code == 200
    assert resp.json() == []


def test_values_are_alphabetically_ordered(client, db, make_file, tag_kind):
    file = make_file()
    for value in ["Zurich", "Amsterdam", "Munich"]:
        _add_tag(db, file, tag_kind, value)

    resp = client.get("/search/tag-facets")

    assert resp.status_code == 200
    assert resp.json()[0]["values"] == ["Amsterdam", "Munich", "Zurich"]


def test_case_insensitive_dedup_deterministic_representative(client, db, make_file, tag_kind):
    """#56 AC: "Berlin" and "berlin" collapse into one value. The
    representative is deterministic (lexicographically smallest raw value),
    not "whichever row was inserted/visited first"."""
    file_a = make_file("a.pdf")
    file_b = make_file("b.pdf")
    _add_tag(db, file_a, tag_kind, "berlin")
    _add_tag(db, file_b, tag_kind, "Berlin")

    resp = client.get("/search/tag-facets")

    assert resp.status_code == 200
    values = resp.json()[0]["values"]
    assert values == ["Berlin"]  # "B" (0x42) sorts before "b" (0x62)


def test_filters_by_type(client, db, make_file, tag_kind, type_label):
    other_type = TypeLabel(name="contract")
    db.add(other_type)
    db.commit()
    db.refresh(other_type)

    invoice_file = make_file("invoice.pdf")
    _add_type(db, invoice_file, type_label)
    _add_tag(db, invoice_file, tag_kind, "Berlin")

    contract_file = make_file("contract.pdf")
    _add_type(db, contract_file, other_type)
    _add_tag(db, contract_file, tag_kind, "Munich")

    resp = client.get("/search/tag-facets", params={"type_label_ids": [str(type_label.id)]})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["values"] == ["Berlin"]


def test_type_filter_ignores_non_confirmed_type_labels_file(client, db, make_file, tag_kind, type_label):
    file = make_file()
    _add_type(db, file, type_label, status="suggested", source="llm")
    _add_tag(db, file, tag_kind, "Berlin")

    resp = client.get("/search/tag-facets", params={"type_label_ids": [str(type_label.id)]})

    assert resp.status_code == 200
    assert resp.json() == []


def test_unknown_type_label_id_returns_404(client):
    resp = client.get("/search/tag-facets", params={"type_label_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404
