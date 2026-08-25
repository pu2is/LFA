"""Tests for POST /search/files (#57, ADR-0002a D3, WF2a manual filter final
query: type/tag intersection, case-insensitive tag matching, sort + NULLS LAST)."""
import base64
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.modules.files.models import File, RegisteredPath
from app.modules.files.schemas import FileRead
from app.modules.labeling.models import TagKind, TagLabel, TypeLabel, TypeLabelFile
from app.modules.search.routes import _encode_cursor


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
    assert {f["filename"] for f in resp.json()["items"]} == {"a.pdf", "b.pdf"}


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
    assert [f["filename"] for f in resp.json()["items"]] == ["invoice.pdf"]


def test_excludes_non_confirmed_type_labels_file(client, db, make_file, type_label):
    file = make_file()
    _add_type(db, file, type_label, status="suggested", source="llm")

    resp = client.post("/search/files", json={"type_label_ids": [str(type_label.id)]})

    assert resp.status_code == 200
    assert resp.json()["items"] == []


def test_filters_by_tag_case_insensitive(client, db, make_file, tag_kind):
    berlin_file = make_file("berlin.pdf")
    _add_tag(db, berlin_file, tag_kind, "berlin")
    munich_file = make_file("munich.pdf")
    _add_tag(db, munich_file, tag_kind, "Munich")

    resp = client.post("/search/files", json={"tags": [{"kind_id": str(tag_kind.id), "value": "Berlin"}]})

    assert resp.status_code == 200
    assert [f["filename"] for f in resp.json()["items"]] == ["berlin.pdf"]


def test_excludes_non_confirmed_tags(client, db, make_file, tag_kind):
    file = make_file()
    _add_tag(db, file, tag_kind, "Berlin", status="rejected", source="llm")

    resp = client.post("/search/files", json={"tags": [{"kind_id": str(tag_kind.id), "value": "Berlin"}]})

    assert resp.status_code == 200
    assert resp.json()["items"] == []


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
    assert [f["filename"] for f in resp.json()["items"]] == ["both.pdf"]


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
    assert {f["filename"] for f in resp.json()["items"]} == {"berlin.pdf", "munich.pdf"}


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
    assert [f["filename"] for f in resp.json()["items"]] == ["both_types.pdf"]


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
    assert resp.json()["items"] == []


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
    assert [f["filename"] for f in resp.json()["items"]] == ["small.pdf", "medium.pdf", "big.pdf"]


def test_sort_by_file_created_at_puts_nulls_last(client, db, make_file):
    now = datetime.now(timezone.utc)
    make_file("no_created_at.pdf", file_created_at=None)
    make_file("older.pdf", file_created_at=now - timedelta(days=2))
    make_file("newer.pdf", file_created_at=now - timedelta(days=1))

    resp = client.post("/search/files", json={"sort_by": "file_created_at", "sort_order": "asc"})

    assert resp.status_code == 200
    assert [f["filename"] for f in resp.json()["items"]] == ["older.pdf", "newer.pdf", "no_created_at.pdf"]


def test_cursor_pagination_crosses_the_nulls_last_boundary(client, db, make_file):
    """_keyset_condition's trickiest branch: a cursor sitting on the last
    non-null row must still admit every null row next (NULLS LAST puts them
    after ALL non-null values, not just later ones), and once inside the
    null group itself, the id tiebreak must still terminate pagination."""
    now = datetime.now(timezone.utc)
    make_file("older.pdf", file_created_at=now - timedelta(days=2))
    make_file("newer.pdf", file_created_at=now - timedelta(days=1))
    make_file("null_a.pdf", file_created_at=None)
    make_file("null_b.pdf", file_created_at=None)

    all_names: list[str] = []
    cursor = None
    for _ in range(5):
        resp = client.post(
            "/search/files",
            json={
                "sort_by": "file_created_at",
                "sort_order": "asc",
                "limit": 1,
                **({"cursor": cursor} if cursor else {}),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        all_names.extend(f["filename"] for f in body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break

    assert all_names[:2] == ["older.pdf", "newer.pdf"]
    assert set(all_names[2:]) == {"null_a.pdf", "null_b.pdf"}
    assert len(all_names) == 4
    assert cursor is None


def test_cursor_pagination_crosses_the_nulls_last_boundary_descending(client, db, make_file):
    """Same boundary as test_cursor_pagination_crosses_the_nulls_last_boundary,
    but sort_order=desc: _keyset_condition's `progressed = column > cursor_value
    if ascending else column < cursor_value` branch flips direction, and NULLS
    LAST still has to hold (nulls stay last in both directions), so this needs
    its own coverage independent of the asc case."""
    now = datetime.now(timezone.utc)
    make_file("older.pdf", file_created_at=now - timedelta(days=2))
    make_file("newer.pdf", file_created_at=now - timedelta(days=1))
    make_file("null_a.pdf", file_created_at=None)
    make_file("null_b.pdf", file_created_at=None)

    all_names: list[str] = []
    cursor = None
    for _ in range(5):
        resp = client.post(
            "/search/files",
            json={
                "sort_by": "file_created_at",
                "sort_order": "desc",
                "limit": 1,
                **({"cursor": cursor} if cursor else {}),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        all_names.extend(f["filename"] for f in body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break

    assert all_names[:2] == ["newer.pdf", "older.pdf"]
    assert set(all_names[2:]) == {"null_a.pdf", "null_b.pdf"}
    assert len(all_names) == 4
    assert cursor is None


# --------------------------------------------------------------------------- #
# Pagination (#58, cursor/keyset -- not offset)
# --------------------------------------------------------------------------- #

def test_default_page_size_applies_when_omitted(client, db, make_file):
    for i in range(5):
        make_file(f"file_{i}.pdf")

    resp = client.post("/search/files", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert body["limit"] == 50
    assert body["next_cursor"] is None
    assert body["total"] == 5
    assert len(body["items"]) == 5


def test_successive_pages_are_disjoint_and_cover_everything(client, db, make_file):
    for i in range(10):
        make_file(f"file_{i:02d}.pdf", file_size=i)

    all_names: list[str] = []
    cursor = None
    for _ in range(4):  # 10 files / limit 4 -> 3 pages, plus one extra call to prove it stays empty
        resp = client.post(
            "/search/files",
            json={"sort_by": "file_size", "sort_order": "asc", "limit": 4, **({"cursor": cursor} if cursor else {})},
        )
        assert resp.status_code == 200
        body = resp.json()
        all_names.extend(f["filename"] for f in body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break

    assert all_names == [f"file_{i:02d}.pdf" for i in range(10)]  # no gaps, no repeats, in order
    assert cursor is None  # pagination actually terminated


def test_total_reflects_full_filtered_set_regardless_of_page(client, db, make_file):
    for i in range(7):
        make_file(f"file_{i}.pdf")

    first = client.post("/search/files", json={"limit": 2})
    assert first.json()["total"] == 7

    second = client.post("/search/files", json={"limit": 2, "cursor": first.json()["next_cursor"]})
    assert second.json()["total"] == 7
    assert len(second.json()["items"]) == 2


def test_cursor_past_the_end_returns_empty_page_not_error(client, db, make_file):
    make_file("a.pdf", file_size=1)
    make_file("b.pdf", file_size=2)
    make_file("c.pdf", file_size=3)

    full_page = client.post("/search/files", json={"sort_by": "file_size", "sort_order": "asc", "limit": 10})
    last_item = full_page.json()["items"][-1]
    stale_cursor = _encode_cursor(FileRead(**last_item), "file_size", "asc")

    resp = client.post(
        "/search/files",
        json={"sort_by": "file_size", "sort_order": "asc", "limit": 10, "cursor": stale_cursor},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["next_cursor"] is None
    assert body["total"] == 3


def _raw_cursor(payload: dict) -> str:
    """Builds a cursor from an arbitrary dict, bypassing _encode_cursor --
    for regression-testing _decode_cursor's validation of malformed payloads
    that a real client couldn't produce but a crafted request could."""
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def test_invalid_cursor_returns_422(client):
    resp = client.post("/search/files", json={"cursor": "not-a-real-cursor"})
    assert resp.status_code == 422


def test_cursor_with_non_string_id_returns_422(client):
    """Regression for #58: uuid.UUID() raises AttributeError (not caught by
    _decode_cursor's except tuple) on a non-string id, which crashed with an
    unhandled 500 instead of the documented 422."""
    cursor = _raw_cursor({"sort_by": "filename", "sort_order": "asc", "v": "a.pdf", "id": 12345})
    resp = client.post("/search/files", json={"sort_by": "filename", "sort_order": "asc", "cursor": cursor})
    assert resp.status_code == 422


def test_cursor_value_type_mismatch_returns_422(client):
    """Regression for #58: an unvalidated 'v' for a non-datetime column (here
    file_size, an int column) used to reach the DB unchecked and crash with a
    raw Postgres error instead of a clean 422."""
    cursor = _raw_cursor(
        {"sort_by": "file_size", "sort_order": "asc", "v": "not-a-number", "id": str(uuid.uuid4())}
    )
    resp = client.post("/search/files", json={"sort_by": "file_size", "sort_order": "asc", "cursor": cursor})
    assert resp.status_code == 422


def test_cursor_for_different_sort_returns_422(client, db, make_file):
    make_file("a.pdf")

    resp = client.post("/search/files", json={"sort_by": "filename", "sort_order": "asc", "limit": 1})
    cursor = resp.json()["next_cursor"] or _encode_cursor(
        FileRead(**resp.json()["items"][0]), "filename", "asc"
    )

    mismatched = client.post("/search/files", json={"sort_by": "file_size", "sort_order": "asc", "cursor": cursor})
    assert mismatched.status_code == 422


def test_limit_zero_returns_422(client):
    resp = client.post("/search/files", json={"limit": 0})
    assert resp.status_code == 422


def test_limit_above_max_returns_422(client):
    resp = client.post("/search/files", json={"limit": 201})
    assert resp.status_code == 422


def test_limit_at_max_is_accepted(client, db, make_file):
    make_file("a.pdf")

    resp = client.post("/search/files", json={"limit": 200})

    assert resp.status_code == 200
    assert resp.json()["limit"] == 200
