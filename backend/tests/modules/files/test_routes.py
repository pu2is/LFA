"""Tests for POST /paths nested-path registration rules (#38)."""
from pathlib import Path

from app.modules.files.models import RegisteredPath


def _register(client, path: Path):
    return client.post("/paths", json={"path": str(path)})


def test_register_descendant_returns_409_naming_conflicting_path(client, tmp_path):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)

    resp_parent = _register(client, parent)
    assert resp_parent.status_code == 201

    resp_child = _register(client, child)

    assert resp_child.status_code == 409
    assert str(parent.resolve()) in resp_child.json()["detail"]


def test_register_ancestor_adopts_orphan_descendants(client, db, tmp_path):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)

    resp_child = _register(client, child)
    assert resp_child.status_code == 201
    child_id = resp_child.json()["id"]

    resp_parent = _register(client, parent)
    assert resp_parent.status_code == 201
    parent_id = resp_parent.json()["id"]

    db.expire_all()
    child_row = db.get(RegisteredPath, child_id)
    assert str(child_row.parent_path_id) == parent_id


def test_already_parented_descendant_is_untouched_by_further_ancestors(client, db, tmp_path):
    grandparent = tmp_path / "grandparent"
    parent = grandparent / "parent"
    child = parent / "child"
    child.mkdir(parents=True)

    resp_child = _register(client, child)
    child_id = resp_child.json()["id"]

    resp_parent = _register(client, parent)
    assert resp_parent.status_code == 201
    parent_id = resp_parent.json()["id"]

    db.expire_all()
    child_row = db.get(RegisteredPath, child_id)
    assert str(child_row.parent_path_id) == parent_id

    resp_grandparent = _register(client, grandparent)
    assert resp_grandparent.status_code == 201
    grandparent_id = resp_grandparent.json()["id"]

    db.expire_all()
    parent_row = db.get(RegisteredPath, parent_id)
    child_row = db.get(RegisteredPath, child_id)
    assert str(parent_row.parent_path_id) == grandparent_id
    assert str(child_row.parent_path_id) == parent_id


def test_exact_duplicate_still_409(client, tmp_path):
    target = tmp_path / "target"
    target.mkdir()

    assert _register(client, target).status_code == 201
    resp = _register(client, target)

    assert resp.status_code == 409
    assert resp.json()["detail"] == "Path is already registered"


def test_unrelated_sibling_paths_unaffected(client, db, tmp_path):
    sibling_a = tmp_path / "sibling_a"
    sibling_b = tmp_path / "sibling_b"
    sibling_a.mkdir()
    sibling_b.mkdir()

    resp_a = _register(client, sibling_a)
    assert resp_a.status_code == 201
    resp_b = _register(client, sibling_b)
    assert resp_b.status_code == 201

    db.expire_all()
    row_a = db.get(RegisteredPath, resp_a.json()["id"])
    row_b = db.get(RegisteredPath, resp_b.json()["id"])
    assert row_a.parent_path_id is None
    assert row_b.parent_path_id is None
