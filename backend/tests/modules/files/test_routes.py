"""Tests for POST /paths nested-path registration rules (#38) and the
advisory lock guarding concurrent path mutations (#62)."""
from pathlib import Path

from sqlalchemy import text

from app.modules.files import service
from app.modules.files.models import RegisteredPath
from app.shared.database import SessionLocal


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


def test_acquire_path_mutation_lock_is_mutually_exclusive_across_sessions():
    """#62: two concurrent path-mutation requests must never both pass the
    ancestor/duplicate check before either commits (the root cause of the
    `/a` + `/a/b` nesting race). Uses two independent sessions/connections
    from SessionLocal rather than the shared `client`/`db` fixtures --
    those two route every request through one connection wrapped in a
    single outer transaction (see conftest.py's `db` fixture), so they
    can't model two backends actually contending for the same advisory
    lock. `pg_try_advisory_xact_lock` (non-blocking) makes the assertion
    deterministic instead of racing real thread timing against a blocking
    call."""
    session_a = SessionLocal()
    session_b = SessionLocal()
    try:
        service.acquire_path_mutation_lock(session_a)  # holds the lock, uncommitted

        still_held = session_b.execute(
            text("SELECT pg_try_advisory_xact_lock(hashtext('lfa:paths:mutation'))")
        ).scalar()
        assert still_held is False

        session_a.rollback()  # releases it -- pg_advisory_xact_lock is tied to the transaction

        now_available = session_b.execute(
            text("SELECT pg_try_advisory_xact_lock(hashtext('lfa:paths:mutation'))")
        ).scalar()
        assert now_available is True
    finally:
        session_a.close()
        session_b.rollback()
        session_b.close()
