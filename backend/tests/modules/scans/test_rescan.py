"""Tests for the Rescan inventory/diff/matching engine (WF1b, ADR-0001b D2/D3, #65)."""
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.modules.files.models import File, RegisteredPath
from app.modules.scans import discovery, rescan


def _register(db, path: Path, parent: RegisteredPath | None = None) -> RegisteredPath:
    row = RegisteredPath(path=str(path.resolve()), parent_path_id=parent.id if parent else None)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _make_file(
    db,
    *,
    path_id,
    full_path: Path,
    file_hash: str,
    file_size: int,
    file_modified_at: datetime,
    file_created_at: datetime | None = None,
    fs_device_id: str | None = None,
    fs_file_id: str | None = None,
) -> File:
    file = File(
        path_id=path_id,
        filename=full_path.name,
        full_path=str(full_path),
        file_type=full_path.suffix.lstrip("."),
        file_size=file_size,
        file_hash=file_hash,
        file_created_at=file_created_at,
        file_modified_at=file_modified_at,
        fs_device_id=fs_device_id,
        fs_file_id=fs_file_id,
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


def _inventory(registered_path: RegisteredPath) -> list[rescan.InventoryFile]:
    return rescan.build_inventory([registered_path])


def test_diff_inventory_classifies_unchanged_file(db, tmp_path):
    registered_path = _register(db, tmp_path)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"pdf-content")
    stat = file_path.stat()

    existing = _make_file(
        db, path_id=registered_path.id, full_path=file_path,
        file_hash=discovery.compute_sha256(file_path), file_size=stat.st_size,
        file_modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert [u.file.id for u in diff.unchanged] == [existing.id]
    assert not any([diff.metadata_refreshed, diff.modified, diff.moved, diff.moved_modified, diff.missing, diff.added])


def test_diff_inventory_classifies_metadata_refresh_when_hash_unchanged(db, tmp_path):
    registered_path = _register(db, tmp_path)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"pdf-content")

    existing = _make_file(
        db, path_id=registered_path.id, full_path=file_path,
        file_hash=discovery.compute_sha256(file_path), file_size=file_path.stat().st_size,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),  # stale mtime -> triggers cheap diff
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert [m.file.id for m in diff.metadata_refreshed] == [existing.id]
    assert not diff.unchanged and not diff.modified


def test_diff_inventory_classifies_modified_when_hash_changes(db, tmp_path):
    registered_path = _register(db, tmp_path)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"new-content-on-disk")

    existing = _make_file(
        db, path_id=registered_path.id, full_path=file_path,
        file_hash="stale-hash-from-before", file_size=1,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert len(diff.modified) == 1
    assert diff.modified[0].file.id == existing.id
    assert diff.modified[0].new_hash == discovery.compute_sha256(file_path)


def test_diff_inventory_classifies_moved_via_filesystem_identity(db, tmp_path):
    registered_path = _register(db, tmp_path)
    original_path = tmp_path / "old_name.pdf"
    original_path.write_bytes(b"pdf-content")
    file_hash = discovery.compute_sha256(original_path)
    old_size = original_path.stat().st_size

    new_path = tmp_path / "new_name.pdf"
    original_path.rename(new_path)  # simulates a move that already happened on disk

    [entry] = list(discovery.iter_inventory(tmp_path))

    existing = _make_file(
        db, path_id=registered_path.id, full_path=original_path,
        file_hash=file_hash, file_size=old_size,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        file_created_at=entry.file_created_at,
        fs_device_id=entry.fs_device_id, fs_file_id=entry.fs_file_id,
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert len(diff.moved) == 1
    assert diff.moved[0].file.id == existing.id
    assert diff.moved[0].match_method == "filesystem_id"
    assert diff.moved[0].inventory.full_path == str(new_path)
    assert not diff.missing and not diff.added and not diff.moved_modified


def test_diff_inventory_classifies_moved_modified_via_filesystem_identity(db, tmp_path):
    registered_path = _register(db, tmp_path)
    original_path = tmp_path / "old_name.pdf"
    original_path.write_bytes(b"original-content")
    old_hash = discovery.compute_sha256(original_path)
    old_size = original_path.stat().st_size

    new_path = tmp_path / "new_name.pdf"
    original_path.rename(new_path)
    new_path.write_bytes(b"changed-content-now-longer")  # same identity, new content

    [entry] = list(discovery.iter_inventory(tmp_path))

    existing = _make_file(
        db, path_id=registered_path.id, full_path=original_path,
        file_hash=old_hash, file_size=old_size,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        file_created_at=entry.file_created_at,
        fs_device_id=entry.fs_device_id, fs_file_id=entry.fs_file_id,
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert len(diff.moved_modified) == 1
    assert diff.moved_modified[0].file.id == existing.id
    assert diff.moved_modified[0].match_method == "filesystem_id"
    assert diff.moved_modified[0].new_hash == discovery.compute_sha256(new_path)
    assert not diff.moved


def test_diff_inventory_classifies_moved_via_hash_when_identity_unavailable(db, tmp_path):
    """A file never identity-tracked by an earlier scan (fs_device_id/
    fs_file_id/file_created_at all NULL) must still resolve via priority 2's
    exact-hash matching (ADR-0001b D3 step 4)."""
    registered_path = _register(db, tmp_path)
    original_path = tmp_path / "old_name.pdf"
    original_path.write_bytes(b"pdf-content")
    file_hash = discovery.compute_sha256(original_path)
    old_size = original_path.stat().st_size

    new_path = tmp_path / "new_name.pdf"
    original_path.rename(new_path)

    existing = _make_file(
        db, path_id=registered_path.id, full_path=original_path,
        file_hash=file_hash, file_size=old_size,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert len(diff.moved) == 1
    assert diff.moved[0].match_method == "hash"
    assert not diff.moved_modified


def test_diff_inventory_classifies_missing_and_added_independently(db, tmp_path):
    registered_path = _register(db, tmp_path)
    (tmp_path / "new_file.pdf").write_bytes(b"brand-new-content")

    existing = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "deleted.pdf",
        file_hash="deleted-file-hash", file_size=10,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert [m.file.id for m in diff.missing] == [existing.id]
    assert len(diff.added) == 1
    assert diff.added[0].inventory.filename == "new_file.pdf"
    assert diff.added[0].file_hash == discovery.compute_sha256(tmp_path / "new_file.pdf")
    assert not diff.moved and not diff.moved_modified


def test_diff_inventory_prioritizes_identity_match_before_attempting_hash_match(db, tmp_path):
    """Regression test: two missing/added pairs share the same content hash.
    If hash-matching ran before (or instead of) identity-matching, both pairs
    would look ambiguous (2 missing vs. 2 added for that hash) and neither
    would resolve. Identity-matching must claim its pair first, shrinking the
    hash-matching pool to 1-vs-1 so the remaining pair can still resolve."""
    registered_path = _register(db, tmp_path)

    old_a = tmp_path / "old_a.pdf"
    old_a.write_bytes(b"shared-content")
    shared_hash = discovery.compute_sha256(old_a)
    new_a = tmp_path / "new_a.pdf"
    old_a.rename(new_a)  # real rename -- identity is tracked and preserved

    new_b = tmp_path / "new_b.pdf"
    new_b.write_bytes(b"shared-content")  # identical content, unrelated file

    [entry_a] = [e for e in discovery.iter_inventory(tmp_path) if e.path.name == "new_a.pdf"]

    file_a = _make_file(
        db, path_id=registered_path.id, full_path=old_a,
        file_hash=shared_hash, file_size=entry_a.file_size,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        file_created_at=entry_a.file_created_at,
        fs_device_id=entry_a.fs_device_id, fs_file_id=entry_a.fs_file_id,
    )
    file_b = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "old_b.pdf",
        file_hash=shared_hash, file_size=len(b"shared-content"),
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        # never identity-tracked (e.g. predates #64) -- only hash can match it
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [file_a, file_b])

    assert {(m.file.id, m.match_method) for m in diff.moved} == {
        (file_a.id, "filesystem_id"),
        (file_b.id, "hash"),
    }
    assert not diff.missing and not diff.added


def test_diff_inventory_does_not_infer_moved_modified_when_creation_time_mismatches(db, tmp_path):
    """Regression test: same filesystem identity, but creation time disagrees
    and content changed -- ADR-0001b D3 forbids auto-claiming moved_modified
    from identity alone. With no exact-hash match available either, this must
    conservatively fall back to missing + added, not invent a pairing."""
    registered_path = _register(db, tmp_path)
    original_path = tmp_path / "old_name.pdf"
    original_path.write_bytes(b"original-content")
    old_hash = discovery.compute_sha256(original_path)
    old_size = original_path.stat().st_size

    new_path = tmp_path / "new_name.pdf"
    original_path.rename(new_path)
    new_path.write_bytes(b"different-content-now")  # same identity, changed content

    [entry] = list(discovery.iter_inventory(tmp_path))
    assert entry.file_created_at is not None  # sanity: this platform tracks creation time

    mismatched_created_at = entry.file_created_at.replace(year=entry.file_created_at.year - 1)

    existing = _make_file(
        db, path_id=registered_path.id, full_path=original_path,
        file_hash=old_hash, file_size=old_size,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        file_created_at=mismatched_created_at,
        fs_device_id=entry.fs_device_id, fs_file_id=entry.fs_file_id,
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert not diff.moved_modified and not diff.moved
    assert [m.file.id for m in diff.missing] == [existing.id]
    assert len(diff.added) == 1


def test_build_inventory_prunes_registered_child_subtree(db, tmp_path):
    parent_dir = tmp_path / "parent"
    child_dir = parent_dir / "child"
    child_dir.mkdir(parents=True)
    (parent_dir / "top.pdf").write_bytes(b"pdf-content")
    (child_dir / "nested.pdf").write_bytes(b"pdf-content")

    parent = _register(db, parent_dir)
    child = _register(db, child_dir, parent=parent)

    inventory = rescan.build_inventory([parent, child])

    by_path_id: dict = {}
    for entry in inventory:
        by_path_id.setdefault(entry.path_id, []).append(entry.filename)

    assert by_path_id[parent.id] == ["top.pdf"]
    assert by_path_id[child.id] == ["nested.pdf"]


def test_build_inventory_raises_on_any_unreadable_root(db, tmp_path):
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    (good_dir / "a.pdf").write_bytes(b"pdf-content")
    bad_dir = tmp_path / "bad"  # never created on disk

    good_path = _register(db, good_dir)
    bad_path = _register(db, bad_dir)

    with pytest.raises(OSError):
        rescan.build_inventory([good_path, bad_path])
