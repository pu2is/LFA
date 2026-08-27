from pathlib import Path

import pytest

from app.modules.scans import discovery


def test_iter_documents_filters_unsupported_lock_and_hidden_files(tmp_path: Path):
    (tmp_path / "report.pdf").write_bytes(b"pdf-content")
    (tmp_path / "contract.docx").write_bytes(b"docx-content")
    (tmp_path / "notes.txt").write_bytes(b"not a tracked document type")
    (tmp_path / "~$contract.docx").write_bytes(b"word lock file")
    (tmp_path / ".hidden.pdf").write_bytes(b"dotfile")

    found = {doc.path.name for doc in discovery.iter_documents(tmp_path)}

    assert found == {"report.pdf", "contract.docx"}


def test_iter_documents_recurses_into_subdirectories(tmp_path: Path):
    sub = tmp_path / "subfolder"
    sub.mkdir()
    (sub / "nested.pdf").write_bytes(b"pdf-content")

    found = {doc.path.name for doc in discovery.iter_documents(tmp_path)}

    assert found == {"nested.pdf"}


def test_describe_populates_metadata(tmp_path: Path):
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"pdf-content")

    [doc] = list(discovery.iter_documents(tmp_path))

    assert doc.file_type == "pdf"
    assert doc.file_size == len(b"pdf-content")
    assert doc.file_hash == discovery.compute_sha256(file_path)
    assert doc.file_modified_at is not None


def test_iter_documents_prunes_excluded_subtree_without_hashing(tmp_path: Path, monkeypatch):
    excluded_dir = tmp_path / "registered_child"
    excluded_dir.mkdir()
    (excluded_dir / "excluded.pdf").write_bytes(b"pdf-content")
    (tmp_path / "included.pdf").write_bytes(b"pdf-content")

    hashed_paths: list[Path] = []
    original_compute_sha256 = discovery.compute_sha256
    monkeypatch.setattr(
        discovery,
        "compute_sha256",
        lambda path: hashed_paths.append(path) or original_compute_sha256(path),
    )

    found = {
        doc.path.name
        for doc in discovery.iter_documents(tmp_path, exclude_roots=frozenset({excluded_dir}))
    }

    assert found == {"included.pdf"}
    assert excluded_dir / "excluded.pdf" not in hashed_paths


def test_iter_documents_with_empty_exclude_roots_matches_default_behavior(tmp_path: Path):
    (tmp_path / "report.pdf").write_bytes(b"pdf-content")
    sub = tmp_path / "subfolder"
    sub.mkdir()
    (sub / "nested.pdf").write_bytes(b"pdf-content")

    found = {
        doc.path.name for doc in discovery.iter_documents(tmp_path, exclude_roots=frozenset())
    }

    assert found == {"report.pdf", "nested.pdf"}


def test_iter_inventory_does_not_hash(tmp_path: Path, monkeypatch):
    (tmp_path / "report.pdf").write_bytes(b"pdf-content")

    hashed_paths: list[Path] = []
    monkeypatch.setattr(
        discovery, "compute_sha256", lambda path: hashed_paths.append(path) or "unused"
    )

    [entry] = list(discovery.iter_inventory(tmp_path))

    assert entry.path.name == "report.pdf"
    assert hashed_paths == []


def test_iter_inventory_populates_cheap_metadata(tmp_path: Path):
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"pdf-content")

    [entry] = list(discovery.iter_inventory(tmp_path))

    assert entry.file_type == "pdf"
    assert entry.file_size == len(b"pdf-content")
    assert entry.file_modified_at is not None
    assert entry.fs_device_id != ""
    assert entry.fs_file_id != ""


def test_iter_inventory_prunes_excluded_subtree(tmp_path: Path):
    excluded_dir = tmp_path / "registered_child"
    excluded_dir.mkdir()
    (excluded_dir / "excluded.pdf").write_bytes(b"pdf-content")
    (tmp_path / "included.pdf").write_bytes(b"pdf-content")

    found = {
        entry.path.name
        for entry in discovery.iter_inventory(tmp_path, exclude_roots=frozenset({excluded_dir}))
    }

    assert found == {"included.pdf"}


def test_iter_inventory_raises_on_missing_root(tmp_path: Path):
    missing_root = tmp_path / "does-not-exist"

    with pytest.raises(OSError):
        list(discovery.iter_inventory(missing_root))


def test_hash_with_retry_returns_hash_for_stable_file(tmp_path: Path):
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"pdf-content")

    assert discovery.hash_with_retry(file_path) == discovery.compute_sha256(file_path)


def test_hash_with_retry_raises_after_one_retry_when_file_keeps_changing(tmp_path: Path, monkeypatch):
    file_path = tmp_path / "unstable.pdf"
    file_path.write_bytes(b"pdf-content")

    class _FakeStat:
        def __init__(self, size: int):
            self.st_size = size
            self.st_mtime = float(size)

    # A new, different size on every stat() call -- never stabilizes across
    # the single retry ADR-0001b D2 allows (2 attempts x 2 stats each).
    sizes = iter(range(4))
    monkeypatch.setattr(Path, "stat", lambda self: _FakeStat(next(sizes)))

    with pytest.raises(discovery.UnstableFileError):
        discovery.hash_with_retry(file_path)
