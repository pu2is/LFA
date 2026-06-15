from pathlib import Path

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
