"""Tests for scan-time pruning of registered child subtrees (#39)."""
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.scans.service import run_scan


def _register(db, path: Path, parent: RegisteredPath | None = None) -> RegisteredPath:
    row = RegisteredPath(path=str(path.resolve()), parent_path_id=parent.id if parent else None)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _queue_scan(db, path_id) -> Job:
    job = Job(type="scan", path_id=path_id, trigger="scan", mode="initial")
    db.add(job)
    db.commit()
    return job


def test_run_scan_prunes_registered_child_subtree_entirely(db, tmp_path):
    parent_dir = tmp_path / "parent"
    child_dir = parent_dir / "child"
    child_dir.mkdir(parents=True)
    (parent_dir / "top.pdf").write_bytes(b"pdf-content")
    (child_dir / "nested.pdf").write_bytes(b"pdf-content")

    parent = _register(db, parent_dir)
    _register(db, child_dir, parent=parent)

    scan_job = _queue_scan(db, parent.id)
    result_job, ingest_jobs = run_scan(db, scan_job.id)

    assert result_job.status == "succeeded"
    filenames = {f.filename for f in db.scalars(select(File)).all()}
    assert filenames == {"top.pdf"}
    assert len(ingest_jobs) == 1


def test_run_scan_leaves_child_owned_file_path_id_untouched_and_fans_out_no_ingest(db, tmp_path):
    parent_dir = tmp_path / "parent"
    child_dir = parent_dir / "child"
    child_dir.mkdir(parents=True)
    doc_path = child_dir / "nested.pdf"
    doc_path.write_bytes(b"pdf-content")

    parent = _register(db, parent_dir)
    child = _register(db, child_dir, parent=parent)

    existing_file = File(
        path_id=child.id,
        filename="nested.pdf",
        full_path=str(doc_path.resolve()),
        file_type="pdf",
        file_size=doc_path.stat().st_size,
        file_hash="preexisting-hash",
        status="ready",
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    db.add(existing_file)
    db.commit()
    db.refresh(existing_file)

    scan_job = _queue_scan(db, parent.id)
    _result_job, ingest_jobs = run_scan(db, scan_job.id)

    db.refresh(existing_file)
    assert existing_file.path_id == child.id
    assert existing_file.file_hash == "preexisting-hash"
    assert existing_file.file_modified_at == datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert ingest_jobs == []


def test_run_scan_still_discovers_files_outside_excluded_subtree(db, tmp_path):
    parent_dir = tmp_path / "parent"
    child_dir = parent_dir / "child"
    sibling_dir = parent_dir / "sibling"
    child_dir.mkdir(parents=True)
    sibling_dir.mkdir()
    (sibling_dir / "sibling.pdf").write_bytes(b"pdf-content")

    parent = _register(db, parent_dir)
    _register(db, child_dir, parent=parent)

    scan_job = _queue_scan(db, parent.id)
    result_job, ingest_jobs = run_scan(db, scan_job.id)

    assert result_job.status == "succeeded"
    filenames = {f.filename for f in db.scalars(select(File)).all()}
    assert filenames == {"sibling.pdf"}
    assert len(ingest_jobs) == 1
