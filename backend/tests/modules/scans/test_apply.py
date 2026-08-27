"""Tests for rescan.apply_diff (#67, ADR-0001b D6): turning an in-memory
RescanDiff into files/file_events/file_match_candidates/child ingest job
rows and paths.last_scanned_at, all in one caller-managed transaction.
"""
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.labeling.models import TypeLabel, TypeLabelFile
from app.modules.scans import discovery, rescan
from app.modules.scans.models import FileEvent, FileMatchCandidate


def _register(db, path: Path) -> RegisteredPath:
    row = RegisteredPath(path=str(path.resolve()))
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _make_file(db, *, path_id, full_path: Path, file_hash: str, file_size: int,
                file_modified_at: datetime, status: str = "ready") -> File:
    file = File(
        path_id=path_id, filename=full_path.name, full_path=str(full_path),
        file_type=full_path.suffix.lstrip("."), file_size=file_size, file_hash=file_hash,
        file_modified_at=file_modified_at, status=status,
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


def _rescan_job(db) -> Job:
    job = Job(type="scan", mode="rescan", trigger="manual")
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _events(db, scan_id: uuid.UUID) -> list[FileEvent]:
    return list(db.scalars(select(FileEvent).where(FileEvent.scan_id == scan_id)))


def _ingest_jobs(db, scan_id: uuid.UUID) -> list[Job]:
    return list(db.scalars(
        select(Job).where(Job.parent_job_id == scan_id, Job.type == "ingest")
    ))


def test_apply_diff_creates_file_event_and_ingest_job_for_added(db, tmp_path):
    registered_path = _register(db, tmp_path)
    (tmp_path / "new.pdf").write_bytes(b"brand-new-content")
    scan_job = _rescan_job(db)

    diff = rescan.diff_inventory(rescan.build_inventory([registered_path]), [])
    rescan.apply_diff(db, scan_job, diff, [registered_path])
    db.commit()

    [file] = list(db.scalars(select(File)))
    assert file.filename == "new.pdf"
    assert file.status == "discovered"
    assert file.embedding_status == "pending"

    [event] = _events(db, scan_job.id)
    assert event.event_type == "added"
    assert event.from_path is None
    assert event.to_hash == file.file_hash

    [ingest_job] = _ingest_jobs(db, scan_job.id)
    assert ingest_job.file_id == file.id
    assert ingest_job.trigger == "scan"
    assert ingest_job.rq_job_id is None


def test_apply_diff_updates_hash_resets_status_and_creates_ingest_job_for_modified(db, tmp_path):
    registered_path = _register(db, tmp_path)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"new-content-on-disk")
    existing = _make_file(
        db, path_id=registered_path.id, full_path=file_path,
        file_hash="stale-hash", file_size=1,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc), status="ready",
    )
    scan_job = _rescan_job(db)

    diff = rescan.diff_inventory(rescan.build_inventory([registered_path]), [existing])
    rescan.apply_diff(db, scan_job, diff, [registered_path])
    db.commit()

    db.refresh(existing)
    assert existing.file_hash == discovery.compute_sha256(file_path)
    assert existing.status == "discovered"
    assert existing.embedding_status == "pending"

    [event] = _events(db, scan_job.id)
    assert event.event_type == "modified"
    assert event.from_hash == "stale-hash"
    assert event.to_hash == existing.file_hash

    [ingest_job] = _ingest_jobs(db, scan_job.id)
    assert ingest_job.file_id == existing.id


def test_apply_diff_moved_updates_path_without_status_change_or_ingest_job(db, tmp_path):
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
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc), status="ready",
    )
    scan_job = _rescan_job(db)

    diff = rescan.diff_inventory(rescan.build_inventory([registered_path]), [existing])
    assert len(diff.moved) == 1
    rescan.apply_diff(db, scan_job, diff, [registered_path])
    db.commit()

    db.refresh(existing)
    assert existing.full_path == str(new_path)
    assert existing.status == "ready"  # untouched: content proven unchanged via hash

    [event] = _events(db, scan_job.id)
    assert event.event_type == "moved"
    assert event.from_path == str(original_path)
    assert event.to_path == str(new_path)
    assert not _ingest_jobs(db, scan_job.id)


def test_apply_diff_marks_missing_and_writes_event_once(db, tmp_path):
    registered_path = _register(db, tmp_path)
    existing = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "gone.pdf",
        file_hash="gone-hash", file_size=10,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc), status="ready",
    )
    scan_job = _rescan_job(db)

    diff = rescan.diff_inventory(rescan.build_inventory([registered_path]), [existing])
    rescan.apply_diff(db, scan_job, diff, [registered_path])
    db.commit()

    db.refresh(existing)
    assert existing.status == "missing"
    [event] = _events(db, scan_job.id)
    assert event.event_type == "missing"
    assert event.from_hash == "gone-hash"
    assert event.to_path is None


def test_apply_diff_skips_missing_event_when_already_missing(db, tmp_path):
    """A file that stays missing across rescans isn't a new semantic change
    (docs/workflow/01d-path-rescan.md file_events: 'Immutable record of one
    semantic manifest change') -- only the first transition into missing
    should write an event."""
    registered_path = _register(db, tmp_path)
    existing = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "gone.pdf",
        file_hash="gone-hash", file_size=10,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc), status="missing",
    )
    scan_job = _rescan_job(db)

    diff = rescan.diff_inventory(rescan.build_inventory([registered_path]), [existing])
    rescan.apply_diff(db, scan_job, diff, [registered_path])
    db.commit()

    assert not _events(db, scan_job.id)


@pytest.mark.parametrize("prior_status", ["missing"])
def test_apply_diff_recovers_previously_missing_file_reappearing_unchanged(db, tmp_path, prior_status):
    """Not an ADR-0001b-specified case: agreed with the user to extend D3's
    hash-backed trust model so a file that reappears at its old path with
    identical metadata (same size/mtime -> `unchanged` classification) is
    recognized instead of silently staying `missing` forever."""
    registered_path = _register(db, tmp_path)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"pdf-content")
    stat = file_path.stat()

    existing = _make_file(
        db, path_id=registered_path.id, full_path=file_path,
        file_hash=discovery.compute_sha256(file_path), file_size=stat.st_size,
        file_modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        status=prior_status,
    )
    scan_job = _rescan_job(db)

    diff = rescan.diff_inventory(rescan.build_inventory([registered_path]), [existing])
    assert [u.file.id for u in diff.unchanged] == [existing.id]
    rescan.apply_diff(db, scan_job, diff, [registered_path])
    db.commit()

    db.refresh(existing)
    assert existing.status == "discovered"
    assert existing.embedding_status == "pending"

    [event] = _events(db, scan_job.id)
    assert event.event_type == "recovered"
    assert event.from_hash == event.to_hash == existing.file_hash

    [ingest_job] = _ingest_jobs(db, scan_job.id)
    assert ingest_job.file_id == existing.id


def test_apply_diff_does_not_touch_unchanged_file_that_was_already_active(db, tmp_path):
    registered_path = _register(db, tmp_path)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"pdf-content")
    stat = file_path.stat()

    existing = _make_file(
        db, path_id=registered_path.id, full_path=file_path,
        file_hash=discovery.compute_sha256(file_path), file_size=stat.st_size,
        file_modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc), status="ready",
    )
    scan_job = _rescan_job(db)

    diff = rescan.diff_inventory(rescan.build_inventory([registered_path]), [existing])
    rescan.apply_diff(db, scan_job, diff, [registered_path])
    db.commit()

    db.refresh(existing)
    assert existing.status == "ready"
    assert not _events(db, scan_job.id)
    assert not _ingest_jobs(db, scan_job.id)


def test_apply_diff_backfills_filesystem_identity_for_unchanged_file(db, tmp_path):
    """#65's bootstrap note: identity matching only starts working once a
    live inventory's fs_device_id/fs_file_id get written back to `files` --
    apply must do this even for the no-op `unchanged` category."""
    registered_path = _register(db, tmp_path)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"pdf-content")
    stat = file_path.stat()

    existing = _make_file(
        db, path_id=registered_path.id, full_path=file_path,
        file_hash=discovery.compute_sha256(file_path), file_size=stat.st_size,
        file_modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc), status="ready",
    )
    assert existing.fs_device_id is None and existing.fs_file_id is None
    scan_job = _rescan_job(db)

    diff = rescan.diff_inventory(rescan.build_inventory([registered_path]), [existing])
    rescan.apply_diff(db, scan_job, diff, [registered_path])
    db.commit()

    db.refresh(existing)
    assert existing.fs_device_id is not None
    assert existing.fs_file_id is not None


def test_apply_diff_flags_labels_need_review_when_modified_file_has_labels(db, tmp_path):
    registered_path = _register(db, tmp_path)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"new-content")
    existing = _make_file(
        db, path_id=registered_path.id, full_path=file_path,
        file_hash="stale-hash", file_size=1,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc), status="ready",
    )
    type_label = TypeLabel(name="invoice")
    db.add(type_label)
    db.flush()
    db.add(TypeLabelFile(file_id=existing.id, type_label_id=type_label.id, source="user", status="confirmed"))
    db.commit()
    scan_job = _rescan_job(db)

    diff = rescan.diff_inventory(rescan.build_inventory([registered_path]), [existing])
    rescan.apply_diff(db, scan_job, diff, [registered_path])
    db.commit()

    db.refresh(existing)
    assert existing.labels_need_review is True


def test_apply_diff_creates_pending_candidate_without_a_file_row(db, tmp_path):
    from unittest.mock import patch

    from app.modules.processing.extraction import ExtractionResult
    from app.modules.scans.text_signature import compute_text_signature

    registered_path = _register(db, tmp_path)
    (tmp_path / "new_name.pdf").write_bytes(b"x" * 1000)
    missing = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "old_name.pdf",
        file_hash="old-hash", file_size=1000,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc), status="ready",
    )
    missing.text_signature = compute_text_signature(
        "Invoice Number 2024-001. This invoice covers consulting services rendered during January."
    )
    db.commit()
    scan_job = _rescan_job(db)

    with patch(
        "app.modules.scans.rescan.extraction.extract_text",
        return_value=ExtractionResult(
            text="Invoice Number 2024-001. This invoice covers consulting services rendered during January.",
            ocr_applied=False,
        ),
    ):
        diff = rescan.diff_inventory(rescan.build_inventory([registered_path]), [missing])
    assert len(diff.fuzzy_candidates) == 1

    rescan.apply_diff(db, scan_job, diff, [registered_path])
    db.commit()

    [candidate] = list(db.scalars(select(FileMatchCandidate).where(FileMatchCandidate.scan_id == scan_job.id)))
    assert candidate.status == "pending"
    assert candidate.missing_file_id == missing.id
    assert candidate.candidate_full_path == str(tmp_path / "new_name.pdf")

    db.refresh(missing)
    assert missing.status == "missing"
    assert db.scalar(select(File).where(File.full_path == str(tmp_path / "new_name.pdf"))) is None


def test_apply_diff_updates_last_scanned_at_for_every_registered_path(db, tmp_path):
    registered_path = _register(db, tmp_path)
    other_dir = tmp_path.parent / f"lfa_apply_other_{uuid.uuid4().hex}"
    other_dir.mkdir()
    other_path = _register(db, other_dir)
    scan_job = _rescan_job(db)

    diff = rescan.diff_inventory(rescan.build_inventory([registered_path, other_path]), [])
    rescan.apply_diff(db, scan_job, diff, [registered_path, other_path])
    db.commit()

    db.refresh(registered_path)
    db.refresh(other_path)
    assert registered_path.last_scanned_at is not None
    assert other_path.last_scanned_at is not None
    assert registered_path.last_scanned_at == other_path.last_scanned_at


def test_apply_diff_is_idempotent_for_file_events_on_retry(db, tmp_path):
    """ADR-0001b D6 / FileEvent's UniqueConstraint(scan_id, file_id,
    event_type): calling apply_diff twice for the same scan_id (a defensive
    retry against the same already-diffed set, e.g. after a crash right at
    the commit boundary) must not raise or double-write an event -- the
    second call's INSERT ... ON CONFLICT DO NOTHING silently no-ops.

    Uses a `modified` file rather than `added`: replaying the same diff
    twice against an `added` entry would try to INSERT a second File row
    with the same full_path and fail on that unique constraint first,
    which would be testing files.full_path's constraint, not file_events'
    idempotency (apply_diff's documented idempotent-retry guarantee is
    scoped to file_events/file_match_candidates, not to re-creating rows).
    """
    registered_path = _register(db, tmp_path)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"new-content-on-disk")
    existing = _make_file(
        db, path_id=registered_path.id, full_path=file_path,
        file_hash="stale-hash", file_size=1,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc), status="ready",
    )
    scan_job = _rescan_job(db)

    diff = rescan.diff_inventory(rescan.build_inventory([registered_path]), [existing])
    rescan.apply_diff(db, scan_job, diff, [registered_path])
    db.flush()
    rescan.apply_diff(db, scan_job, diff, [registered_path])
    db.commit()

    assert len(_events(db, scan_job.id)) == 1
