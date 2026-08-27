"""Tests for files.service.list_files()'s per-file latest-job join (#36) and
upsert_file's ON CONFLICT upsert (#63)."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.files.service import list_files, upsert_file
from app.modules.jobs.models import Job

_FAKE_PATH = "D:/lfa_test_files_service"
_BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_path(db, suffix: str = ""):
    path = RegisteredPath(path=_FAKE_PATH + suffix)
    db.add(path)
    db.flush()
    return path


def _make_file(db, path, filename: str = "a.pdf"):
    file = File(
        path_id=path.id,
        filename=filename,
        full_path=f"{path.path}/{filename}",
        file_type="pdf",
        file_size=100,
        file_hash=f"hash-{filename}",
        file_modified_at=_BASE_TIME,
    )
    db.add(file)
    db.flush()
    return file


def _make_job(
    db,
    file,
    *,
    type: str,
    status: str,
    created_at: datetime,
    error_message: str | None = None,
    mode: str = "default",
    trigger: str = "scan",
):
    job = Job(
        type=type,
        file_id=file.id,
        trigger=trigger,
        mode=mode,
        status=status,
        error_message=error_message,
        created_at=created_at,
    )
    db.add(job)
    db.flush()
    return job


def _row_for(rows, file_id):
    for file, job_status, error_msg, job_type in rows:
        if file.id == file_id:
            return job_status, error_msg, job_type
    raise AssertionError(f"file {file_id} not in result rows")


def test_list_files_no_jobs_returns_none_fields(db):
    path = _make_path(db)
    file = _make_file(db, path)
    db.commit()

    rows = list_files(db, path.id)

    assert _row_for(rows, file.id) == (None, None, None)


def test_list_files_shows_latest_ingest_status(db):
    path = _make_path(db)
    file = _make_file(db, path)
    _make_job(db, file, type="ingest", status="succeeded", created_at=_BASE_TIME)
    db.commit()

    rows = list_files(db, path.id)

    assert _row_for(rows, file.id) == ("succeeded", None, "ingest")


def test_list_files_prefers_more_recent_label_failure_over_older_ingest_success(db):
    """Core #36 behavior: a later label-job failure must win over an earlier
    successful ingest, so label failures are no longer invisible in the list.
    """
    path = _make_path(db)
    file = _make_file(db, path)
    _make_job(db, file, type="ingest", status="succeeded", created_at=_BASE_TIME)
    _make_job(
        db, file,
        type="label",
        status="failed",
        mode="initial",
        trigger="manual",
        error_message="LLM parse error",
        created_at=_BASE_TIME + timedelta(minutes=5),
    )
    db.commit()

    rows = list_files(db, path.id)

    assert _row_for(rows, file.id) == ("failed", "LLM parse error", "label")


def test_list_files_ignores_embed_jobs(db):
    """embed has its own files.embedding_status column; must not leak into
    processing_job_status/processing_error_message even if more recent.
    """
    path = _make_path(db)
    file = _make_file(db, path)
    _make_job(db, file, type="ingest", status="succeeded", created_at=_BASE_TIME)
    _make_job(
        db, file,
        type="embed",
        status="failed",
        error_message="Ollama unreachable",
        created_at=_BASE_TIME + timedelta(minutes=5),
    )
    db.commit()

    rows = list_files(db, path.id)

    assert _row_for(rows, file.id) == ("succeeded", None, "ingest")


def test_list_files_filters_by_path_id(db):
    path_a = _make_path(db, "_a")
    path_b = _make_path(db, "_b")
    file_a = _make_file(db, path_a, filename="a.pdf")
    file_b = _make_file(db, path_b, filename="b.pdf")
    db.commit()

    rows = list_files(db, path_a.id)

    file_ids = {file.id for file, *_ in rows}
    assert file_a.id in file_ids
    assert file_b.id not in file_ids


# --------------------------------------------------------------------------- #
# upsert_file's ON CONFLICT upsert (#63)
# --------------------------------------------------------------------------- #

def test_upsert_file_inserts_new_row(db):
    path = _make_path(db, "_insert")
    db.commit()
    full_path = f"{path.path}/new.pdf"

    file = upsert_file(
        db,
        path_id=path.id,
        full_path=full_path,
        filename="new.pdf",
        file_type="pdf",
        file_size=100,
        file_hash="hash-new",
        file_created_at=None,
        file_modified_at=_BASE_TIME,
    )
    db.commit()

    assert file.full_path == full_path
    assert file.status == "discovered"
    assert file.ocr_applied is False


def test_upsert_file_refreshes_metadata_but_preserves_pipeline_fields(db):
    """Scope-out: refreshing filesystem metadata on an existing row must not
    reset `status` or `ocr_applied` -- those reflect later pipeline stages
    (OCR, labeling) that a scan must not silently undo even if the file's
    content changed (03_er-diagram.md "modified" handling)."""
    path = _make_path(db, "_refresh")
    db.commit()
    full_path = f"{path.path}/existing.pdf"

    file = upsert_file(
        db,
        path_id=path.id,
        full_path=full_path,
        filename="existing.pdf",
        file_type="pdf",
        file_size=100,
        file_hash="hash-v1",
        file_created_at=None,
        file_modified_at=_BASE_TIME,
    )
    file.status = "ready"
    file.ocr_applied = True
    db.commit()

    later = _BASE_TIME + timedelta(hours=1)
    updated = upsert_file(
        db,
        path_id=path.id,
        full_path=full_path,
        filename="existing.pdf",
        file_type="pdf",
        file_size=200,
        file_hash="hash-v2",
        file_created_at=None,
        file_modified_at=later,
    )
    db.commit()

    assert updated.id == file.id
    assert updated.file_size == 200
    assert updated.file_hash == "hash-v2"
    assert updated.file_modified_at == later
    assert updated.status == "ready"
    assert updated.ocr_applied is True


def test_upsert_file_survives_a_racing_duplicate_insert(db):
    """#63: two concurrent upserts for the same full_path (e.g. overlapping
    rescans of overlapping registered paths) used to both SELECT and find
    nothing, then race to INSERT -- the loser's flush/commit crashed on the
    files.full_path unique constraint (uncaught outside OSError in run_scan,
    leaving the scan job stuck at status=running forever, see #63). INSERT
    ... ON CONFLICT DO UPDATE makes this a structural non-issue.

    Simulated here the same way #53/#54 simulate a racing writer: two calls
    against a session that never re-reads its own not-yet-flushed insert.
    db.no_autoflush is what makes the second call also see "not found" --
    it matches production's actual SessionLocal(autoflush=False) (see
    app/shared/database.py), which is exactly the condition that let the old
    SELECT-then-insert code race with itself in the first place.
    """
    path = _make_path(db, "_race")
    db.commit()
    full_path = f"{path.path}/racing.pdf"

    with db.no_autoflush:
        upsert_file(
            db,
            path_id=path.id,
            full_path=full_path,
            filename="racing.pdf",
            file_type="pdf",
            file_size=100,
            file_hash="hash-a",
            file_created_at=None,
            file_modified_at=_BASE_TIME,
        )
        second = upsert_file(  # simulated racing writer
            db,
            path_id=path.id,
            full_path=full_path,
            filename="racing.pdf",
            file_type="pdf",
            file_size=200,
            file_hash="hash-b",
            file_created_at=None,
            file_modified_at=_BASE_TIME,
        )

    db.commit()  # must not raise IntegrityError

    rows = list(db.scalars(select(File).where(File.full_path == full_path)))
    assert len(rows) == 1
    assert rows[0].id == second.id
    assert rows[0].file_size == 200
    assert rows[0].file_hash == "hash-b"
