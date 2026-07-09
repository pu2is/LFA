"""Tests for files.service.list_files()'s per-file latest-job join (#36)."""
from datetime import datetime, timedelta, timezone

from app.modules.files.models import File, RegisteredPath
from app.modules.files.service import list_files
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
