import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.main import app
from app.modules.files.models import File, RegisteredPath
from app.modules.processing.models import ProcessingJob
from app.shared.database import SessionLocal

client = TestClient(app)

# Fake path string — doesn't need to exist on disk because we insert directly into DB.
_FAKE_PATH = "D:/lfa_test_fake_route_path"


@pytest.fixture
def db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def path_and_discovered_file(db: Session):
    """Insert a RegisteredPath + one discovered File; clean up after the test."""
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="invoice.pdf",
        full_path=f"{_FAKE_PATH}/invoice.pdf",
        file_type="pdf",
        file_size=2048,
        file_hash="deadbeef" * 8,
        file_modified_at=datetime.now(timezone.utc),
        status="discovered",
    )
    db.add(file)
    db.commit()
    db.refresh(path)
    db.refresh(file)

    yield path, file

    db.execute(delete(ProcessingJob).where(ProcessingJob.file_id == file.id))
    db.delete(file)
    db.delete(path)
    db.commit()


@pytest.fixture
def path_with_mixed_files(db: Session):
    """One discovered file + one ready file under the same path."""
    path = RegisteredPath(path=_FAKE_PATH + "_mixed")
    db.add(path)
    db.flush()

    discovered = File(
        path_id=path.id,
        filename="new.pdf",
        full_path=f"{_FAKE_PATH}_mixed/new.pdf",
        file_type="pdf",
        file_size=1024,
        file_hash="aabbccdd" * 8,
        file_modified_at=datetime.now(timezone.utc),
        status="discovered",
    )
    ready = File(
        path_id=path.id,
        filename="old.pdf",
        full_path=f"{_FAKE_PATH}_mixed/old.pdf",
        file_type="pdf",
        file_size=512,
        file_hash="11223344" * 8,
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add_all([discovered, ready])
    db.commit()
    db.refresh(discovered)
    db.refresh(ready)

    yield path, discovered, ready

    for f in (discovered, ready):
        db.execute(delete(ProcessingJob).where(ProcessingJob.file_id == f.id))
        db.delete(f)
    db.delete(path)
    db.commit()


def _mock_rq_job() -> tuple[MagicMock, str]:
    rq_job = MagicMock()
    rq_job.id = str(uuid.uuid4())
    return rq_job, rq_job.id


# ---------------------------------------------------------------------------
# POST /process/files
# ---------------------------------------------------------------------------

@patch("app.modules.processing.routes.labeling_queue")
def test_post_process_files_returns_202_and_queued_job(mock_q, path_and_discovered_file):
    _, file = path_and_discovered_file
    rq_job, rq_id = _mock_rq_job()
    mock_q.enqueue.return_value = rq_job

    resp = client.post("/process/files", json={"file_ids": [str(file.id)]})

    assert resp.status_code == 202
    data = resp.json()
    assert len(data) == 1
    assert data[0]["file_id"] == str(file.id)
    assert data[0]["status"] == "queued"
    assert data[0]["rq_job_id"] == rq_id
    mock_q.enqueue.assert_called_once()


@patch("app.modules.processing.routes.labeling_queue")
def test_post_process_files_skips_non_discovered(mock_q, path_with_mixed_files):
    _, discovered, ready = path_with_mixed_files
    rq_job, _ = _mock_rq_job()
    mock_q.enqueue.return_value = rq_job

    resp = client.post(
        "/process/files",
        json={"file_ids": [str(discovered.id), str(ready.id)]},
    )

    assert resp.status_code == 202
    data = resp.json()
    # Only the discovered file gets a job; the ready file is silently skipped.
    assert len(data) == 1
    assert data[0]["file_id"] == str(discovered.id)
    mock_q.enqueue.assert_called_once()


@patch("app.modules.processing.routes.labeling_queue")
def test_post_process_files_unknown_id_returns_404(mock_q, db):
    resp = client.post("/process/files", json={"file_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404
    mock_q.enqueue.assert_not_called()


# ---------------------------------------------------------------------------
# POST /process/paths
# ---------------------------------------------------------------------------

@patch("app.modules.processing.routes.labeling_queue")
def test_post_process_paths_enqueues_only_discovered(mock_q, path_with_mixed_files):
    path, discovered, _ready = path_with_mixed_files
    rq_job, _ = _mock_rq_job()
    mock_q.enqueue.return_value = rq_job

    resp = client.post("/process/paths", json={"path_ids": [str(path.id)]})

    assert resp.status_code == 202
    data = resp.json()
    assert len(data) == 1
    assert data[0]["file_id"] == str(discovered.id)


@patch("app.modules.processing.routes.labeling_queue")
def test_post_process_paths_unknown_id_returns_404(mock_q, db):
    resp = client.post("/process/paths", json={"path_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404
    mock_q.enqueue.assert_not_called()


# ---------------------------------------------------------------------------
# GET /files — processing_job_status polling
# ---------------------------------------------------------------------------

def test_get_files_returns_processing_job_status(db: Session, path_and_discovered_file):
    path, file = path_and_discovered_file

    # Directly insert a ProcessingJob so we don't need RQ.
    job = ProcessingJob(file_id=file.id, triggered_by="manual", status="labeling")
    db.add(job)
    db.commit()

    resp = client.get("/files", params={"path_id": str(path.id)})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == str(file.id)
    assert data[0]["status"] == "discovered"
    assert data[0]["processing_job_status"] == "labeling"


def test_get_files_processing_job_status_is_none_when_no_job(path_and_discovered_file):
    path, file = path_and_discovered_file

    resp = client.get("/files", params={"path_id": str(path.id)})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["processing_job_status"] is None
