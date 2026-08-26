"""Tests for POST /label/files and POST /label/paths endpoints."""
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.main import app
from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.labeling import service
from app.modules.labeling.models import TagKind, TagLabel, TypeLabel, TypeLabelFile
from app.shared.database import get_db
from app.shared.queue import JOB_RETRY
from tests.conftest import mock_rq_job

_FAKE_PATH = "D:/lfa_test_label_routes"


@pytest.fixture
def path_and_ready_file(db: Session):
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="ready.pdf",
        full_path=f"{_FAKE_PATH}/ready.pdf",
        file_type="pdf",
        file_size=2048,
        file_hash="aabbccdd" * 8,
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add(file)
    db.commit()
    db.refresh(path)
    db.refresh(file)
    yield path, file


@pytest.fixture
def path_with_mixed_files(db: Session):
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.flush()

    ready = File(
        path_id=path.id,
        filename="ready.pdf",
        full_path=f"{_FAKE_PATH}/ready.pdf",
        file_type="pdf",
        file_size=1024,
        file_hash="11223344" * 8,
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    discovered = File(
        path_id=path.id,
        filename="new.pdf",
        full_path=f"{_FAKE_PATH}/new.pdf",
        file_type="pdf",
        file_size=512,
        file_hash="55667788" * 8,
        file_modified_at=datetime.now(timezone.utc),
        status="discovered",
    )
    db.add_all([ready, discovered])
    db.commit()
    db.refresh(ready)
    db.refresh(discovered)
    yield path, ready, discovered


_LABELED_FAKE_PATH = "D:/lfa_test_label_routes_labeled"


@pytest.fixture
def ready_file_with_labels(db: Session):
    """A ready file that already has tag_labels — triggers augment mode
    (service.get_files_with_type_or_tag_labels).

    Uses its own path constant (distinct from _FAKE_PATH/path_and_ready_file)
    so a test can request both fixtures together without a paths.path unique
    violation (see test_label_files_batch_assigns_mode_per_file_not_bulk)."""
    path = RegisteredPath(path=_LABELED_FAKE_PATH)
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="labeled.pdf",
        full_path=f"{_LABELED_FAKE_PATH}/labeled.pdf",
        file_type="pdf",
        file_size=2048,
        file_hash="cafebabe" * 8,
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add(file)
    db.flush()

    kind = TagKind(name="lr_test_person")
    db.add(kind)
    db.flush()

    tag = TagLabel(
        file_id=file.id,
        kind_id=kind.id,
        value="lr_test_value",
        source="llm",
        status="confirmed",
    )
    db.add(tag)
    db.commit()
    db.refresh(file)
    yield path, file


_INCOMPLETE_INITIAL_FAKE_PATH = "D:/lfa_test_label_routes_incomplete"


@pytest.fixture
def ready_file_with_incomplete_initial_job(db: Session):
    """A ready file whose initial job's Call 1 (type) succeeded and committed
    a type_labels_files row, but the job then failed before Call 2/3 ever ran
    -- the #61 repro. Row presence alone would misroute a retrigger into
    augment; the fix must catch this via the failed mode=initial job."""
    path = RegisteredPath(path=_INCOMPLETE_INITIAL_FAKE_PATH)
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="incomplete.pdf",
        full_path=f"{_INCOMPLETE_INITIAL_FAKE_PATH}/incomplete.pdf",
        file_type="pdf",
        file_size=2048,
        file_hash="deadbeef" * 8,
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add(file)
    db.flush()

    type_label = TypeLabel(name="lr_test_incomplete_invoice")
    db.add(type_label)
    db.flush()
    db.add(TypeLabelFile(file_id=file.id, type_label_id=type_label.id, source="llm", status="suggested"))
    db.add(
        Job(
            type="label",
            file_id=file.id,
            trigger="manual",
            mode="initial",
            status="failed",
            stage="kinds",
            error_message="LLM timeout on Call 2",
        )
    )
    db.commit()
    db.refresh(file)
    yield path, file


@pytest.fixture
def three_ready_files(db: Session):
    """Three ready files under one path, for batch-order/atomicity tests."""
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.flush()

    files = [
        File(
            path_id=path.id,
            filename=f"ready-{i}.pdf",
            full_path=f"{_FAKE_PATH}/ready-{i}.pdf",
            file_type="pdf",
            file_size=1024,
            file_hash=f"{i}1223344" * 8,
            file_modified_at=datetime.now(timezone.utc),
            status="ready",
        )
        for i in range(3)
    ]
    db.add_all(files)
    db.commit()
    for file in files:
        db.refresh(file)
    yield files


# ---------------------------------------------------------------------------
# POST /label/files — initial mode
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.label_queue")
def test_label_files_initial_returns_202(mock_q, client, path_and_ready_file):
    _, file = path_and_ready_file
    mock_q.enqueue.return_value = mock_rq_job()

    resp = client.post("/label/files", json={"file_ids": [str(file.id)]})

    assert resp.status_code == 202
    data = resp.json()
    assert len(data["enqueued"]) == 1
    assert data["enqueued"][0]["file_id"] == str(file.id)
    assert data["enqueued"][0]["mode"] == "initial"
    assert data["skipped"] == []


@patch("app.modules.labeling.routes.label_queue")
def test_label_files_skips_non_ready(mock_q, client, path_with_mixed_files):
    _, ready, discovered = path_with_mixed_files
    mock_q.enqueue.return_value = mock_rq_job()

    resp = client.post(
        "/label/files",
        json={"file_ids": [str(ready.id), str(discovered.id)]},
    )

    assert resp.status_code == 202
    data = resp.json()
    assert len(data["enqueued"]) == 1
    assert data["enqueued"][0]["file_id"] == str(ready.id)
    assert len(data["skipped"]) == 1
    assert data["skipped"][0]["file_id"] == str(discovered.id)


@patch("app.modules.labeling.routes.label_queue")
def test_label_files_unknown_id_returns_404(mock_q, client):
    resp = client.post("/label/files", json={"file_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404
    mock_q.enqueue.assert_not_called()


# ---------------------------------------------------------------------------
# POST /label/files — augment mode
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.label_queue")
def test_label_files_augment_for_labeled_file(mock_q, client, ready_file_with_labels):
    _, file = ready_file_with_labels
    mock_q.enqueue.return_value = mock_rq_job()

    resp = client.post("/label/files", json={"file_ids": [str(file.id)]})

    assert resp.status_code == 202
    data = resp.json()
    assert len(data["enqueued"]) == 1
    assert data["enqueued"][0]["mode"] == "augment"


@patch("app.modules.labeling.routes.label_queue")
def test_label_files_batch_assigns_mode_per_file_not_bulk(mock_q, client, ready_file_with_labels, path_and_ready_file):
    """One batched existence check must not smear "labeled" across the
    whole request -- each file's mode reflects only its own rows (#47 review:
    file_has_type_or_tag_labels became a batched get_files_with_type_or_tag_labels)."""
    _, labeled_file = ready_file_with_labels
    _, unlabeled_file = path_and_ready_file
    mock_q.enqueue.side_effect = [mock_rq_job(), mock_rq_job()]

    resp = client.post("/label/files", json={"file_ids": [str(labeled_file.id), str(unlabeled_file.id)]})

    assert resp.status_code == 202
    mode_by_file = {e["file_id"]: e["mode"] for e in resp.json()["enqueued"]}
    assert mode_by_file[str(labeled_file.id)] == "augment"
    assert mode_by_file[str(unlabeled_file.id)] == "initial"


# ---------------------------------------------------------------------------
# POST /label/files — #61: incomplete prior initial job must reroute to
# initial, not be misrouted into augment just because Call 1's row exists.
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.label_queue")
def test_label_files_reroutes_to_initial_after_incomplete_initial_job(
    mock_q, client, ready_file_with_incomplete_initial_job
):
    """Regression for #61: Call 1 (type) succeeded and committed a
    type_labels_files row, then the initial job failed before Call 2/3. A
    manual retrigger must redo 'initial', not 'augment' -- augment only ever
    asks about kinds with existing tag_labels rows, finds none here, and
    would silently report success having done nothing."""
    _, file = ready_file_with_incomplete_initial_job
    mock_q.enqueue.return_value = mock_rq_job()

    resp = client.post("/label/files", json={"file_ids": [str(file.id)]})

    assert resp.status_code == 202
    data = resp.json()
    assert len(data["enqueued"]) == 1
    assert data["enqueued"][0]["mode"] == "initial"
    assert data["skipped"] == []


@patch("app.modules.labeling.routes.label_queue")
def test_label_files_augment_when_prior_initial_job_succeeded(mock_q, client, db, ready_file_with_labels):
    """A file whose most recent label job is a fully succeeded 'initial' run
    must still route to 'augment' on the next manual retrigger -- succeeded
    status must not be mistaken for 'incomplete'."""
    _, file = ready_file_with_labels
    db.add(Job(type="label", file_id=file.id, trigger="manual", mode="initial", status="succeeded"))
    db.commit()
    mock_q.enqueue.return_value = mock_rq_job()

    resp = client.post("/label/files", json={"file_ids": [str(file.id)]})

    assert resp.status_code == 202
    assert resp.json()["enqueued"][0]["mode"] == "augment"


# ---------------------------------------------------------------------------
# service.get_files_with_incomplete_initial_job — #61 mode-detection helper
# ---------------------------------------------------------------------------

def test_get_files_with_incomplete_initial_job_flags_failed_initial(db: Session):
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id, filename="failed-initial.pdf", full_path=f"{_FAKE_PATH}/failed-initial.pdf",
        file_type="pdf", file_size=100, file_hash="gfij-failed" * 4,
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(file)
    db.flush()
    db.add(Job(type="label", file_id=file.id, trigger="manual", mode="initial", status="failed", stage="kinds"))
    db.commit()

    assert service.get_files_with_incomplete_initial_job(db, [file.id]) == {file.id}


def test_get_files_with_incomplete_initial_job_uses_latest_job_only(db: Session):
    """An older failed initial job followed by a newer succeeded one must not
    flag the file -- only the SINGLE latest label job decides."""
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id, filename="retried.pdf", full_path=f"{_FAKE_PATH}/retried.pdf",
        file_type="pdf", file_size=100, file_hash="gfij-retried" * 4,
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(file)
    db.flush()

    older = datetime.now(timezone.utc) - timedelta(minutes=10)
    db.add(
        Job(
            type="label", file_id=file.id, trigger="manual", mode="initial", status="failed",
            stage="kinds", created_at=older,
        )
    )
    db.add(Job(type="label", file_id=file.id, trigger="manual", mode="initial", status="succeeded"))
    db.commit()

    assert service.get_files_with_incomplete_initial_job(db, [file.id]) == set()


def test_get_files_with_incomplete_initial_job_ignores_files_without_jobs(db: Session):
    """Rows added directly (e.g. a manual tag, no Job row at all) must not be
    flagged -- absence of job history means "nothing to redo", not "incomplete"."""
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id, filename="no-job.pdf", full_path=f"{_FAKE_PATH}/no-job.pdf",
        file_type="pdf", file_size=100, file_hash="gfij-nojob" * 4,
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add(file)
    db.commit()

    assert service.get_files_with_incomplete_initial_job(db, [file.id]) == set()


def test_get_files_with_incomplete_initial_job_empty_input_returns_empty_set(db: Session):
    assert service.get_files_with_incomplete_initial_job(db, []) == set()


# ---------------------------------------------------------------------------
# service.get_files_with_type_or_tag_labels — batched routing check
# ---------------------------------------------------------------------------

def test_get_files_with_type_or_tag_labels_returns_only_labeled_files(db: Session):
    path = RegisteredPath(path=_FAKE_PATH)
    db.add(path)
    db.flush()

    tagged = File(
        path_id=path.id, filename="tagged.pdf", full_path=f"{_FAKE_PATH}/tagged.pdf",
        file_type="pdf", file_size=100, file_hash="gftl-tagged" * 4,
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    untagged = File(
        path_id=path.id, filename="untagged.pdf", full_path=f"{_FAKE_PATH}/untagged.pdf",
        file_type="pdf", file_size=100, file_hash="gftl-untagged" * 4,
        file_modified_at=datetime.now(timezone.utc), status="ready",
    )
    db.add_all([tagged, untagged])
    db.flush()

    kind = TagKind(name="gftl_test_person")
    db.add(kind)
    db.flush()
    db.add(TagLabel(file_id=tagged.id, kind_id=kind.id, value="Angela Merkel", source="llm", status="confirmed"))
    db.commit()

    result = service.get_files_with_type_or_tag_labels(db, [tagged.id, untagged.id])

    assert result == {tagged.id}


def test_get_files_with_type_or_tag_labels_empty_input_returns_empty_set(db: Session):
    assert service.get_files_with_type_or_tag_labels(db, []) == set()


# ---------------------------------------------------------------------------
# POST /label/files — creates correct Job row
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.label_queue")
def test_label_files_creates_job_in_db(mock_q, client, db, path_and_ready_file):
    _, file = path_and_ready_file
    mock_q.enqueue.return_value = mock_rq_job()

    resp = client.post("/label/files", json={"file_ids": [str(file.id)]})

    data = resp.json()
    job_id = data["enqueued"][0]["job_id"]
    job = db.get(Job, uuid.UUID(job_id))
    assert job is not None
    assert job.type == "label"
    assert job.trigger == "manual"
    assert job.mode == "initial"
    assert job.file_id == file.id


# ---------------------------------------------------------------------------
# POST /label/paths
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.label_queue")
def test_label_paths_enqueues_ready_files(mock_q, client, path_with_mixed_files):
    path, ready, _discovered = path_with_mixed_files
    mock_q.enqueue.return_value = mock_rq_job()

    resp = client.post("/label/paths", json={"path_ids": [str(path.id)]})

    assert resp.status_code == 202
    data = resp.json()
    enqueued_file_ids = {e["file_id"] for e in data["enqueued"]}
    assert str(ready.id) in enqueued_file_ids


@patch("app.modules.labeling.routes.label_queue")
def test_label_paths_unknown_path_returns_404(mock_q, client):
    resp = client.post("/label/paths", json={"path_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404
    mock_q.enqueue.assert_not_called()


# ---------------------------------------------------------------------------
# Enqueue bookkeeping and retry consistency (#33)
# ---------------------------------------------------------------------------

@patch("app.modules.labeling.routes.label_queue")
def test_label_files_stores_rq_job_id_on_job(mock_q, client, db, path_and_ready_file):
    _, file = path_and_ready_file
    mock_q.enqueue.return_value = mock_rq_job("rq-abc-123")

    resp = client.post("/label/files", json={"file_ids": [str(file.id)]})

    job_id = resp.json()["enqueued"][0]["job_id"]
    job = db.get(Job, uuid.UUID(job_id))
    assert job.rq_job_id == "rq-abc-123"


@patch("app.modules.labeling.routes.label_queue")
def test_label_files_enqueues_without_at_front_using_shared_retry(mock_q, client, path_and_ready_file):
    _, file = path_and_ready_file
    mock_q.enqueue.return_value = mock_rq_job()

    client.post("/label/files", json={"file_ids": [str(file.id)]})

    _args, kwargs = mock_q.enqueue.call_args
    assert kwargs.get("retry") is JOB_RETRY
    assert "at_front" not in kwargs


@patch("app.modules.labeling.routes.label_queue")
def test_label_files_batch_enqueues_in_submission_order(mock_q, client, three_ready_files):
    mock_q.enqueue.side_effect = [mock_rq_job() for _ in three_ready_files]
    file_ids = [str(f.id) for f in three_ready_files]

    resp = client.post("/label/files", json={"file_ids": file_ids})

    enqueued_file_ids = [e["file_id"] for e in resp.json()["enqueued"]]
    assert enqueued_file_ids == file_ids

    # job.id passed as the 2nd positional arg to enqueue(), in call order.
    submitted_job_ids = [call.args[1] for call in mock_q.enqueue.call_args_list]
    response_job_ids = [uuid.UUID(e["job_id"]) for e in resp.json()["enqueued"]]
    assert submitted_job_ids == response_job_ids


def test_label_files_batch_rolls_back_all_jobs_if_one_enqueue_fails(db: Session, three_ready_files):
    """All-or-nothing: a mid-batch enqueue failure must not leave partial Job rows."""

    def override_get_db():
        # Mirror the real get_db's try/finally: an exception mid-request must
        # roll back whatever this request flushed, same as production.
        try:
            yield db
        finally:
            db.rollback()

    app.dependency_overrides[get_db] = override_get_db
    local_client = TestClient(app, raise_server_exceptions=False)

    with patch("app.modules.labeling.routes.label_queue") as mock_q:
        mock_q.enqueue.side_effect = [mock_rq_job(), RuntimeError("redis unreachable"), mock_rq_job()]
        file_ids = [str(f.id) for f in three_ready_files]

        resp = local_client.post("/label/files", json={"file_ids": file_ids})

    app.dependency_overrides.clear()

    assert resp.status_code == 500
    remaining = list(db.scalars(select(Job).where(Job.file_id.in_([f.id for f in three_ready_files]))))
    assert remaining == []
