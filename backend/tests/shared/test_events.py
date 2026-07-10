import uuid

from app.modules.jobs.models import Job
from app.shared import events


def test_publish_job_status_includes_fields_that_are_set(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        events, "publish_event", lambda event_type, data: captured.update(event_type=event_type, data=data)
    )

    job = Job(
        id=uuid.uuid4(),
        type="scan",
        path_id=uuid.uuid4(),
        status="running",
        stage="discovering",
        trigger="scan",
        mode="initial",
        error_message="partial failure",
    )

    events.publish_job_status(job, file_count=3)

    assert captured["event_type"] == "job_status"
    assert captured["data"] == {
        "job_id": str(job.id),
        "type": "scan",
        "status": "running",
        "path_id": str(job.path_id),
        "stage": "discovering",
        "mode": "initial",
        "error_message": "partial failure",
        "file_count": 3,
    }


def test_publish_job_status_omits_fields_that_are_unset(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(events, "publish_event", lambda event_type, data: captured.update(data=data))

    job = Job(
        id=uuid.uuid4(),
        type="embed",
        file_id=uuid.uuid4(),
        status="succeeded",
        trigger="scan",
    )

    events.publish_job_status(job)

    assert captured["data"] == {
        "job_id": str(job.id),
        "type": "embed",
        "status": "succeeded",
        "file_id": str(job.file_id),
    }
