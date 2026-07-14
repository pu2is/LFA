"""run_label's job-lifecycle bookkeeping (suggestion logic itself is covered
in test_suggest_labels.py / test_merge_labels.py)."""
from unittest.mock import patch

import pytest

from app.modules.labeling.tasks import run_label
from tests.factories import make_failed_job


@pytest.fixture
def label_job(db):
    return make_failed_job(
        db, job_type="label", trigger="manual", mode="initial", file_status="ready", error_message="LLM parse error"
    )


@pytest.fixture
def augment_label_job(db):
    return make_failed_job(
        db, job_type="label", trigger="manual", mode="augment", file_status="ready", error_message="LLM parse error"
    )


@patch("app.modules.labeling.tasks.suggest_labels")
def test_run_label_clears_stale_error_message_on_success(mock_suggest_labels, db, label_job):
    mock_suggest_labels.return_value = ([], [])

    result = run_label(db, label_job.id)

    assert result.status == "succeeded"
    assert result.error_message is None
    assert result.stage is None


# --------------------------------------------------------------------------- #
# ADR-0001 D3: initial mode's 3-stage handoff to suggest_labels
# --------------------------------------------------------------------------- #

@patch("app.modules.labeling.tasks.suggest_labels")
def test_run_label_initial_sets_type_stage_before_calling_suggest_labels(mock_suggest_labels, db, label_job):
    """run_label owns entering the FIRST stage; suggest_labels advances the rest."""
    captured_stage = None

    def _capture(_db, _file_id, job, **_kwargs):
        nonlocal captured_stage
        captured_stage = job.stage
        return ([], [])

    mock_suggest_labels.side_effect = _capture

    run_label(db, label_job.id)

    assert captured_stage == "type"


@patch("app.modules.labeling.tasks.suggest_labels")
def test_run_label_initial_failure_preserves_stage_reached(mock_suggest_labels, db, label_job):
    """A mid-flow failure must mark the job failed WITHOUT resetting job.stage
    -- the stage it died at is diagnostic info (see docs/03_er-diagram.md)."""
    def _fail(_db, _file_id, job, **_kwargs):
        job.stage = "kinds"
        raise RuntimeError("ollama unreachable")

    mock_suggest_labels.side_effect = _fail

    with pytest.raises(RuntimeError, match="ollama unreachable"):
        run_label(db, label_job.id)

    assert label_job.status == "failed"
    assert label_job.error_message == "ollama unreachable"
    assert label_job.stage == "kinds"


# --------------------------------------------------------------------------- #
# Augment mode is untouched by the D3 rewrite (separate follow-up ticket)
# --------------------------------------------------------------------------- #

@patch("app.modules.labeling.tasks.suggest_labels_augment")
def test_run_label_augment_still_uses_single_labeling_stage(mock_suggest_labels_augment, db, augment_label_job):
    captured_stage = None

    def _capture(_db, _file_id, **_kwargs):
        nonlocal captured_stage
        captured_stage = augment_label_job.stage
        return []

    mock_suggest_labels_augment.side_effect = _capture

    result = run_label(db, augment_label_job.id)

    assert captured_stage == "labeling"
    assert result.status == "succeeded"
    assert result.stage is None
