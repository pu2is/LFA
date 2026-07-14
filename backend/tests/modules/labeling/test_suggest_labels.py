"""Unit tests for labeling.suggestion.suggest_labels() -- ADR-0001 D3's
three-stage initial flow (type -> kinds -> per-kind tags).

All tests use a MagicMock LLM -- Ollama is never called. with_structured_output
is invoked exactly 3 times per run, in order (type, kinds, tags); the tags
chain's invoke() is called once per kind select_kinds() actually chose, so
mock outputs for that stage must be supplied in that same (Call-2) order.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.labeling.models import TagKind, TagLabel, TypeLabel, TypeLabelFile
from app.modules.labeling.prompts import (
    InitialKindCandidate,
    InitialKindSuggestionOutput,
    InitialTypeCandidate,
    InitialTypeSuggestionOutput,
    TagValuesOutput,
)
from app.modules.labeling.suggestion import suggest_labels
from app.modules.rag.models import FileChunk


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def file_and_job(db):
    path = RegisteredPath(path="/tmp/lfa_suggest_labels_init_test")
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="doc.pdf",
        full_path="/tmp/lfa_suggest_labels_init_test/doc.pdf",
        file_type="pdf",
        file_size=1000,
        file_hash="suggest-labels-init-test",
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add(file)
    db.flush()
    db.add(FileChunk(file_id=file.id, chunk_index=0, content="Invoice No. 123 issued to Angela Merkel."))
    db.commit()

    job = Job(type="label", file_id=file.id, trigger="manual", mode="initial", stage="type")
    db.add(job)
    db.commit()
    db.refresh(file)
    db.refresh(job)
    return file, job


@pytest.fixture
def seeded_catalogs(db):
    """3 type_labels + 3 tag_kinds, seeded directly (not via presets)."""
    types = [TypeLabel(name=n) for n in ["invoice", "contract", "report"]]
    kinds = [TagKind(name=n) for n in ["person", "organization", "place"]]
    db.add_all(types + kinds)
    db.commit()
    for row in types + kinds:
        db.refresh(row)
    return types, kinds


def _mock_llm(
    type_names: list[str] | None = None,
    kind_names: list[str] | None = None,
    tag_values_per_kind: list[list[str]] | None = None,
) -> MagicMock:
    """Build a mock BaseChatModel for the 3-stage initial flow.

    tag_values_per_kind must be supplied in the order select_kinds() will
    actually choose kinds in (Call 2's output order, filtered to catalog
    matches) -- not necessarily the same order/length as kind_names.
    """
    type_output = InitialTypeSuggestionOutput(types=[InitialTypeCandidate(name=n) for n in (type_names or [])])
    kind_output = InitialKindSuggestionOutput(kinds=[InitialKindCandidate(name=n) for n in (kind_names or [])])

    type_chain = MagicMock()
    type_chain.invoke.return_value = type_output

    kind_chain = MagicMock()
    kind_chain.invoke.return_value = kind_output

    tag_chain = MagicMock()
    tag_chain.invoke.side_effect = [
        TagValuesOutput(values=values) for values in (tag_values_per_kind or [])
    ]

    mock_llm = MagicMock()
    mock_llm.with_structured_output.side_effect = [type_chain, kind_chain, tag_chain]
    return mock_llm, type_chain, kind_chain, tag_chain


# --------------------------------------------------------------------------- #
# Happy-path tests
# --------------------------------------------------------------------------- #

def test_suggest_labels_produces_type_and_tag_rows(db, file_and_job, seeded_catalogs):
    file, job = file_and_job
    llm, *_ = _mock_llm(
        type_names=["invoice"],
        kind_names=["person"],
        tag_values_per_kind=[["Angela Merkel"]],
    )

    type_rows, tag_rows = suggest_labels(db, file.id, job, llm=llm)

    assert len(type_rows) == 1
    assert type_rows[0].source == "llm"
    assert type_rows[0].status == "suggested"

    assert len(tag_rows) == 1
    assert tag_rows[0].value == "Angela Merkel"
    assert tag_rows[0].source == "llm"
    assert tag_rows[0].status == "suggested"


def test_suggest_labels_drops_unknown_type_name(db, file_and_job, seeded_catalogs):
    """A type name not in the type_labels catalog is silently dropped."""
    file, job = file_and_job
    llm, *_ = _mock_llm(type_names=["invoice", "not_a_real_type"], kind_names=[], tag_values_per_kind=[])

    type_rows, _ = suggest_labels(db, file.id, job, llm=llm)

    assert len(type_rows) == 1


def test_suggest_labels_deduplicates_repeated_type_pick(db, file_and_job, seeded_catalogs):
    file, job = file_and_job
    llm, *_ = _mock_llm(type_names=["invoice", "invoice"], kind_names=[], tag_values_per_kind=[])

    type_rows, _ = suggest_labels(db, file.id, job, llm=llm)

    assert len(type_rows) == 1


def test_suggest_labels_drops_unknown_kind_name(db, file_and_job, seeded_catalogs):
    """A kind name not in the tag_kinds catalog never reaches Call 3."""
    file, job = file_and_job
    llm, _type_chain, _kind_chain, tag_chain = _mock_llm(
        type_names=[], kind_names=["not_a_real_kind"], tag_values_per_kind=[]
    )

    _, tag_rows = suggest_labels(db, file.id, job, llm=llm)

    assert tag_rows == []
    tag_chain.invoke.assert_not_called()


def test_suggest_labels_skips_empty_kind_values(db, file_and_job, seeded_catalogs):
    """A chosen kind whose Call-3 output is empty writes nothing -- no error."""
    file, job = file_and_job
    llm, *_ = _mock_llm(type_names=[], kind_names=["person"], tag_values_per_kind=[[]])

    _, tag_rows = suggest_labels(db, file.id, job, llm=llm)

    assert tag_rows == []


def test_suggest_labels_deduplicates_repeated_tag_value(db, file_and_job, seeded_catalogs):
    file, job = file_and_job
    llm, *_ = _mock_llm(
        type_names=[], kind_names=["person"], tag_values_per_kind=[["Angela Merkel", "Angela Merkel"]]
    )

    _, tag_rows = suggest_labels(db, file.id, job, llm=llm)

    assert len(tag_rows) == 1


def test_suggest_labels_handles_multiple_chosen_kinds_in_order(db, file_and_job, seeded_catalogs):
    file, job = file_and_job
    llm, *_ = _mock_llm(
        type_names=[],
        kind_names=["person", "place"],
        tag_values_per_kind=[["Angela Merkel"], ["Berlin"]],
    )

    _, tag_rows = suggest_labels(db, file.id, job, llm=llm)

    values_by_kind_name = {row.value: db.get(TagKind, row.kind_id).name for row in tag_rows}
    assert values_by_kind_name == {"Angela Merkel": "person", "Berlin": "place"}


# --------------------------------------------------------------------------- #
# Catalog auto-seeding
# --------------------------------------------------------------------------- #

def test_suggest_labels_auto_populates_both_catalogs_when_empty(db, file_and_job):
    """When type_labels/tag_kinds are both empty, presets fill them before the LLM runs."""
    file, job = file_and_job
    assert db.scalars(select(TypeLabel)).first() is None
    assert db.scalars(select(TagKind)).first() is None

    llm, *_ = _mock_llm(type_names=[], kind_names=[], tag_values_per_kind=[])
    suggest_labels(db, file.id, job, llm=llm)

    assert len(list(db.scalars(select(TypeLabel)))) > 0
    assert len(list(db.scalars(select(TagKind)))) > 0


def test_suggest_labels_returns_empty_when_no_chunks(db, seeded_catalogs):
    path = RegisteredPath(path="/tmp/lfa_no_chunks_init_test")
    db.add(path)
    db.flush()
    file = File(
        path_id=path.id,
        filename="empty.pdf",
        full_path="/tmp/lfa_no_chunks_init_test/empty.pdf",
        file_type="pdf",
        file_size=0,
        file_hash="no-chunks-init-test",
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add(file)
    db.flush()
    job = Job(type="label", file_id=file.id, trigger="manual", mode="initial", stage="type")
    db.add(job)
    db.commit()

    mock_llm = MagicMock()
    type_rows, tag_rows = suggest_labels(db, file.id, job, llm=mock_llm)

    assert (type_rows, tag_rows) == ([], [])
    mock_llm.with_structured_output.assert_not_called()


# --------------------------------------------------------------------------- #
# Stage progression (GET /jobs visibility) and partial-failure durability
# --------------------------------------------------------------------------- #

def test_suggest_labels_advances_stage_through_kinds_and_tags(db, file_and_job, seeded_catalogs):
    file, job = file_and_job
    llm, *_ = _mock_llm(
        type_names=["invoice"], kind_names=["person"], tag_values_per_kind=[["Angela Merkel"]]
    )

    stages_seen: list[str | None] = []

    def _record(_db, _job):
        stages_seen.append(_job.stage)

    with patch("app.modules.labeling.suggestion.mark_progress", side_effect=_record):
        suggest_labels(db, file.id, job, llm=llm)

    # job.stage started at "type" (set by the caller, run_label); suggest_labels
    # itself only needs to advance it through the remaining two stages.
    assert stages_seen == ["kinds", "tags"]


def test_suggest_labels_rerun_on_same_file_is_idempotent_not_a_crash(db, file_and_job, seeded_catalogs):
    """RQ retries reuse the same job row (#33). Once this file has ANY
    type_labels_files/tag_labels row, /label routes it to mode=augment (see
    service.file_has_type_or_tag_labels) instead of calling suggest_labels
    again -- but suggest_labels itself must still be safe to call twice
    directly (e.g. a retry mid-run, before any row exists yet): running it
    twice with overlapping LLM output must not hit the (file_id,
    type_label_id) / (file_id, kind_id, value) UNIQUE constraints."""
    file, job = file_and_job

    def _make_llm():
        llm, *_ = _mock_llm(
            type_names=["invoice"], kind_names=["person"], tag_values_per_kind=[["Angela Merkel"]]
        )
        return llm

    first_types, first_tags = suggest_labels(db, file.id, job, llm=_make_llm())
    second_types, second_tags = suggest_labels(db, file.id, job, llm=_make_llm())

    assert len(first_types) == 1 and len(first_tags) == 1
    assert second_types == [] and second_tags == []  # nothing NEW to insert -- no crash either

    assert len(list(db.scalars(select(TypeLabelFile).where(TypeLabelFile.file_id == file.id)))) == 1
    assert len(list(db.scalars(select(TagLabel).where(TagLabel.file_id == file.id)))) == 1


def test_suggest_labels_preserves_earlier_writes_on_mid_flow_failure(db, file_and_job, seeded_catalogs):
    """Call 2 failing must not roll back Call 1's already-committed type picks."""
    file, job = file_and_job
    llm, _type_chain, kind_chain, _tag_chain = _mock_llm(type_names=["invoice"])
    kind_chain.invoke.side_effect = RuntimeError("ollama unreachable")

    with pytest.raises(RuntimeError, match="ollama unreachable"):
        suggest_labels(db, file.id, job, llm=llm)

    survived = list(db.scalars(select(TypeLabelFile).where(TypeLabelFile.file_id == file.id)))
    assert len(survived) == 1


# --------------------------------------------------------------------------- #
# Chunk sizing (01b: ~1-2 pages, not the full document)
# --------------------------------------------------------------------------- #

def test_suggest_labels_defaults_to_a_small_chunk_window(db, seeded_catalogs):
    path = RegisteredPath(path="/tmp/lfa_chunk_window_test")
    db.add(path)
    db.flush()
    file = File(
        path_id=path.id,
        filename="long.pdf",
        full_path="/tmp/lfa_chunk_window_test/long.pdf",
        file_type="pdf",
        file_size=1000,
        file_hash="chunk-window-test",
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add(file)
    db.flush()
    # More chunks than the default window; only the marker in the LAST chunk
    # should be excluded from what Call 1 sees.
    for i in range(8):
        marker = "LATE_CHUNK_MARKER" if i == 7 else f"early content {i}"
        db.add(FileChunk(file_id=file.id, chunk_index=i, content=marker))
    db.commit()
    job = Job(type="label", file_id=file.id, trigger="manual", mode="initial", stage="type")
    db.add(job)
    db.commit()

    llm, type_chain, *_ = _mock_llm(type_names=[], kind_names=[], tag_values_per_kind=[])
    suggest_labels(db, file.id, job, llm=llm)

    sent_messages = type_chain.invoke.call_args.args[0]
    sent_text = "\n".join(getattr(m, "content", "") for m in sent_messages)
    assert "LATE_CHUNK_MARKER" not in sent_text
    assert "early content 0" in sent_text
