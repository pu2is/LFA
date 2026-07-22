"""Unit tests for labeling.suggestion.suggest_labels_augment() -- ADR-0001 D4
f1 (mode=augment): one independent call per tag_kind this file already has
values under, append-only.

All tests use a MagicMock LLM -- Ollama is never called. Kinds are iterated
in name order (see suggestion.py), so multi-kind tests use alphabetically
distinct names and align mock outputs to that order.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.labeling.models import TagKind, TagLabel, TypeLabelFile
from app.modules.labeling.prompts import TagValuesOutput
from app.modules.labeling.suggestion import suggest_labels_augment
from app.modules.rag.models import FileChunk


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def file_id(db):
    path = RegisteredPath(path="/tmp/lfa_suggest_labels_augment_test")
    db.add(path)
    db.flush()

    file = File(
        path_id=path.id,
        filename="doc.pdf",
        full_path="/tmp/lfa_suggest_labels_augment_test/doc.pdf",
        file_type="pdf",
        file_size=1000,
        file_hash="suggest-labels-augment-test",
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add(file)
    db.flush()
    db.add(FileChunk(file_id=file.id, chunk_index=0, content="Invoice issued to Angela Merkel and Acme Corp."))
    db.commit()
    db.refresh(file)
    return file.id


@pytest.fixture
def person_kind(db):
    kind = TagKind(name="person")
    db.add(kind)
    db.commit()
    db.refresh(kind)
    return kind


@pytest.fixture
def organization_kind(db):
    kind = TagKind(name="organization")
    db.add(kind)
    db.commit()
    db.refresh(kind)
    return kind


def _mock_llm(values_per_call: list[list[str]]) -> tuple[MagicMock, MagicMock]:
    """with_structured_output is called once and reused across all kinds;
    invoke()'s side_effect supplies one TagValuesOutput per kind, in the
    same (name) order suggest_labels_augment iterates kinds."""
    chain = MagicMock()
    chain.invoke.side_effect = [TagValuesOutput(values=v) for v in values_per_call]
    llm = MagicMock()
    llm.with_structured_output.return_value = chain
    return llm, chain


# --------------------------------------------------------------------------- #
# Happy path / early exits
# --------------------------------------------------------------------------- #

def test_augment_inserts_new_value_for_existing_kind(db, file_id, person_kind):
    db.add(TagLabel(file_id=file_id, kind_id=person_kind.id, value="Angela Merkel", source="llm", status="confirmed"))
    db.commit()

    llm, _ = _mock_llm([["Barack Obama"]])
    result = suggest_labels_augment(db, file_id, llm=llm)

    assert len(result) == 1
    assert result[0].value == "Barack Obama"
    assert result[0].source == "llm"
    assert result[0].status == "suggested"


def test_augment_returns_empty_when_no_existing_tag_labels(db, file_id):
    """No existing tag_labels -> nothing to augment; LLM is never called."""
    mock_llm = MagicMock()
    result = suggest_labels_augment(db, file_id, llm=mock_llm)

    assert result == []
    mock_llm.with_structured_output.assert_not_called()


def test_augment_returns_empty_when_no_chunks(db, person_kind):
    path = RegisteredPath(path="/tmp/lfa_augment_no_chunks_test")
    db.add(path)
    db.flush()
    file = File(
        path_id=path.id,
        filename="empty.pdf",
        full_path="/tmp/lfa_augment_no_chunks_test/empty.pdf",
        file_type="pdf",
        file_size=0,
        file_hash="augment-no-chunks-test",
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add(file)
    db.flush()
    db.add(TagLabel(file_id=file.id, kind_id=person_kind.id, value="Angela Merkel", source="llm", status="confirmed"))
    db.commit()

    mock_llm = MagicMock()
    result = suggest_labels_augment(db, file.id, llm=mock_llm)

    assert result == []
    mock_llm.with_structured_output.assert_not_called()


def test_augment_handles_multiple_existing_kinds_in_name_order(db, file_id, person_kind, organization_kind):
    db.add(TagLabel(file_id=file_id, kind_id=person_kind.id, value="Angela Merkel", source="llm", status="confirmed"))
    db.add(TagLabel(file_id=file_id, kind_id=organization_kind.id, value="Acme Corp", source="llm", status="confirmed"))
    db.commit()

    # "organization" < "person" alphabetically.
    llm, _ = _mock_llm([["New Org"], ["New Person"]])
    result = suggest_labels_augment(db, file_id, llm=llm)

    values_by_kind = {row.value: db.get(TagKind, row.kind_id).name for row in result}
    assert values_by_kind == {"New Org": "organization", "New Person": "person"}


# --------------------------------------------------------------------------- #
# Append-only semantics (AC: confirmed/rejected/suggested stay byte-identical)
# --------------------------------------------------------------------------- #

def test_augment_never_touches_confirmed_rows(db, file_id, person_kind):
    confirmed = TagLabel(file_id=file_id, kind_id=person_kind.id, value="Angela Merkel", source="llm", status="confirmed")
    db.add(confirmed)
    db.commit()
    db.refresh(confirmed)
    confirmed_snapshot = (confirmed.value, confirmed.source, confirmed.status, confirmed.created_at)

    llm, _ = _mock_llm([["Barack Obama"]])
    suggest_labels_augment(db, file_id, llm=llm)

    db.refresh(confirmed)
    assert (confirmed.value, confirmed.source, confirmed.status, confirmed.created_at) == confirmed_snapshot


def test_augment_never_flips_rejected_back_to_suggested(db, file_id, person_kind):
    rejected = TagLabel(file_id=file_id, kind_id=person_kind.id, value="Wrong Name", source="llm", status="rejected")
    db.add(rejected)
    db.commit()
    db.refresh(rejected)

    # LLM re-suggests the exact rejected value plus one genuinely new one.
    llm, _ = _mock_llm([["Wrong Name", "Barack Obama"]])
    result = suggest_labels_augment(db, file_id, llm=llm)

    assert [row.value for row in result] == ["Barack Obama"]
    db.refresh(rejected)
    assert rejected.status == "rejected"


def test_augment_never_touches_existing_suggested_rows(db, file_id, person_kind):
    suggested = TagLabel(file_id=file_id, kind_id=person_kind.id, value="Angela Merkel", source="llm", status="suggested")
    db.add(suggested)
    db.commit()
    db.refresh(suggested)

    llm, _ = _mock_llm([["Barack Obama"]])
    suggest_labels_augment(db, file_id, llm=llm)

    db.refresh(suggested)
    assert suggested.status == "suggested"
    assert suggested.value == "Angela Merkel"


def test_augment_deduplicates_against_all_existing_values_regardless_of_status(db, file_id, person_kind):
    """Confirmed, rejected, AND suggested values are all in the don't-repeat set."""
    db.add_all(
        [
            TagLabel(file_id=file_id, kind_id=person_kind.id, value="Confirmed Person", source="llm", status="confirmed"),
            TagLabel(file_id=file_id, kind_id=person_kind.id, value="Rejected Person", source="llm", status="rejected"),
            TagLabel(file_id=file_id, kind_id=person_kind.id, value="Suggested Person", source="llm", status="suggested"),
        ]
    )
    db.commit()

    llm, _ = _mock_llm([["Confirmed Person", "Rejected Person", "Suggested Person", "Genuinely New Person"]])
    result = suggest_labels_augment(db, file_id, llm=llm)

    assert [row.value for row in result] == ["Genuinely New Person"]


def test_augment_deduplicates_case_variant_of_existing_value(db, file_id, person_kind):
    """#49: augment's temperature=0.7 tends to re-suggest case variants of a
    value already there -- "angela merkel" must not duplicate "Angela Merkel"."""
    db.add(TagLabel(file_id=file_id, kind_id=person_kind.id, value="Angela Merkel", source="llm", status="confirmed"))
    db.commit()

    llm, _ = _mock_llm([["angela merkel"]])
    result = suggest_labels_augment(db, file_id, llm=llm)

    assert result == []


def test_augment_skips_empty_output_for_a_kind(db, file_id, person_kind):
    db.add(TagLabel(file_id=file_id, kind_id=person_kind.id, value="Angela Merkel", source="llm", status="confirmed"))
    db.commit()

    llm, _ = _mock_llm([[]])
    result = suggest_labels_augment(db, file_id, llm=llm)

    assert result == []


# --------------------------------------------------------------------------- #
# Structural guarantee: never writes type_labels_files
# --------------------------------------------------------------------------- #

def test_augment_never_writes_type_labels_files(db, file_id, person_kind):
    db.add(TagLabel(file_id=file_id, kind_id=person_kind.id, value="Angela Merkel", source="llm", status="confirmed"))
    db.commit()

    llm, _ = _mock_llm([["Barack Obama"]])
    suggest_labels_augment(db, file_id, llm=llm)

    assert list(db.scalars(select(TypeLabelFile).where(TypeLabelFile.file_id == file_id))) == []


# --------------------------------------------------------------------------- #
# Per-kind commit isolation and retry-safety
# --------------------------------------------------------------------------- #

def test_augment_kind_failure_preserves_earlier_kinds_writes(db, file_id, person_kind, organization_kind):
    db.add(TagLabel(file_id=file_id, kind_id=person_kind.id, value="Angela Merkel", source="llm", status="confirmed"))
    db.add(TagLabel(file_id=file_id, kind_id=organization_kind.id, value="Acme Corp", source="llm", status="confirmed"))
    db.commit()

    # "organization" is processed first (name order); make its call succeed,
    # then fail on "person".
    chain = MagicMock()
    chain.invoke.side_effect = [TagValuesOutput(values=["New Org"]), RuntimeError("ollama unreachable")]
    llm = MagicMock()
    llm.with_structured_output.return_value = chain

    with pytest.raises(RuntimeError, match="ollama unreachable"):
        suggest_labels_augment(db, file_id, llm=llm)

    survived = list(db.scalars(select(TagLabel).where(TagLabel.file_id == file_id, TagLabel.value == "New Org")))
    assert len(survived) == 1


def test_augment_rerun_is_idempotent_not_a_crash(db, file_id, person_kind):
    db.add(TagLabel(file_id=file_id, kind_id=person_kind.id, value="Angela Merkel", source="llm", status="confirmed"))
    db.commit()

    llm1, _ = _mock_llm([["Barack Obama"]])
    first = suggest_labels_augment(db, file_id, llm=llm1)

    llm2, _ = _mock_llm([["Barack Obama"]])
    second = suggest_labels_augment(db, file_id, llm=llm2)

    assert len(first) == 1
    assert second == []  # already there -- no crash, no duplicate
    all_values = list(db.scalars(select(TagLabel.value).where(TagLabel.file_id == file_id)))
    assert sorted(all_values) == ["Angela Merkel", "Barack Obama"]


# --------------------------------------------------------------------------- #
# Injectable temperature
# --------------------------------------------------------------------------- #

def test_augment_defaults_to_a_positive_temperature_when_no_llm_injected(db, file_id, person_kind):
    db.add(TagLabel(file_id=file_id, kind_id=person_kind.id, value="Angela Merkel", source="llm", status="confirmed"))
    db.commit()

    with patch("app.modules.labeling.suggestion.ChatOllama") as mock_chat_ollama:
        mock_chat_ollama.return_value.with_structured_output.return_value.invoke.return_value = TagValuesOutput(
            values=[]
        )
        suggest_labels_augment(db, file_id)

    _args, kwargs = mock_chat_ollama.call_args
    assert kwargs["temperature"] > 0


def test_augment_temperature_is_injectable(db, file_id, person_kind):
    db.add(TagLabel(file_id=file_id, kind_id=person_kind.id, value="Angela Merkel", source="llm", status="confirmed"))
    db.commit()

    with patch("app.modules.labeling.suggestion.ChatOllama") as mock_chat_ollama:
        mock_chat_ollama.return_value.with_structured_output.return_value.invoke.return_value = TagValuesOutput(
            values=[]
        )
        suggest_labels_augment(db, file_id, temperature=0.3)

    _args, kwargs = mock_chat_ollama.call_args
    assert kwargs["temperature"] == 0.3
