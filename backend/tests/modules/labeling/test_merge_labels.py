"""Tests for labeling.service.normalize_label_name, and for merge.write_type_
candidates' (#53) and write_tag_candidates' (#54) atomicity.

merge.py's other write functions are covered elsewhere: write_initial_
candidates has no remaining caller (kept per #45's scope-out, not tested
further); select_kinds is covered in test_suggest_labels.py, since ADR-0001's
flow makes it meaningful only in sequence, not in isolation.
"""
from datetime import datetime, timezone

from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.labeling.merge import write_tag_candidates, write_type_candidates
from app.modules.labeling.models import TagKind, TagLabel, TypeLabel, TypeLabelFile
from app.modules.labeling.prompts import InitialTypeCandidate, InitialTypeSuggestionOutput, TagValuesOutput
from app.modules.labeling.service import normalize_label_name, upsert_user_tag_label


def test_normalize_label_name_replaces_spaces_with_underscores():
    assert normalize_label_name("Bank Statement") == "bank_statement"
    assert normalize_label_name("  Tax  ") == "tax"
    assert normalize_label_name("INVOICE") == "invoice"
    assert normalize_label_name("car_rental_agreement") == "car_rental_agreement"


# --------------------------------------------------------------------------- #
# write_type_candidates atomicity (#53)
# --------------------------------------------------------------------------- #

def _file(db) -> File:
    path = RegisteredPath(path="/tmp/lfa_write_type_candidates_test")
    db.add(path)
    db.flush()
    file = File(
        path_id=path.id,
        filename="doc.pdf",
        full_path="/tmp/lfa_write_type_candidates_test/doc.pdf",
        file_type="pdf",
        file_size=1000,
        file_hash="write-type-candidates-test",
        file_modified_at=datetime.now(timezone.utc),
        status="ready",
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


def test_write_type_candidates_survives_a_racing_duplicate_insert(db):
    """#53: the old SELECT-then-INSERT let two concurrent writers (RQ retries
    reusing the same job row, see #33) both see "not yet present" and race to
    insert the same (file_id, type_label_id) row -- the loser's commit then
    crashed on the UNIQUE constraint. INSERT ... ON CONFLICT DO NOTHING makes
    this a structural non-issue: simulated here the same way #51 simulates
    the catalog-seed race, by re-running the write with overlapping output
    against a session that never re-reads existing rows first."""
    file = _file(db)
    type_label = TypeLabel(name="invoice")
    db.add(type_label)
    db.commit()
    db.refresh(type_label)

    output = InitialTypeSuggestionOutput(types=[InitialTypeCandidate(name="invoice")])

    first = write_type_candidates(db, file.id, output, [type_label])
    second = write_type_candidates(db, file.id, output, [type_label])  # simulated racing writer

    assert len(first) == 1
    assert first[0].type_label_id == type_label.id
    assert second == []  # conflicting insert silently skipped -- no IntegrityError

    rows = list(db.scalars(select(TypeLabelFile).where(TypeLabelFile.file_id == file.id)))
    assert len(rows) == 1


def test_write_type_candidates_inserts_non_conflicting_rows_only(db):
    """A mix of a brand-new type and one that already exists on the file:
    only the new one comes back, and no exception is raised for the other."""
    file = _file(db)
    invoice, contract = TypeLabel(name="invoice"), TypeLabel(name="contract")
    db.add_all([invoice, contract])
    db.commit()
    db.refresh(invoice)
    db.refresh(contract)

    write_type_candidates(db, file.id, InitialTypeSuggestionOutput(types=[InitialTypeCandidate(name="invoice")]), [invoice, contract])

    output = InitialTypeSuggestionOutput(
        types=[InitialTypeCandidate(name="invoice"), InitialTypeCandidate(name="contract")]
    )
    rows = write_type_candidates(db, file.id, output, [invoice, contract])

    assert len(rows) == 1
    assert rows[0].type_label_id == contract.id


# --------------------------------------------------------------------------- #
# write_tag_candidates atomicity (#54)
# --------------------------------------------------------------------------- #

def test_write_tag_candidates_survives_a_racing_duplicate_insert(db):
    """#54: the old SELECT-then-INSERT let two concurrent writers (e.g.
    overlapping suggest_labels/augment runs on the same file) both see "not
    yet present" and race to insert the same (file_id, kind_id, lower(value))
    row -- the loser's commit then crashed on the UNIQUE index. INSERT ...
    ON CONFLICT DO NOTHING makes this a structural non-issue, simulated here
    by re-running the write with overlapping output -- same convention as
    #53/#51."""
    file = _file(db)
    kind = TagKind(name="person")
    db.add(kind)
    db.commit()
    db.refresh(kind)

    output = TagValuesOutput(values=["Angela Merkel"])

    first = write_tag_candidates(db, file.id, kind, output)
    second = write_tag_candidates(db, file.id, kind, output)  # simulated racing writer

    assert len(first) == 1
    assert first[0].value == "Angela Merkel"
    assert second == []  # conflicting insert silently skipped -- no IntegrityError

    rows = list(db.scalars(select(TagLabel).where(TagLabel.file_id == file.id, TagLabel.kind_id == kind.id)))
    assert len(rows) == 1


def test_write_tag_candidates_survives_a_case_variant_race(db):
    """#49's case-insensitive index applies to the race too: a stale
    existing_values snapshot (e.g. augment's upfront query, taken before a
    concurrent writer's commit) must not defeat ON CONFLICT -- "berlin"
    racing against an already-committed "Berlin" is still a no-op, not a
    crash, even though the (deliberately stale) existing_values passed in
    here doesn't know about it."""
    file = _file(db)
    kind = TagKind(name="place")
    db.add(kind)
    db.commit()
    db.refresh(kind)

    write_tag_candidates(db, file.id, kind, TagValuesOutput(values=["Berlin"]))

    # existing_values intentionally stale/empty -- mirrors a caller whose
    # upfront snapshot predates the other writer's commit.
    rows = write_tag_candidates(db, file.id, kind, TagValuesOutput(values=["berlin"]), existing_values=set())

    assert rows == []
    all_rows = list(db.scalars(select(TagLabel).where(TagLabel.file_id == file.id, TagLabel.kind_id == kind.id)))
    assert len(all_rows) == 1
    assert all_rows[0].value == "Berlin"


def test_write_tag_candidates_survives_a_concurrent_manual_upsert(db):
    """#54: a user manually confirming a tag via upsert_user_tag_label
    (#50's atomic upsert) while an augment/initial job's write_tag_candidates
    is mid-flight for the same (file, kind, value) must not fail the job --
    whichever commits second just finds nothing left to insert."""
    file = _file(db)
    kind = TagKind(name="organization")
    db.add(kind)
    db.commit()
    db.refresh(kind)

    upsert_user_tag_label(db, file.id, kind.id, "Deutsche Bank")  # simulated concurrent manual add

    rows = write_tag_candidates(db, file.id, kind, TagValuesOutput(values=["Deutsche Bank"]))

    assert rows == []  # no IntegrityError -- the user's row already won
    all_rows = list(db.scalars(select(TagLabel).where(TagLabel.file_id == file.id, TagLabel.kind_id == kind.id)))
    assert len(all_rows) == 1
    assert all_rows[0].source == "user"
    assert all_rows[0].status == "confirmed"
