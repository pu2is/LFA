"""Tests for labeling.service.normalize_label_name, and for merge.write_type_
candidates' atomicity (#53).

merge.py's other write functions are covered elsewhere: write_initial_
candidates has no remaining caller (kept per #45's scope-out, not tested
further); select_kinds/write_tag_candidates are covered in
test_suggest_labels.py and test_suggest_labels_augment.py, since ADR-0001's
flows make them meaningful only in sequence, not in isolation.
"""
from datetime import datetime, timezone

from sqlalchemy import select

from app.modules.files.models import File, RegisteredPath
from app.modules.labeling.merge import write_type_candidates
from app.modules.labeling.models import TypeLabel, TypeLabelFile
from app.modules.labeling.prompts import InitialTypeCandidate, InitialTypeSuggestionOutput
from app.modules.labeling.service import normalize_label_name


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
