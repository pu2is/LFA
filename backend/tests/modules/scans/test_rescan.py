"""Tests for the Rescan inventory/diff/matching engine (WF1b, ADR-0001b D2/D3,
#65 deterministic matching + #66 fuzzy recovery)."""
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from app.modules.files.models import File, RegisteredPath
from app.modules.processing.extraction import ExtractionResult
from app.modules.scans import discovery, rescan
from app.modules.scans.text_signature import compute_text_signature


def _register(db, path: Path, parent: RegisteredPath | None = None) -> RegisteredPath:
    row = RegisteredPath(path=str(path.resolve()), parent_path_id=parent.id if parent else None)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _make_file(
    db,
    *,
    path_id,
    full_path: Path,
    file_hash: str,
    file_size: int,
    file_modified_at: datetime,
    file_created_at: datetime | None = None,
    fs_device_id: str | None = None,
    fs_file_id: str | None = None,
    text_signature: str | None = None,
) -> File:
    file = File(
        path_id=path_id,
        filename=full_path.name,
        full_path=str(full_path),
        file_type=full_path.suffix.lstrip("."),
        file_size=file_size,
        file_hash=file_hash,
        file_created_at=file_created_at,
        file_modified_at=file_modified_at,
        fs_device_id=fs_device_id,
        fs_file_id=fs_file_id,
        text_signature=text_signature,
    )
    db.add(file)
    db.commit()
    db.refresh(file)
    return file


def _inventory(registered_path: RegisteredPath) -> list[rescan.InventoryFile]:
    return rescan.build_inventory([registered_path])


def test_diff_inventory_classifies_unchanged_file(db, tmp_path):
    registered_path = _register(db, tmp_path)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"pdf-content")
    stat = file_path.stat()

    existing = _make_file(
        db, path_id=registered_path.id, full_path=file_path,
        file_hash=discovery.compute_sha256(file_path), file_size=stat.st_size,
        file_modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert [u.file.id for u in diff.unchanged] == [existing.id]
    assert not any([diff.metadata_refreshed, diff.modified, diff.moved, diff.moved_modified, diff.missing, diff.added])


def test_diff_inventory_classifies_metadata_refresh_when_hash_unchanged(db, tmp_path):
    registered_path = _register(db, tmp_path)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"pdf-content")

    existing = _make_file(
        db, path_id=registered_path.id, full_path=file_path,
        file_hash=discovery.compute_sha256(file_path), file_size=file_path.stat().st_size,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),  # stale mtime -> triggers cheap diff
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert [m.file.id for m in diff.metadata_refreshed] == [existing.id]
    assert not diff.unchanged and not diff.modified


def test_diff_inventory_classifies_modified_when_hash_changes(db, tmp_path):
    registered_path = _register(db, tmp_path)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"new-content-on-disk")

    existing = _make_file(
        db, path_id=registered_path.id, full_path=file_path,
        file_hash="stale-hash-from-before", file_size=1,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert len(diff.modified) == 1
    assert diff.modified[0].file.id == existing.id
    assert diff.modified[0].new_hash == discovery.compute_sha256(file_path)


def test_diff_inventory_classifies_moved_via_filesystem_identity(db, tmp_path):
    registered_path = _register(db, tmp_path)
    original_path = tmp_path / "old_name.pdf"
    original_path.write_bytes(b"pdf-content")
    file_hash = discovery.compute_sha256(original_path)
    old_size = original_path.stat().st_size

    new_path = tmp_path / "new_name.pdf"
    original_path.rename(new_path)  # simulates a move that already happened on disk

    [entry] = list(discovery.iter_inventory(tmp_path))

    existing = _make_file(
        db, path_id=registered_path.id, full_path=original_path,
        file_hash=file_hash, file_size=old_size,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        file_created_at=entry.file_created_at,
        fs_device_id=entry.fs_device_id, fs_file_id=entry.fs_file_id,
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert len(diff.moved) == 1
    assert diff.moved[0].file.id == existing.id
    assert diff.moved[0].match_method == "filesystem_id"
    assert diff.moved[0].inventory.full_path == str(new_path)
    assert not diff.missing and not diff.added and not diff.moved_modified


def test_diff_inventory_classifies_moved_modified_via_filesystem_identity(db, tmp_path):
    registered_path = _register(db, tmp_path)
    original_path = tmp_path / "old_name.pdf"
    original_path.write_bytes(b"original-content")
    old_hash = discovery.compute_sha256(original_path)
    old_size = original_path.stat().st_size

    new_path = tmp_path / "new_name.pdf"
    original_path.rename(new_path)
    new_path.write_bytes(b"changed-content-now-longer")  # same identity, new content

    [entry] = list(discovery.iter_inventory(tmp_path))

    existing = _make_file(
        db, path_id=registered_path.id, full_path=original_path,
        file_hash=old_hash, file_size=old_size,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        file_created_at=entry.file_created_at,
        fs_device_id=entry.fs_device_id, fs_file_id=entry.fs_file_id,
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert len(diff.moved_modified) == 1
    assert diff.moved_modified[0].file.id == existing.id
    assert diff.moved_modified[0].match_method == "filesystem_id"
    assert diff.moved_modified[0].new_hash == discovery.compute_sha256(new_path)
    assert not diff.moved


def test_diff_inventory_classifies_moved_via_hash_when_identity_unavailable(db, tmp_path):
    """A file never identity-tracked by an earlier scan (fs_device_id/
    fs_file_id/file_created_at all NULL) must still resolve via priority 2's
    exact-hash matching (ADR-0001b D3 step 4)."""
    registered_path = _register(db, tmp_path)
    original_path = tmp_path / "old_name.pdf"
    original_path.write_bytes(b"pdf-content")
    file_hash = discovery.compute_sha256(original_path)
    old_size = original_path.stat().st_size

    new_path = tmp_path / "new_name.pdf"
    original_path.rename(new_path)

    existing = _make_file(
        db, path_id=registered_path.id, full_path=original_path,
        file_hash=file_hash, file_size=old_size,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert len(diff.moved) == 1
    assert diff.moved[0].match_method == "hash"
    assert not diff.moved_modified


def test_diff_inventory_classifies_missing_and_added_independently(db, tmp_path):
    registered_path = _register(db, tmp_path)
    (tmp_path / "new_file.pdf").write_bytes(b"brand-new-content")

    existing = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "deleted.pdf",
        file_hash="deleted-file-hash", file_size=10,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert [m.file.id for m in diff.missing] == [existing.id]
    assert len(diff.added) == 1
    assert diff.added[0].inventory.filename == "new_file.pdf"
    assert diff.added[0].file_hash == discovery.compute_sha256(tmp_path / "new_file.pdf")
    assert not diff.moved and not diff.moved_modified


def test_diff_inventory_prioritizes_identity_match_before_attempting_hash_match(db, tmp_path):
    """Regression test: two missing/added pairs share the same content hash.
    If hash-matching ran before (or instead of) identity-matching, both pairs
    would look ambiguous (2 missing vs. 2 added for that hash) and neither
    would resolve. Identity-matching must claim its pair first, shrinking the
    hash-matching pool to 1-vs-1 so the remaining pair can still resolve."""
    registered_path = _register(db, tmp_path)

    old_a = tmp_path / "old_a.pdf"
    old_a.write_bytes(b"shared-content")
    shared_hash = discovery.compute_sha256(old_a)
    new_a = tmp_path / "new_a.pdf"
    old_a.rename(new_a)  # real rename -- identity is tracked and preserved

    new_b = tmp_path / "new_b.pdf"
    new_b.write_bytes(b"shared-content")  # identical content, unrelated file

    [entry_a] = [e for e in discovery.iter_inventory(tmp_path) if e.path.name == "new_a.pdf"]

    file_a = _make_file(
        db, path_id=registered_path.id, full_path=old_a,
        file_hash=shared_hash, file_size=entry_a.file_size,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        file_created_at=entry_a.file_created_at,
        fs_device_id=entry_a.fs_device_id, fs_file_id=entry_a.fs_file_id,
    )
    file_b = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "old_b.pdf",
        file_hash=shared_hash, file_size=len(b"shared-content"),
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        # never identity-tracked (e.g. predates #64) -- only hash can match it
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [file_a, file_b])

    assert {(m.file.id, m.match_method) for m in diff.moved} == {
        (file_a.id, "filesystem_id"),
        (file_b.id, "hash"),
    }
    assert not diff.missing and not diff.added


def test_diff_inventory_does_not_infer_moved_modified_when_creation_time_mismatches(db, tmp_path):
    """Regression test: same filesystem identity, but creation time disagrees
    and content changed -- ADR-0001b D3 forbids auto-claiming moved_modified
    from identity alone. With no exact-hash match available either, this must
    conservatively fall back to missing + added, not invent a pairing."""
    registered_path = _register(db, tmp_path)
    original_path = tmp_path / "old_name.pdf"
    original_path.write_bytes(b"original-content")
    old_hash = discovery.compute_sha256(original_path)
    old_size = original_path.stat().st_size

    new_path = tmp_path / "new_name.pdf"
    original_path.rename(new_path)
    new_path.write_bytes(b"different-content-now")  # same identity, changed content

    [entry] = list(discovery.iter_inventory(tmp_path))
    assert entry.file_created_at is not None  # sanity: this platform tracks creation time

    mismatched_created_at = entry.file_created_at.replace(year=entry.file_created_at.year - 1)

    existing = _make_file(
        db, path_id=registered_path.id, full_path=original_path,
        file_hash=old_hash, file_size=old_size,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        file_created_at=mismatched_created_at,
        fs_device_id=entry.fs_device_id, fs_file_id=entry.fs_file_id,
    )

    diff = rescan.diff_inventory(_inventory(registered_path), [existing])

    assert not diff.moved_modified and not diff.moved
    assert [m.file.id for m in diff.missing] == [existing.id]
    assert len(diff.added) == 1


def test_build_inventory_prunes_registered_child_subtree(db, tmp_path):
    parent_dir = tmp_path / "parent"
    child_dir = parent_dir / "child"
    child_dir.mkdir(parents=True)
    (parent_dir / "top.pdf").write_bytes(b"pdf-content")
    (child_dir / "nested.pdf").write_bytes(b"pdf-content")

    parent = _register(db, parent_dir)
    child = _register(db, child_dir, parent=parent)

    inventory = rescan.build_inventory([parent, child])

    by_path_id: dict = {}
    for entry in inventory:
        by_path_id.setdefault(entry.path_id, []).append(entry.filename)

    assert by_path_id[parent.id] == ["top.pdf"]
    assert by_path_id[child.id] == ["nested.pdf"]


def test_build_inventory_raises_on_any_unreadable_root(db, tmp_path):
    good_dir = tmp_path / "good"
    good_dir.mkdir()
    (good_dir / "a.pdf").write_bytes(b"pdf-content")
    bad_dir = tmp_path / "bad"  # never created on disk

    good_path = _register(db, good_dir)
    bad_path = _register(db, bad_dir)

    with pytest.raises(OSError):
        rescan.build_inventory([good_path, bad_path])


# --- Fuzzy recovery (#66, ADR-0001b D3 step 5 / D5) ---------------------
#
# All comparisons in these tests are against _TINY_EDIT_TEXT (what the mocked
# extraction returns for the added file), and similarity scores below are
# measured relative to it, not to _ORIGINAL_TEXT -- SimHash similarity isn't
# transitive, so this matters. Chosen so their pairwise similarity to
# _TINY_EDIT_TEXT spreads out enough to exercise threshold and margin
# decisions deterministically (see text_signature.SIMILARITY_THRESHOLD=0.90,
# UNIQUENESS_MARGIN=0.05):
#   _ORIGINAL_TEXT             ~0.953 (single number changed, in reverse)
#   _CLOSE_SECOND_EDIT_TEXT    ~0.938 (a different single phrase changed)
#   _TWO_SENTENCE_EDIT_TEXT    ~0.797 (two sentences rewritten)
#   _REWRITTEN_TEXT            ~0.5   (unrelated content)
_ORIGINAL_TEXT = (
    "Invoice Number 2024-001. This invoice covers consulting services rendered\n"
    "during the month of January 2024 for the engineering department. Services included\n"
    "system architecture review, code quality audits, and mentoring of junior developers.\n"
    "The total amount due for these services is 1500 EUR, payable within thirty days of\n"
    "the invoice date. Please remit payment to the account listed below."
)
_TINY_EDIT_TEXT = _ORIGINAL_TEXT.replace("1500 EUR", "1550 EUR")
_CLOSE_SECOND_EDIT_TEXT = _ORIGINAL_TEXT.replace("code quality audits", "code review sessions")
_ONE_SENTENCE_EDIT_TEXT = _ORIGINAL_TEXT.replace(
    "Please remit payment to the account listed below.",
    "Kindly wire the funds to our updated banking details as soon as possible, thank you very much indeed.",
)
_TWO_SENTENCE_EDIT_TEXT = _ONE_SENTENCE_EDIT_TEXT.replace(
    "Services included\nsystem architecture review, code quality audits, and mentoring of junior developers.",
    "Work performed spanned infrastructure planning, security assessments, and onboarding support for new hires.",
)
_REWRITTEN_TEXT = (
    "A completely different summary of unrelated project work delivered in early 2024 for a "
    "separate client engagement, covering topics such as data migration and vendor negotiation, "
    "follow-up sessions scheduled for next quarter as needed."
)


def _mock_extract_text(text: str):
    return patch(
        "app.modules.scans.rescan.extraction.extract_text",
        return_value=ExtractionResult(text=text, ocr_applied=False),
    )


def test_diff_inventory_creates_fuzzy_candidate_for_unique_above_threshold_match(db, tmp_path):
    registered_path = _register(db, tmp_path)
    added_path = tmp_path / "new_name.pdf"
    added_path.write_bytes(b"x" * 1000)

    missing = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "old_name.pdf",
        file_hash="old-hash", file_size=1000,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text_signature=compute_text_signature(_ORIGINAL_TEXT),
    )

    with _mock_extract_text(_TINY_EDIT_TEXT):
        diff = rescan.diff_inventory(_inventory(registered_path), [missing])

    assert len(diff.fuzzy_candidates) == 1
    candidate = diff.fuzzy_candidates[0]
    assert candidate.missing_file.id == missing.id
    assert candidate.inventory.full_path == str(added_path)
    assert candidate.similarity_score == pytest.approx(0.953125)
    assert not diff.added  # claimed by the fuzzy candidate, not left as `added`
    assert [m.file.id for m in diff.missing] == [missing.id]  # stays missing while pending (D5)


def test_diff_inventory_no_fuzzy_candidate_when_metadata_narrowing_finds_nothing(db, tmp_path):
    """Wrong file_type is enough to exclude a metadata candidate, regardless
    of text similarity -- extraction should never even run."""
    registered_path = _register(db, tmp_path)
    added_path = tmp_path / "new_name.pdf"
    added_path.write_bytes(b"x" * 1000)

    missing = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "old_name.docx",
        file_hash="old-hash", file_size=1000,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text_signature=compute_text_signature(_ORIGINAL_TEXT),
    )

    with _mock_extract_text(_TINY_EDIT_TEXT) as mock_extract:
        diff = rescan.diff_inventory(_inventory(registered_path), [missing])
        mock_extract.assert_not_called()

    assert not diff.fuzzy_candidates
    assert len(diff.added) == 1
    assert [m.file.id for m in diff.missing] == [missing.id]


def test_diff_inventory_no_fuzzy_candidate_when_below_similarity_threshold(db, tmp_path):
    registered_path = _register(db, tmp_path)
    added_path = tmp_path / "new_name.pdf"
    added_path.write_bytes(b"x" * 1000)

    missing = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "old_name.pdf",
        file_hash="old-hash", file_size=1000,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text_signature=compute_text_signature(_ORIGINAL_TEXT),
    )

    with _mock_extract_text(_REWRITTEN_TEXT):
        diff = rescan.diff_inventory(_inventory(registered_path), [missing])

    assert not diff.fuzzy_candidates
    assert len(diff.added) == 1
    assert [m.file.id for m in diff.missing] == [missing.id]


def test_diff_inventory_no_fuzzy_candidate_when_no_extractable_text(db, tmp_path):
    registered_path = _register(db, tmp_path)
    added_path = tmp_path / "new_name.pdf"
    added_path.write_bytes(b"x" * 1000)

    missing = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "old_name.pdf",
        file_hash="old-hash", file_size=1000,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text_signature=compute_text_signature(_ORIGINAL_TEXT),
    )

    with patch(
        "app.modules.scans.rescan.extraction.extract_text",
        side_effect=RuntimeError("no text layer and OCR disabled"),
    ):
        diff = rescan.diff_inventory(_inventory(registered_path), [missing])

    assert not diff.fuzzy_candidates
    assert len(diff.added) == 1


def test_diff_inventory_creates_fuzzy_candidate_when_best_match_clears_uniqueness_margin(db, tmp_path):
    """Regression test (uniqueness margin, clear-winner side): metadata
    narrowing finds two missing candidates, but one is a much closer text
    match than the other -- the winner must clear the runner-up by at least
    UNIQUENESS_MARGIN, which it does here (~0.953 vs ~0.813)."""
    registered_path = _register(db, tmp_path)
    added_path = tmp_path / "new_name.pdf"
    added_path.write_bytes(b"x" * 1000)

    close_match = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "close.pdf",
        file_hash="close-hash", file_size=1000,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text_signature=compute_text_signature(_ORIGINAL_TEXT),
    )
    far_match = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "far.pdf",
        file_hash="far-hash", file_size=1000,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text_signature=compute_text_signature(_TWO_SENTENCE_EDIT_TEXT),
    )

    with _mock_extract_text(_TINY_EDIT_TEXT):
        diff = rescan.diff_inventory(_inventory(registered_path), [close_match, far_match])

    assert len(diff.fuzzy_candidates) == 1
    assert diff.fuzzy_candidates[0].missing_file.id == close_match.id
    assert not diff.added
    assert {m.file.id for m in diff.missing} == {close_match.id, far_match.id}


def test_diff_inventory_no_fuzzy_candidate_when_top_two_matches_are_within_uniqueness_margin(db, tmp_path):
    """Regression test (uniqueness margin, ambiguous side): two missing
    candidates both score above SIMILARITY_THRESHOLD, but too close to each
    other (~0.953 vs ~0.938, under UNIQUENESS_MARGIN=0.05 apart) to trust
    either one -- must produce no candidate rather than guess."""
    registered_path = _register(db, tmp_path)
    added_path = tmp_path / "new_name.pdf"
    added_path.write_bytes(b"x" * 1000)

    candidate_a = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "a.pdf",
        file_hash="a-hash", file_size=1000,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text_signature=compute_text_signature(_ORIGINAL_TEXT),
    )
    candidate_b = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "b.pdf",
        file_hash="b-hash", file_size=1000,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text_signature=compute_text_signature(_CLOSE_SECOND_EDIT_TEXT),
    )

    with _mock_extract_text(_TINY_EDIT_TEXT):
        diff = rescan.diff_inventory(_inventory(registered_path), [candidate_a, candidate_b])

    assert not diff.fuzzy_candidates
    assert len(diff.added) == 1
    assert {m.file.id for m in diff.missing} == {candidate_a.id, candidate_b.id}


def test_diff_inventory_no_fuzzy_candidate_when_missing_file_has_no_text_signature(db, tmp_path):
    """Bootstrap/legacy case (issue #66 scope): a missing file ingested
    before this feature has text_signature=NULL and must be silently
    excluded from comparison, not treated as a match or a crash."""
    registered_path = _register(db, tmp_path)
    added_path = tmp_path / "new_name.pdf"
    added_path.write_bytes(b"x" * 1000)

    missing = _make_file(
        db, path_id=registered_path.id, full_path=tmp_path / "old_name.pdf",
        file_hash="old-hash", file_size=1000,
        file_modified_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        text_signature=None,
    )

    with _mock_extract_text(_TINY_EDIT_TEXT):
        diff = rescan.diff_inventory(_inventory(registered_path), [missing])

    assert not diff.fuzzy_candidates
    assert len(diff.added) == 1
