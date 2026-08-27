"""Inventory/diff/matching engine plus apply for Global Rescan (WF1b,
ADR-0001b D2/D3/D6). build_inventory/diff_inventory produce an in-memory
classification of the full filesystem inventory against the current `files`
manifest with no database writes; apply_diff is the only function here that
writes to the database, applying that classification to
`files`/`file_events`/`file_match_candidates` in one caller-managed
transaction.
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.modules.files.models import File, RegisteredPath
from app.modules.jobs.models import Job
from app.modules.labeling.service import get_files_with_type_or_tag_labels
from app.modules.processing import cleaning, extraction
from app.modules.scans import discovery, text_signature
from app.modules.scans.models import FileEvent, FileMatchCandidate

# ADR-0001b D3 step 5, metadata narrowing: how far a candidate's file_size may
# be from the added file's size before it is cheap-filtered out, pre-text-
# extraction. Deliberately generous -- this only bounds how many candidates
# get their text extracted/compared, not the actual match decision (that is
# text_signature.SIMILARITY_THRESHOLD/UNIQUENESS_MARGIN).
MIN_SIZE_RATIO = 0.5
MAX_SIZE_RATIO = 2.0


@dataclass(frozen=True)
class InventoryFile:
    """One discovery.InventoryEntry tagged with the registered root it was
    found under, so a later apply step knows which path_id to write."""

    path_id: uuid.UUID
    path: Path
    filename: str
    file_type: str
    file_size: int
    file_created_at: datetime | None
    file_modified_at: datetime
    fs_device_id: str
    fs_file_id: str

    @property
    def full_path(self) -> str:
        return str(self.path)


def build_inventory(registered_paths: list[RegisteredPath]) -> list[InventoryFile]:
    """Walk every registered root (ADR-0001b D2). `registered_paths` is the
    path set the caller fixed once at worker start (D1) -- nested-path
    pruning excludes each root's own direct registered children (computed
    here from that same fixed set, not a fresh per-root query) from its
    walk, since they are inventoried separately as their own root; this
    keeps every file inventoried exactly once.

    Propagates OSError from discovery.iter_inventory on any missing/
    unreadable root or walk error -- the caller must not apply a partial
    inventory (D2's "zero side effects" root-failure rule).
    """
    inventory: list[InventoryFile] = []
    for root in registered_paths:
        children = frozenset(
            Path(p.path) for p in registered_paths if p.parent_path_id == root.id
        )
        for entry in discovery.iter_inventory(Path(root.path), exclude_roots=children):
            inventory.append(InventoryFile(
                path_id=root.id,
                path=entry.path,
                filename=entry.path.name,
                file_type=entry.file_type,
                file_size=entry.file_size,
                file_created_at=entry.file_created_at,
                file_modified_at=entry.file_modified_at,
                fs_device_id=entry.fs_device_id,
                fs_file_id=entry.fs_file_id,
            ))
    return inventory


@dataclass(frozen=True)
class UnchangedFile:
    file: File
    inventory: InventoryFile


@dataclass(frozen=True)
class MetadataRefreshed:
    """Size/mtime changed but content hash didn't (e.g. a touch) -- metadata
    gets refreshed, but this is not a semantic change: no file_events row."""

    file: File
    inventory: InventoryFile


@dataclass(frozen=True)
class ModifiedFile:
    file: File
    inventory: InventoryFile
    new_hash: str
    match_method: str = "path"


@dataclass(frozen=True)
class MovedFile:
    file: File
    inventory: InventoryFile
    match_method: str  # "filesystem_id" | "hash"


@dataclass(frozen=True)
class MovedModifiedFile:
    """Only reachable via filesystem-identity matching (ADR-0001b D3): exact-
    hash matching by construction means identical content, i.e. `moved`."""

    file: File
    inventory: InventoryFile
    new_hash: str
    match_method: str = "filesystem_id"


@dataclass(frozen=True)
class MissingFile:
    file: File


@dataclass(frozen=True)
class AddedFile:
    inventory: InventoryFile
    file_hash: str


@dataclass(frozen=True)
class FuzzyCandidate:
    """Pending recovery proposal (ADR-0001b D5) -- shaped after
    file_match_candidates' columns, minus the ones only Apply (#67) can
    assign (id, scan_id, status, timestamps). Never auto-applied."""

    missing_file: File
    inventory: InventoryFile
    candidate_hash: str
    similarity_score: float


@dataclass(frozen=True)
class RescanDiff:
    unchanged: list[UnchangedFile]
    metadata_refreshed: list[MetadataRefreshed]
    modified: list[ModifiedFile]
    moved: list[MovedFile]
    moved_modified: list[MovedModifiedFile]
    missing: list[MissingFile]
    added: list[AddedFile]
    fuzzy_candidates: list[FuzzyCandidate]


def diff_inventory(inventory: list[InventoryFile], current_files: list[File]) -> RescanDiff:
    """Classify `inventory` against `current_files` per ADR-0001b D3's cost-
    ordered matching ladder (steps 1-6). Hashing happens on demand via
    discovery.hash_with_retry, only for files whose cheap metadata changed or
    that are move candidates; an UnstableFileError propagates to fail the
    whole Rescan (D2).
    """
    inventory_by_path = {entry.full_path: entry for entry in inventory}
    current_by_path = {file.full_path: file for file in current_files}

    unchanged: list[UnchangedFile] = []
    metadata_refreshed: list[MetadataRefreshed] = []
    modified: list[ModifiedFile] = []

    for full_path, entry in inventory_by_path.items():
        existing = current_by_path.get(full_path)
        if existing is None:
            continue
        if (existing.file_size, existing.file_modified_at) == (entry.file_size, entry.file_modified_at):
            unchanged.append(UnchangedFile(file=existing, inventory=entry))
            continue
        new_hash = discovery.hash_with_retry(entry.path)
        if new_hash == existing.file_hash:
            metadata_refreshed.append(MetadataRefreshed(file=existing, inventory=entry))
        else:
            modified.append(ModifiedFile(file=existing, inventory=entry, new_hash=new_hash))

    missing_pool: dict[uuid.UUID, File] = {
        file.id: file for path, file in current_by_path.items() if path not in inventory_by_path
    }
    added_pool: dict[str, InventoryFile] = {
        path: entry for path, entry in inventory_by_path.items() if path not in current_by_path
    }
    # Every surviving added candidate needs its hash sooner or later --
    # `added` requires it for the not-null files.file_hash column, and both
    # move-matching steps below need it to compare against a missing file's
    # hash -- so hash each one exactly once, up front, and share the result.
    added_hashes = {path: discovery.hash_with_retry(entry.path) for path, entry in added_pool.items()}

    moved: list[MovedFile] = []
    moved_modified: list[MovedModifiedFile] = []
    _match_by_identity(missing_pool, added_pool, added_hashes, moved, moved_modified)
    _match_by_hash(missing_pool, added_pool, added_hashes, moved)

    # Step 5: fuzzy recovery. Pops matched entries from added_pool (no file
    # row is created for a pending candidate -- D5), but never from
    # missing_pool: the missing file stays `missing` while pending, and may
    # be the target of more than one candidate.
    fuzzy_candidates = _match_fuzzy_candidates(missing_pool, added_pool, added_hashes)

    missing = [MissingFile(file=file) for file in missing_pool.values()]
    added = [
        AddedFile(inventory=entry, file_hash=added_hashes[path])
        for path, entry in added_pool.items()
    ]

    return RescanDiff(
        unchanged=unchanged,
        metadata_refreshed=metadata_refreshed,
        modified=modified,
        moved=moved,
        moved_modified=moved_modified,
        fuzzy_candidates=fuzzy_candidates,
        missing=missing,
        added=added,
    )


def _match_by_identity(
    missing_pool: dict[uuid.UUID, File],
    added_pool: dict[str, InventoryFile],
    added_hashes: dict[str, str],
    moved: list[MovedFile],
    moved_modified: list[MovedModifiedFile],
) -> None:
    """ADR-0001b D3 step 3: unique 1:1 pairing on (fs_device_id, fs_file_id,
    file_created_at) -- bundling creation time into the grouping key means a
    missing/added pair only lands in the same group when identity AND
    creation time agree, so a null or mismatched creation time can never
    produce a moved_modified match here (it simply groups elsewhere alone).
    """
    missing_groups: dict[tuple[str, str, datetime], list[uuid.UUID]] = defaultdict(list)
    for file_id, file in missing_pool.items():
        if file.fs_device_id and file.fs_file_id and file.file_created_at is not None:
            missing_groups[(file.fs_device_id, file.fs_file_id, file.file_created_at)].append(file_id)

    added_groups: dict[tuple[str, str, datetime], list[str]] = defaultdict(list)
    for path, entry in added_pool.items():
        if entry.file_created_at is not None:
            added_groups[(entry.fs_device_id, entry.fs_file_id, entry.file_created_at)].append(path)

    for identity_key, missing_ids in missing_groups.items():
        added_paths = added_groups.get(identity_key, [])
        if len(missing_ids) != 1 or len(added_paths) != 1:
            continue  # ambiguous (or no counterpart) -- leave for the next ladder step

        file = missing_pool.pop(missing_ids[0])
        path = added_paths[0]
        entry = added_pool.pop(path)
        new_hash = added_hashes[path]
        if new_hash == file.file_hash:
            moved.append(MovedFile(file=file, inventory=entry, match_method="filesystem_id"))
        else:
            moved_modified.append(MovedModifiedFile(file=file, inventory=entry, new_hash=new_hash))


def _match_by_hash(
    missing_pool: dict[uuid.UUID, File],
    added_pool: dict[str, InventoryFile],
    added_hashes: dict[str, str],
    moved: list[MovedFile],
) -> None:
    """ADR-0001b D3 step 4: unique 1:1 pairing on exact SHA-256, for whatever
    priority 1 (identity) couldn't resolve. Duplicate hashes on either side
    are ambiguity, not a match -- left as missing/added, never auto-merged.
    """
    missing_by_hash: dict[str, list[uuid.UUID]] = defaultdict(list)
    for file_id, file in missing_pool.items():
        missing_by_hash[file.file_hash].append(file_id)

    added_by_hash: dict[str, list[str]] = defaultdict(list)
    for path, entry in added_pool.items():
        added_by_hash[added_hashes[path]].append(path)

    for file_hash, missing_ids in missing_by_hash.items():
        added_paths = added_by_hash.get(file_hash, [])
        if len(missing_ids) != 1 or len(added_paths) != 1:
            continue

        file = missing_pool.pop(missing_ids[0])
        path = added_paths[0]
        entry = added_pool.pop(path)
        moved.append(MovedFile(file=file, inventory=entry, match_method="hash"))


def _size_compatible(candidate_size: int, reference_size: int) -> bool:
    """ADR-0001b D3 step 5 metadata narrowing: is `candidate_size` within
    [MIN_SIZE_RATIO, MAX_SIZE_RATIO] of `reference_size`? A zero-byte
    reference only matches other zero-byte files (the ratio is undefined)."""
    if reference_size == 0:
        return candidate_size == 0
    ratio = candidate_size / reference_size
    return MIN_SIZE_RATIO <= ratio <= MAX_SIZE_RATIO


def _match_fuzzy_candidates(
    missing_pool: dict[uuid.UUID, File],
    added_pool: dict[str, InventoryFile],
    added_hashes: dict[str, str],
) -> list[FuzzyCandidate]:
    """ADR-0001b D3 step 5 / D5: for each still-unmatched added file, narrow
    missing_pool by file_type + size range, then compare normalized-text
    SimHash only against that narrowed set -- never against the full
    missing_pool, and never via OCR/embedding/LLM (D3).

    A single narrowed candidate only needs to clear SIMILARITY_THRESHOLD.
    Multiple narrowed candidates additionally need the top score to beat the
    runner-up by UNIQUENESS_MARGIN, or the match is ambiguous and dropped.
    Matched added entries are popped from added_pool (no file row is created
    for a pending candidate -- D5); missing_pool is left untouched, since the
    missing file stays `missing` while the candidate is pending and may end
    up targeted by more than one candidate.
    """
    candidates: list[FuzzyCandidate] = []

    for path, entry in list(added_pool.items()):
        metadata_matches = [
            file
            for file in missing_pool.values()
            if file.file_type == entry.file_type and _size_compatible(entry.file_size, file.file_size)
        ]
        if not metadata_matches:
            continue

        try:
            result = extraction.extract_text(entry.path, entry.file_type, use_ocr=False)
        except Exception:
            # No text layer, OCR disabled, unsupported type, or a loader
            # choking on malformed content -- all are "no usable text" here
            # (D3: stop, don't escalate), same broad catch run_ingest uses
            # around this same call for the same reason.
            continue
        added_signature = text_signature.compute_text_signature(cleaning.clean(result.text))
        if added_signature is None:
            continue

        scored = sorted(
            (
                (file, text_signature.similarity(added_signature, file.text_signature))
                for file in metadata_matches
                if file.text_signature is not None
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        if not scored:
            continue

        best_file, best_score = scored[0]
        if best_score < text_signature.SIMILARITY_THRESHOLD:
            continue
        if len(scored) > 1 and (best_score - scored[1][1]) < text_signature.UNIQUENESS_MARGIN:
            continue  # ambiguous: metadata narrowed to >1, and text didn't clearly pick a winner

        added_pool.pop(path)
        candidates.append(FuzzyCandidate(
            missing_file=best_file,
            inventory=entry,
            candidate_hash=added_hashes[path],
            similarity_score=best_score,
        ))

    return candidates


def apply_diff(
    db: Session,
    scan_job: Job,
    diff: RescanDiff,
    registered_paths: list[RegisteredPath],
) -> None:
    """Apply a RescanDiff to the database (ADR-0001b D6): `files` mutations,
    `file_events`, `file_match_candidates`, child ingest job rows, and every
    registered path's `last_scanned_at`, all through this one call. Does not
    commit and does not touch `scan_job.stage` -- the caller
    (service.run_rescan) commits once so the whole apply is atomic, and owns
    job.stage transitions itself.

    `file_events`/`file_match_candidates` are written via INSERT ... ON
    CONFLICT DO NOTHING against their (scan_id, ...) unique constraints, so
    calling this twice for the same scan_id can never double-write an event
    or candidate -- even though the documented recovery path for an apply
    failure is a fresh Rescan (D6), not retrying this same call.
    """
    now = datetime.now(timezone.utc)

    review_candidates = [item.file for item in diff.modified] + [item.file for item in diff.moved_modified]
    needs_review = get_files_with_type_or_tag_labels(db, [file.id for file in review_candidates])

    for item in diff.unchanged:
        _refresh_metadata(item.file, item.inventory)
        _apply_recovery_if_missing(db, scan_job, item.file, item.inventory)

    for item in diff.metadata_refreshed:
        _refresh_metadata(item.file, item.inventory)
        _apply_recovery_if_missing(db, scan_job, item.file, item.inventory)

    for item in diff.modified:
        file = item.file
        from_hash = file.file_hash
        _refresh_metadata(file, item.inventory)
        file.file_hash = item.new_hash
        file.status = "discovered"
        file.embedding_status = "pending"
        if file.id in needs_review:
            file.labels_need_review = True
        _write_event(
            db, scan_id=scan_job.id, file_id=file.id, event_type="modified",
            from_path=item.inventory.full_path, to_path=item.inventory.full_path,
            from_hash=from_hash, to_hash=item.new_hash, match_method=item.match_method,
        )
        db.add(Job(type="ingest", file_id=file.id, parent_job_id=scan_job.id, trigger="scan"))

    for item in diff.moved:
        file = item.file
        from_path = file.full_path
        was_missing = file.status == "missing"
        _refresh_metadata(file, item.inventory)
        if was_missing:
            file.status = "discovered"
            file.embedding_status = "pending"
        _write_event(
            db, scan_id=scan_job.id, file_id=file.id, event_type="moved",
            from_path=from_path, to_path=item.inventory.full_path,
            from_hash=file.file_hash, to_hash=file.file_hash, match_method=item.match_method,
        )
        if was_missing:
            db.add(Job(type="ingest", file_id=file.id, parent_job_id=scan_job.id, trigger="scan"))

    for item in diff.moved_modified:
        file = item.file
        from_path = file.full_path
        from_hash = file.file_hash
        _refresh_metadata(file, item.inventory)
        file.file_hash = item.new_hash
        file.status = "discovered"
        file.embedding_status = "pending"
        if file.id in needs_review:
            file.labels_need_review = True
        _write_event(
            db, scan_id=scan_job.id, file_id=file.id, event_type="moved_modified",
            from_path=from_path, to_path=item.inventory.full_path,
            from_hash=from_hash, to_hash=item.new_hash, match_method=item.match_method,
        )
        db.add(Job(type="ingest", file_id=file.id, parent_job_id=scan_job.id, trigger="scan"))

    for item in diff.missing:
        file = item.file
        if file.status == "missing":
            continue  # already recorded on a prior Rescan -- staying missing isn't a new change
        from_path = file.full_path
        from_hash = file.file_hash
        file.status = "missing"
        _write_event(
            db, scan_id=scan_job.id, file_id=file.id, event_type="missing",
            from_path=from_path, to_path=None, from_hash=from_hash, to_hash=None, match_method=None,
        )

    for item in diff.added:
        file_id = uuid.uuid4()
        db.add(File(
            id=file_id,
            path_id=item.inventory.path_id,
            filename=item.inventory.filename,
            full_path=item.inventory.full_path,
            file_type=item.inventory.file_type,
            file_size=item.inventory.file_size,
            file_hash=item.file_hash,
            fs_device_id=item.inventory.fs_device_id,
            fs_file_id=item.inventory.fs_file_id,
            file_created_at=item.inventory.file_created_at,
            file_modified_at=item.inventory.file_modified_at,
        ))
        _write_event(
            db, scan_id=scan_job.id, file_id=file_id, event_type="added",
            from_path=None, to_path=item.inventory.full_path,
            from_hash=None, to_hash=item.file_hash, match_method=None,
        )
        db.add(Job(type="ingest", file_id=file_id, parent_job_id=scan_job.id, trigger="scan"))

    for candidate in diff.fuzzy_candidates:
        stmt = (
            pg_insert(FileMatchCandidate)
            .values(
                id=uuid.uuid4(),
                scan_id=scan_job.id,
                missing_file_id=candidate.missing_file.id,
                candidate_path_id=candidate.inventory.path_id,
                candidate_full_path=candidate.inventory.full_path,
                candidate_hash=candidate.candidate_hash,
                candidate_size=candidate.inventory.file_size,
                candidate_modified_at=candidate.inventory.file_modified_at,
                similarity_score=candidate.similarity_score,
            )
            .on_conflict_do_nothing(index_elements=[
                FileMatchCandidate.scan_id,
                FileMatchCandidate.missing_file_id,
                FileMatchCandidate.candidate_full_path,
            ])
        )
        db.execute(stmt)

    for path in registered_paths:
        path.last_scanned_at = now


def _refresh_metadata(file: File, inventory: InventoryFile) -> None:
    """Refresh the cheap metadata every matched category shares, including
    fs_device_id/fs_file_id -- ADR-0001b D3's identity matching only starts
    working once these are backfilled from a live inventory (see #65's
    bootstrap note in docs/workflow/01d-path-rescan.md), so every category
    that reuses an existing file row refreshes them, not just the ones with
    another reason to write.
    """
    file.path_id = inventory.path_id
    file.full_path = inventory.full_path
    file.filename = inventory.filename
    file.file_type = inventory.file_type
    file.file_size = inventory.file_size
    file.file_created_at = inventory.file_created_at
    file.file_modified_at = inventory.file_modified_at
    file.fs_device_id = inventory.fs_device_id
    file.fs_file_id = inventory.fs_file_id


def _apply_recovery_if_missing(db: Session, scan_job: Job, file: File, inventory: InventoryFile) -> None:
    """unchanged/metadata_refreshed normally write nothing beyond the metadata
    refresh -- but if `file` was status=missing, disk truth just proved this
    is the same file (same path, hash confirmed unchanged) via the same
    hash-backed evidence the rest of D3 relies on elsewhere, so it un-misses
    it the same way `modified`/`moved` already do for their own cases.
    """
    if file.status != "missing":
        return
    file.status = "discovered"
    file.embedding_status = "pending"
    _write_event(
        db, scan_id=scan_job.id, file_id=file.id, event_type="recovered",
        from_path=inventory.full_path, to_path=inventory.full_path,
        from_hash=file.file_hash, to_hash=file.file_hash, match_method="path",
    )
    db.add(Job(type="ingest", file_id=file.id, parent_job_id=scan_job.id, trigger="scan"))


def _write_event(
    db: Session,
    *,
    scan_id: uuid.UUID,
    file_id: uuid.UUID,
    event_type: str,
    from_path: str | None,
    to_path: str | None,
    from_hash: str | None,
    to_hash: str | None,
    match_method: str | None,
) -> None:
    stmt = (
        pg_insert(FileEvent)
        .values(
            id=uuid.uuid4(), scan_id=scan_id, file_id=file_id, event_type=event_type,
            from_path=from_path, to_path=to_path, from_hash=from_hash, to_hash=to_hash,
            match_method=match_method,
        )
        .on_conflict_do_nothing(index_elements=[FileEvent.scan_id, FileEvent.file_id, FileEvent.event_type])
    )
    db.execute(stmt)
