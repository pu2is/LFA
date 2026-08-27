"""Pure inventory/diff/matching engine for Global Rescan (WF1b, ADR-0001b
D2/D3). Produces an in-memory classification of the full filesystem
inventory against the current `files` manifest -- no database writes here;
applying the result to `files`/`file_events` is #67, fuzzy recovery for
whatever is left unmatched is #66.
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.modules.files.models import File, RegisteredPath
from app.modules.scans import discovery


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
class RescanDiff:
    unchanged: list[UnchangedFile]
    metadata_refreshed: list[MetadataRefreshed]
    modified: list[ModifiedFile]
    moved: list[MovedFile]
    moved_modified: list[MovedModifiedFile]
    missing: list[MissingFile]
    added: list[AddedFile]


def diff_inventory(inventory: list[InventoryFile], current_files: list[File]) -> RescanDiff:
    """Classify `inventory` against `current_files` per ADR-0001b D3's cost-
    ordered matching ladder (steps 1-4 and 6 -- fuzzy recovery, step 5, is
    #66). Hashing happens on demand via discovery.hash_with_retry, only for
    files whose cheap metadata changed or that are move candidates; an
    UnstableFileError propagates to fail the whole Rescan (D2).
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
