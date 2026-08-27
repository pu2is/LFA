"""Filesystem-only helpers for finding documents under a registered path.

No database access here on purpose: everything in this file is pure and can
be unit-tested without a Session, leaving scans/service.py to deal with the
Scan/File lifecycle.
"""

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Word leaves a temporary "~$<name>.docx" lock file next to an open document;
# it must never be treated as a real document.
LOCK_FILE_PREFIX = "~$"

# Static product config (like labeling/presets.py): adding a new format is a
# code change, not a per-environment setting -- scans should behave the same
# on every machine.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".doc"})

# Stream hashing in chunks so multi-hundred-MB PDFs don't load fully into memory.
HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class DiscoveredFile:
    path: Path
    file_type: str
    file_size: int
    file_hash: str
    file_created_at: datetime | None
    file_modified_at: datetime


@dataclass(frozen=True)
class InventoryEntry:
    """Cheap, hash-less stat of one document (ADR-0001b D2): Rescan needs the
    full inventory before it knows which files even need hashing, so this
    stays metadata-only -- see iter_inventory."""

    path: Path
    file_type: str
    file_size: int
    file_created_at: datetime | None
    file_modified_at: datetime
    # Stable filesystem identity for deterministic move matching (ADR-0001b
    # D3) -- always populated from os.stat, never None; whether it is
    # actually unique/stable is validated by the matcher, not here.
    fs_device_id: str
    fs_file_id: str


class UnstableFileError(OSError):
    """A file's size/mtime kept changing across stat -> hash -> stat, even
    after the single retry ADR-0001b D2 allows -- Rescan must fail rather
    than hash content that may not match either observed state."""


def is_supported_document(path: Path) -> bool:
    if path.name.startswith(".") or path.name.startswith(LOCK_FILE_PREFIX):
        return False
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _raise(exc: OSError) -> None:
    raise exc


def _walk_supported_documents(
    root: Path, exclude_roots: frozenset[Path], on_error
) -> Iterator[Path]:
    """Shared pruning walk (see docs/workflow/01a-path-scan.md): already-
    registered child paths are pruned in-place from `dirnames` before
    Path.walk() descends into them, so files under a registered child
    subtree get zero stat -- not merely zero upsert.
    """
    for dirpath, dirnames, filenames in root.walk(on_error=on_error):
        dirnames[:] = [name for name in dirnames if dirpath / name not in exclude_roots]
        for filename in filenames:
            path = dirpath / filename
            if path.is_file() and is_supported_document(path):
                yield path


def iter_documents(
    root: Path, exclude_roots: frozenset[Path] = frozenset()
) -> Iterator[DiscoveredFile]:
    """Recursively yield every supported document under `root`, hashed eagerly
    (mode=initial scan semantics, see docs/workflow/01a-path-scan.md).
    """
    for path in _walk_supported_documents(root, exclude_roots, on_error=None):
        yield _describe(path)


def iter_inventory(
    root: Path, exclude_roots: frozenset[Path] = frozenset()
) -> Iterator[InventoryEntry]:
    """Cheap metadata-only walk for Rescan (ADR-0001b D2): reuses the same
    pruning as iter_documents, but never hashes -- hashing is deferred to the
    diff step, only for files whose cheap metadata actually changed.

    Unlike iter_documents, propagates any OSError encountered during the walk
    (missing/unreadable root, mid-walk scandir failure) instead of letting
    Path.walk() silently swallow it: docs/workflow/01d-path-rescan.md requires
    a single unreadable root to fail the whole Rescan with zero side effects.
    """
    for path in _walk_supported_documents(root, exclude_roots, on_error=_raise):
        yield _describe_cheap(path)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_with_retry(path: Path) -> str:
    """`stat -> SHA-256 -> stat` (ADR-0001b D2): if size/mtime differ between
    the two stats, the file changed mid-hash -- retry once, then fail.
    """
    for _ in range(2):
        before = path.stat()
        digest = compute_sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime):
            return digest
    raise UnstableFileError(f"{path} kept changing while hashing")


def _describe(path: Path) -> DiscoveredFile:
    stat = path.stat()
    return DiscoveredFile(
        path=path,
        file_type=path.suffix.lower().removeprefix("."),
        file_size=stat.st_size,
        file_hash=compute_sha256(path),
        file_created_at=_creation_time(stat),
        file_modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
    )


def _describe_cheap(path: Path) -> InventoryEntry:
    stat = path.stat()
    return InventoryEntry(
        path=path,
        file_type=path.suffix.lower().removeprefix("."),
        file_size=stat.st_size,
        file_created_at=_creation_time(stat),
        file_modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        fs_device_id=str(stat.st_dev),
        fs_file_id=str(stat.st_ino),
    )


def _creation_time(stat: os.stat_result) -> datetime | None:
    """Best-effort filesystem creation time.

    st_birthtime is available on Windows (Python 3.12+) and macOS, but many
    Linux filesystems don't track a birth time at all -- fall back to None
    per the cross-platform note in docs/03_er-diagram.md.
    """
    try:
        return datetime.fromtimestamp(stat.st_birthtime, tz=timezone.utc)
    except AttributeError:
        return None
