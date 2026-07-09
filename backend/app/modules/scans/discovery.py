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


def is_supported_document(path: Path) -> bool:
    if path.name.startswith(".") or path.name.startswith(LOCK_FILE_PREFIX):
        return False
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def iter_documents(root: Path) -> Iterator[DiscoveredFile]:
    """Recursively yield every supported document under `root`."""
    for path in root.rglob("*"):
        if path.is_file() and is_supported_document(path):
            yield _describe(path)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _creation_time(stat: os.stat_result) -> datetime | None:
    """Best-effort filesystem creation time.

    st_birthtime is available on Windows (Python 3.12+) and macOS, but many
    Linux filesystems don't track a birth time at all -- fall back to None
    per the cross-platform note in docs/3_er-diagram.md.
    """
    try:
        return datetime.fromtimestamp(stat.st_birthtime, tz=timezone.utc)
    except AttributeError:
        return None
