"""Write LLM suggestion output into type_labels_files / tag_labels, with
dedup against the existing catalog / existing rows."""
import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.labeling.models import TagKind, TagLabel, TypeLabel, TypeLabelFile
from app.modules.labeling.prompts import InitialKindSuggestionOutput, InitialTypeSuggestionOutput, TagValuesOutput
from app.modules.labeling.service import normalize_label_name

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Initial labeling, ADR-0001 D3 (mode=initial): writes for the type_labels_files
# / tag_labels tables. Each write commits immediately (not just flush) -- a
# failure in a later call must not roll back suggestions a prior call already
# produced, see docs/workflow/01b-file-label-initial.md.
# --------------------------------------------------------------------------- #

def write_type_candidates(
    db: Session,
    file_id: uuid.UUID,
    output: InitialTypeSuggestionOutput,
    types: list[TypeLabel],
) -> list[TypeLabelFile]:
    """Write Call-1 (type) output to type_labels_files. Type is a closed
    catalog -- candidates not matching an existing TypeLabel are dropped.

    Idempotent against (file_id, type_label_id): RQ retries reuse the same
    job row (see #33), and a retried/re-triggered run will call this again
    with the same file -- re-suggesting a type already written here must be
    a no-op, not a UNIQUE-constraint crash that leaves the job stuck.
    """
    type_by_name = {normalize_label_name(t.name): t for t in types}
    existing_type_ids = set(
        db.scalars(select(TypeLabelFile.type_label_id).where(TypeLabelFile.file_id == file_id))
    )
    rows: list[TypeLabelFile] = []
    seen: set[uuid.UUID] = set()

    for candidate in output.types:
        type_label = type_by_name.get(normalize_label_name(candidate.name))
        if type_label is None:
            logger.debug("write_type_candidates: %r not in type catalog — skipping", candidate.name)
            continue
        if type_label.id in seen or type_label.id in existing_type_ids:
            continue
        seen.add(type_label.id)

        row = TypeLabelFile(file_id=file_id, type_label_id=type_label.id, source="llm", status="suggested")
        db.add(row)
        rows.append(row)

    db.commit()
    return rows


def select_kinds(output: InitialKindSuggestionOutput, kinds: list[TagKind]) -> list[TagKind]:
    """Match Call-2 (kinds) output against the tag_kinds catalog.

    Nothing is written to the DB here -- Call 2 only decides which kinds
    Call 3 loops over; kind names not in the catalog are dropped.
    """
    kind_by_name = {normalize_label_name(k.name): k for k in kinds}
    chosen: list[TagKind] = []
    seen: set[uuid.UUID] = set()

    for candidate in output.kinds:
        kind = kind_by_name.get(normalize_label_name(candidate.name))
        if kind is None or kind.id in seen:
            continue
        seen.add(kind.id)
        chosen.append(kind)

    return chosen


def write_tag_candidates(
    db: Session,
    file_id: uuid.UUID,
    kind: TagKind,
    output: TagValuesOutput,
) -> list[TagLabel]:
    """Write one kind's worth of tag values to tag_labels -- shared by initial
    Call 3 and augment's per-kind call, since both are "insert new values
    under this (file, kind), dedup against what's already there."

    Unlike type names, tag values are free text and are NOT case-normalized
    (e.g. a person's name) -- only exact-duplicate/blank values are dropped.
    An empty output simply writes nothing; no special-casing needed.

    Idempotent against (file_id, kind_id, value): a retried/re-triggered run
    (or augment re-suggesting a value already there) is a no-op, never a
    UNIQUE-constraint crash. This is also exactly augment's own append-only
    requirement (docs/workflow/01c-file-label-augment.md) -- confirmed/
    rejected/suggested rows already in the DB are never touched, since only
    genuinely-new values reach the INSERT below.
    """
    existing_values = set(
        db.scalars(
            select(TagLabel.value).where(TagLabel.file_id == file_id, TagLabel.kind_id == kind.id)
        )
    )
    rows: list[TagLabel] = []
    seen: set[str] = set()

    for raw_value in output.values:
        value = raw_value.strip()
        if not value or value in seen or value in existing_values:
            continue
        seen.add(value)

        row = TagLabel(file_id=file_id, kind_id=kind.id, value=value, source="llm", status="suggested")
        db.add(row)
        rows.append(row)

    db.commit()
    return rows
