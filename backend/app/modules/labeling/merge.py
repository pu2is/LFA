"""Write LLM suggestion output into type_labels_files / tag_labels, with
dedup against the existing catalog / existing rows."""
import logging
import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
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

    Idempotent against (file_id, type_label_id) via INSERT ... ON CONFLICT
    DO NOTHING ... RETURNING (#53): the prior SELECT-then-INSERT let two
    concurrent writers (RQ retries reusing the same job row, see #33, or a
    retried/re-triggered run) both read "not yet present" and race to insert
    the same row, so the loser's commit crashed on the UNIQUE constraint
    despite this function's documented idempotency. The atomic upsert makes
    losing that race a silent no-op -- the conflicting row just isn't in
    RETURNING -- instead.
    """
    type_by_name = {normalize_label_name(t.name): t for t in types}
    seen: set[uuid.UUID] = set()
    values: list[dict] = []

    for candidate in output.types:
        type_label = type_by_name.get(normalize_label_name(candidate.name))
        if type_label is None:
            logger.debug("write_type_candidates: %r not in type catalog — skipping", candidate.name)
            continue
        if type_label.id in seen:
            continue
        seen.add(type_label.id)
        values.append({"file_id": file_id, "type_label_id": type_label.id, "source": "llm", "status": "suggested"})

    if not values:
        return []

    stmt = (
        pg_insert(TypeLabelFile)
        .values(values)
        .on_conflict_do_nothing(index_elements=[TypeLabelFile.file_id, TypeLabelFile.type_label_id])
        .returning(TypeLabelFile)
    )
    rows = list(db.scalars(stmt))
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
    *,
    existing_values: set[str] | None = None,
) -> list[TagLabel]:
    """Write one kind's worth of tag values to tag_labels -- shared by initial
    Call 3 and augment's per-kind call, since both are "insert new values
    under this (file, kind), dedup against what's already there."

    Unlike type names, tag values are free text and are NOT case-normalized
    on write (e.g. a person's name) -- but duplicates are now judged case-
    insensitively (#49: "Berlin" vs "berlin"), matching the DB's
    ix_tag_labels_file_kind_value_lower index. Only blank values are dropped
    outright. An empty output simply writes nothing; no special-casing needed.

    Idempotent against (file_id, kind_id, lower(value)): a retried/re-
    triggered run (or augment re-suggesting a value already there, including
    a case variant) is a no-op, never a UNIQUE-constraint crash. This is also
    exactly augment's own append-only requirement (docs/workflow/01c-file-
    label-augment.md) -- confirmed/rejected/suggested rows already in the DB
    are never touched, since only genuinely-new values reach the INSERT below.

    existing_values: pass this (file, kind)'s current values, lowercased, if
    the caller already has them loaded (augment does, from its own upfront
    query) to skip the redundant re-query; omitted (initial's Call 3, which
    has no prior read of tag_labels) queries and lowercases them here.
    """
    if existing_values is None:
        existing_values = {
            v.lower()
            for v in db.scalars(
                select(TagLabel.value).where(TagLabel.file_id == file_id, TagLabel.kind_id == kind.id)
            )
        }
    rows: list[TagLabel] = []
    seen: set[str] = set()

    for raw_value in output.values:
        value = raw_value.strip()
        value_lower = value.lower()
        if not value or value_lower in seen or value_lower in existing_values:
            continue
        seen.add(value_lower)

        row = TagLabel(file_id=file_id, kind_id=kind.id, value=value, source="llm", status="suggested")
        db.add(row)
        rows.append(row)

    db.commit()
    return rows
