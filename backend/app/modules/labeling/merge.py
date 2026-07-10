"""Write LLM suggestion output into file_labels, with dedup against the existing catalog."""
import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.labeling.models import FileLabel, Label
from app.modules.labeling.prompts import AugmentSuggestionOutput, LabelSuggestionOutput
from app.modules.labeling.service import normalize_label_name

logger = logging.getLogger(__name__)


def write_initial_candidates(
    db: Session,
    file_id: uuid.UUID,
    output: LabelSuggestionOutput,
    labels: list[Label],
) -> list[FileLabel]:
    """Write LLM candidates to file_labels for first-time labeling (mode=initial).

    Pure INSERT — initial mode targets files with no existing file_labels.
    """
    label_by_name = {normalize_label_name(lbl.name): lbl for lbl in labels}
    file_labels: list[FileLabel] = []
    seen_label_ids: set[uuid.UUID] = set()

    for candidate in output.catalog_picks:
        norm = normalize_label_name(candidate.name)
        lbl = label_by_name.get(norm)
        if lbl is None:
            logger.debug("write_initial: catalog pick %r not in label catalog — skipping", candidate.name)
            continue
        if lbl.id in seen_label_ids:
            continue
        seen_label_ids.add(lbl.id)

        fl = FileLabel(
            file_id=file_id,
            label_id=lbl.id,
            label_name=lbl.name,
            source="llm",
            status="suggested",
        )
        db.add(fl)
        file_labels.append(fl)

    seen_free_names: set[str] = set()
    for candidate in output.free_suggestions:
        normalized = normalize_label_name(candidate.name)
        if not normalized:
            continue
        if normalized in label_by_name:
            continue
        if normalized in seen_free_names:
            continue
        seen_free_names.add(normalized)

        fl = FileLabel(
            file_id=file_id,
            label_id=None,
            label_name=normalized,
            source="llm",
            status="suggested",
        )
        db.add(fl)
        file_labels.append(fl)

    db.flush()
    return file_labels


def append_augment_candidates(
    db: Session,
    file_id: uuid.UUID,
    output: AugmentSuggestionOutput,
    labels: list[Label],
) -> list[FileLabel]:
    """Append-only write for augment mode: INSERT new names, never touch existing rows."""
    existing = list(db.scalars(select(FileLabel).where(FileLabel.file_id == file_id)))
    existing_names: set[str] = {normalize_label_name(fl.label_name) for fl in existing}

    label_by_name = {normalize_label_name(lbl.name): lbl for lbl in labels}
    file_labels: list[FileLabel] = []
    seen_names: set[str] = set()

    for candidate in output.new_labels:
        normalized = normalize_label_name(candidate.name)
        if not normalized:
            continue
        if normalized in existing_names:
            continue
        if normalized in seen_names:
            continue
        seen_names.add(normalized)

        lbl = label_by_name.get(normalized)
        fl = FileLabel(
            file_id=file_id,
            label_id=lbl.id if lbl else None,
            label_name=lbl.name if lbl else normalized,
            source="llm",
            status="suggested",
        )
        db.add(fl)
        file_labels.append(fl)

    db.flush()
    return file_labels
