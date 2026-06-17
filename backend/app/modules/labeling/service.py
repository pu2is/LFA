import logging
import uuid

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.labeling.models import FileLabel, Label
from app.modules.labeling.presets import OPTIONAL_LABELS, RECOMMENDED_LABELS
from app.modules.rag.models import FileChunk
from app.shared.config import settings

logger = logging.getLogger(__name__)

# Confidence thresholds (per WF1 in docs/workflows.md).
# ≥ HIGH  → UI pre-selects the label
# ≥ DROP  → stored as "suggested", user sees it
# < DROP  → discarded
CONFIDENCE_THRESHOLD_DROP = 0.0
CONFIDENCE_THRESHOLD_HIGH = 0.75



# --------------------------------------------------------------------------- #
# LLM output schema
# --------------------------------------------------------------------------- #

class CatalogCandidate(BaseModel):
    name: str = Field(description="Label name exactly as it appears in the available labels list")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0–1.0")


class FreetextCandidate(BaseModel):
    name: str = Field(description="A specific label name you invented; use lowercase_with_underscores")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0–1.0")


class LabelSuggestionOutput(BaseModel):
    catalog_picks: list[CatalogCandidate] = Field(
        default_factory=list,
        description="Labels chosen from the provided list that apply to this document",
    )
    free_suggestions: list[FreetextCandidate] = Field(
        default_factory=list,
        description=(
            "Additional specific labels you invented that better describe this document "
            "and are NOT already covered by the catalog labels above. "
            "Use lowercase_with_underscores."
        ),
    )


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document classification assistant. "
            "Your job is to label documents as thoroughly as possible.\n\n"
            "You have two tasks:\n"
            "1. Select every applicable label from the provided catalog list.\n"
            "2. Suggest additional fine-grained labels NOT in the list "
            "if they describe the document more specifically (e.g. 'car_rental_agreement' "
            "instead of just 'contract'). Use lowercase_with_underscores.\n\n"
            "For every label (catalog or invented) assign a confidence score 0.0–1.0. "
            "Include all labels with confidence >= 0.25. Be generous — more labels help the user.",
        ),
        (
            "human",
            "Available catalog labels: {label_names}\n\n"
            "Document excerpt:\n{text}\n\n"
            "Return catalog_picks (from the list above) and free_suggestions (your own additions).",
        ),
    ]
)


# --------------------------------------------------------------------------- #
# Label CRUD (pre-existing)
# --------------------------------------------------------------------------- #

def list_labels(db: Session) -> list[Label]:
    return list(db.scalars(select(Label)))


def get_label(db: Session, label_id: uuid.UUID) -> Label | None:
    return db.get(Label, label_id)


def bulk_create_labels(db: Session, names: list[str]) -> tuple[list[Label], list[str]]:
    # In-request dedupe, preserving first-seen order.
    unique_names = list(dict.fromkeys(names))

    existing_names = set(db.scalars(select(Label.name).where(Label.name.in_(unique_names))))
    skipped = [name for name in unique_names if name in existing_names]
    labels = [Label(name=name) for name in unique_names if name not in existing_names]

    # One transaction for the whole batch: all rows land or none do.
    db.add_all(labels)
    db.commit()
    for label in labels:
        db.refresh(label)
    return labels, skipped


def delete_label(db: Session, label: Label) -> None:
    db.delete(label)
    db.commit()


# --------------------------------------------------------------------------- #
# LLM label suggestion
# --------------------------------------------------------------------------- #

def _ensure_label_catalog(db: Session) -> list[Label]:
    """Return all labels in the catalog; auto-populate from presets if empty."""
    labels = list(db.scalars(select(Label)))
    if labels:
        return labels

    logger.info("suggest_labels: label catalog is empty — auto-populating from presets")
    all_preset_names = list(RECOMMENDED_LABELS) + list(OPTIONAL_LABELS)
    new_labels = [Label(name=name) for name in all_preset_names]
    db.add_all(new_labels)
    db.commit()
    for lbl in new_labels:
        db.refresh(lbl)
    return new_labels


def suggest_labels(
    db: Session,
    file_id: uuid.UUID,
    *,
    llm: BaseChatModel | None = None,
    max_chunks: int | None = None,
) -> list[FileLabel]:
    """Call the LLM to suggest labels for a file, then persist file_labels rows.

    Two output paths:
    - catalog_picks: LLM selects from the existing Label catalog → FileLabel with label_id set
    - free_suggestions: LLM invents finer-grained names → FileLabel with label_name set, label_id NULL

    If the label catalog is empty, all presets are auto-inserted before calling the LLM.

    Confidence filtering: DROP threshold = 0.0 (both paths — all suggestions stored for dev review).
    Labels >= HIGH (0.75) are UI pre-selected; others stored as "suggested" for user review.

    The LLM client is injectable so tests can pass a mock without hitting Ollama.

    Error handling:
    - httpx connection/timeout → re-raised (caller marks job failed, RQ retries)
    - All other errors → logged, returns [] (chunks are still valuable)
    """
    labels = _ensure_label_catalog(db)

    all_chunks = list(
        db.scalars(
            select(FileChunk)
            .where(FileChunk.file_id == file_id)
            .order_by(FileChunk.chunk_index)
        )
    )
    if not all_chunks:
        logger.warning("suggest_labels: no chunks found for file %s", file_id)
        return []

    chunks = all_chunks if max_chunks is None else all_chunks[:max_chunks]

    if llm is None:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    structured_llm = llm.with_structured_output(LabelSuggestionOutput)
    messages = _PROMPT.format_messages(
        label_names=", ".join(lbl.name for lbl in labels),
        text="\n\n".join(c.content for c in chunks),
    )

    try:
        output: LabelSuggestionOutput = structured_llm.invoke(messages)
    except (httpx.ConnectError, httpx.TimeoutException):
        raise
    except Exception as exc:
        logger.warning("suggest_labels: non-fatal LLM error for file %s: %s", file_id, exc)
        return []

    file_labels: list[FileLabel] = []

    # --- Path A: catalog picks ---
    label_by_name = {lbl.name.lower(): lbl for lbl in labels}
    seen_label_ids: set[uuid.UUID] = set()
    for candidate in output.catalog_picks:
        if candidate.confidence < CONFIDENCE_THRESHOLD_DROP:
            continue
        lbl = label_by_name.get(candidate.name.lower())
        if lbl is None:
            logger.debug("suggest_labels: catalog pick %r not found in labels — skipping", candidate.name)
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
            confidence=candidate.confidence,
        )
        db.add(fl)
        file_labels.append(fl)

    # --- Path B: free-text suggestions ---
    seen_free_names: set[str] = set()
    for candidate in output.free_suggestions:
        if candidate.confidence < CONFIDENCE_THRESHOLD_DROP:
            continue
        normalized = candidate.name.strip().lower().replace(" ", "_")
        if not normalized:
            continue
        # Skip if LLM essentially re-suggested a catalog label by another name.
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
            confidence=candidate.confidence,
        )
        db.add(fl)
        file_labels.append(fl)

    logger.info(
        "suggest_labels: file %s → %d catalog picks, %d free-text suggestions",
        file_id,
        len([f for f in file_labels if f.label_id is not None]),
        len([f for f in file_labels if f.label_id is None]),
    )

    db.flush()
    return file_labels


# --------------------------------------------------------------------------- #
# File-label review operations
# --------------------------------------------------------------------------- #

def list_file_labels(db: Session, file_id: uuid.UUID) -> list[FileLabel]:
    return list(db.scalars(select(FileLabel).where(FileLabel.file_id == file_id)))


def get_file_label_by_id(db: Session, file_label_id: uuid.UUID) -> FileLabel | None:
    """Fetch a file_label row by its own PK (works for both catalog and free-text rows)."""
    return db.get(FileLabel, file_label_id)


def get_file_label_by_catalog(db: Session, file_id: uuid.UUID, label_id: uuid.UUID) -> FileLabel | None:
    """Fetch a catalog file_label row by (file_id, label_id). Used for duplicate checks."""
    return db.scalar(
        select(FileLabel).where(
            FileLabel.file_id == file_id,
            FileLabel.label_id == label_id,
        )
    )


def batch_patch_file_labels(
    db: Session,
    file_id: uuid.UUID,
    operations: list[tuple[uuid.UUID, str]],
) -> list[FileLabel]:
    """Confirm or reject file_label rows in bulk, addressed by file_labels.id.

    All-or-nothing: raises ValueError listing any IDs not found or not belonging
    to this file so the caller can return a 404 before touching the DB.
    """
    to_update: list[tuple[FileLabel, str]] = []
    missing: list[str] = []
    for file_label_id, action in operations:
        fl = get_file_label_by_id(db, file_label_id)
        if fl is None or fl.file_id != file_id:
            missing.append(str(file_label_id))
        else:
            to_update.append((fl, action))
    if missing:
        raise ValueError(f"file_label not found for label_id(s): {', '.join(missing)}")

    for fl, action in to_update:
        fl.status = "confirmed" if action == "confirm" else "rejected"

    db.commit()
    for fl, _ in to_update:
        db.refresh(fl)
    return [fl for fl, _ in to_update]


def add_user_label(db: Session, file_id: uuid.UUID, label: Label) -> FileLabel:
    fl = FileLabel(
        file_id=file_id,
        label_id=label.id,
        label_name=label.name,
        source="user",
        status="confirmed",
        confidence=None,
    )
    db.add(fl)
    db.commit()
    db.refresh(fl)
    return fl


def remove_file_label(db: Session, fl: FileLabel) -> None:
    db.delete(fl)
    db.commit()
