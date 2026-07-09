import logging
import uuid
from datetime import datetime, timezone

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.labeling.models import FileLabel, Label
from app.modules.labeling.presets import OPTIONAL_LABELS, RECOMMENDED_LABELS
from app.modules.rag.models import FileChunk
from app.shared.config import settings
from app.shared.events import publish_event

logger = logging.getLogger(__name__)

# Confidence thresholds (per WF1 in docs/workflows.md).
# ≥ HIGH  → UI pre-selects the label
# ≥ DROP  → stored as "suggested", user sees it
# < DROP  → discarded
CONFIDENCE_THRESHOLD_DROP = 0.0
CONFIDENCE_THRESHOLD_HIGH = 0.75


def normalize_label_name(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")



# --------------------------------------------------------------------------- #
# LLM output schema
# --------------------------------------------------------------------------- #

def _coerce_confidence(v: object) -> object:
    """Normalize an LLM-provided confidence to the 0.0–1.0 range.

    Small local models (e.g. qwen2.5:3b) ignore the 0–1 instruction and often
    return a 0–100 percentage. Without this, pydantic's le=1.0 check fails the
    WHOLE structured-output parse, suggest_labels swallows it, and the label job
    silently produces zero labels. Divide >1 values by 100 and clamp to [0, 1].
    """
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return v  # let pydantic raise its normal validation error
    if f > 1.0:
        f = f / 100.0
    return min(max(f, 0.0), 1.0)


class CatalogCandidate(BaseModel):
    name: str = Field(description="Label name exactly as it appears in the available labels list")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0–1.0")

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: object) -> object:
        return _coerce_confidence(v)


class FreetextCandidate(BaseModel):
    name: str = Field(description="A specific label name you invented; use lowercase_with_underscores")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0–1.0")

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: object) -> object:
        return _coerce_confidence(v)


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


def _write_initial_candidates(
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
        if candidate.confidence < CONFIDENCE_THRESHOLD_DROP:
            continue
        norm = normalize_label_name(candidate.name)
        lbl = label_by_name.get(norm)
        if lbl is None:
            logger.debug("_write_initial: catalog pick %r not in label catalog — skipping", candidate.name)
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

    seen_free_names: set[str] = set()
    for candidate in output.free_suggestions:
        if candidate.confidence < CONFIDENCE_THRESHOLD_DROP:
            continue
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
            confidence=candidate.confidence,
        )
        db.add(fl)
        file_labels.append(fl)

    db.flush()
    return file_labels


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

    Error handling (fail-loud):
    - Any LLM/parse error → logged and re-raised. run_label marks the job failed
      and records error_message; RQ retries transient failures (e.g. Ollama down).
      We do NOT swallow: a label job that produced nothing must look failed, not
      succeeded. Labeling is its own job, so failing it never undoes ingest/embed.
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
    except Exception as exc:
        # Fail loud — see "Error handling" in the docstring.
        logger.warning("suggest_labels: LLM error for file %s: %s", file_id, exc)
        raise

    file_labels = _write_initial_candidates(db, file_id, output, labels)

    logger.info(
        "suggest_labels: file %s → %d catalog picks, %d free-text suggestions",
        file_id,
        len([f for f in file_labels if f.label_id is not None]),
        len([f for f in file_labels if f.label_id is None]),
    )

    return file_labels



# --------------------------------------------------------------------------- #
# Augment labeling (mode=augment, #26)
# --------------------------------------------------------------------------- #

class AugmentCandidate(BaseModel):
    name: str = Field(description="A new label name; use lowercase_with_underscores")


class AugmentSuggestionOutput(BaseModel):
    new_labels: list[AugmentCandidate] = Field(
        default_factory=list,
        description="New labels that describe this document from a different angle",
    )


_AUGMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document classification assistant. "
            "The user already has labels on this document and wants MORE labels "
            "from DIFFERENT angles or finer granularity.\n\n"
            "Rules:\n"
            "- DO NOT repeat any label from the existing list below.\n"
            "- DO NOT suggest synonyms or near-synonyms of rejected labels.\n"
            "- Use the confirmed labels as positive style references "
            "(the user likes this level of specificity).\n"
            "- Invent specific, fine-grained labels in lowercase_with_underscores.\n"
            "- Only suggest labels you are confident about. "
            "If nothing fits, return an empty list.\n"
            "- You may pick from the catalog OR invent new names.",
        ),
        (
            "human",
            "Confirmed labels (user likes these): {confirmed}\n"
            "Rejected labels (avoid these and synonyms): {rejected}\n"
            "All existing label names (do NOT repeat): {all_existing}\n\n"
            "Available catalog labels: {catalog}\n\n"
            "Document excerpt:\n{text}\n\n"
            "Return new_labels only — labels NOT in the existing list above.",
        ),
    ]
)


def _append_augment_candidates(
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
            confidence=None,
        )
        db.add(fl)
        file_labels.append(fl)

    db.flush()
    return file_labels


def suggest_labels_augment(
    db: Session,
    file_id: uuid.UUID,
    *,
    llm: BaseChatModel | None = None,
    max_chunks: int | None = None,
) -> list[FileLabel]:
    """Augment prompt: suggest NEW labels for a file that already has labels."""
    labels = _ensure_label_catalog(db)

    all_chunks = list(
        db.scalars(
            select(FileChunk)
            .where(FileChunk.file_id == file_id)
            .order_by(FileChunk.chunk_index)
        )
    )
    if not all_chunks:
        logger.warning("suggest_labels_augment: no chunks for file %s", file_id)
        return []

    chunks = all_chunks if max_chunks is None else all_chunks[:max_chunks]

    existing = list(db.scalars(select(FileLabel).where(FileLabel.file_id == file_id)))
    confirmed = [fl.label_name for fl in existing if fl.status == "confirmed"]
    rejected = [fl.label_name for fl in existing if fl.status == "rejected"]
    all_existing = [fl.label_name for fl in existing]

    if llm is None:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    structured_llm = llm.with_structured_output(AugmentSuggestionOutput)
    messages = _AUGMENT_PROMPT.format_messages(
        confirmed=", ".join(confirmed) or "(none)",
        rejected=", ".join(rejected) or "(none)",
        all_existing=", ".join(all_existing) or "(none)",
        catalog=", ".join(lbl.name for lbl in labels),
        text="\n\n".join(c.content for c in chunks),
    )

    try:
        output: AugmentSuggestionOutput = structured_llm.invoke(messages)
    except Exception as exc:
        # Fail loud — a label job that produced nothing must look failed.
        logger.warning("suggest_labels_augment: LLM error for file %s: %s", file_id, exc)
        raise

    file_labels = _append_augment_candidates(db, file_id, output, labels)

    logger.info(
        "suggest_labels_augment: file %s → %d new labels appended",
        file_id,
        len(file_labels),
    )
    return file_labels


# --------------------------------------------------------------------------- #
# Label job runner (unified jobs table)
# --------------------------------------------------------------------------- #

def _publish_label_event(job) -> None:
    data: dict = {
        "job_id": str(job.id),
        "type": job.type,
        "file_id": str(job.file_id),
        "status": job.status,
        "mode": job.mode,
    }
    if job.error_message:
        data["error_message"] = job.error_message
    publish_event("job_status", data)


def run_label(
    db: Session,
    job_id: uuid.UUID,
    *,
    llm: BaseChatModel | None = None,
):
    """Execute a label job: dispatch to initial or augment based on job.mode."""
    from app.modules.files.models import File
    from app.modules.jobs.models import Job

    job = db.get(Job, job_id)
    if job is None:
        raise ValueError(f"Job {job_id} not found")

    file = db.get(File, job.file_id)
    if file is None:
        raise ValueError(f"File {job.file_id} not found")

    job.status = "running"
    job.stage = "labeling"
    job.started_at = datetime.now(timezone.utc)
    db.commit()
    _publish_label_event(job)

    try:
        if job.mode == "augment":
            suggest_labels_augment(db, file.id, llm=llm)
        else:
            suggest_labels(db, file.id, llm=llm)
    except Exception as exc:
        job.status = "failed"
        job.error_message = str(exc)
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        _publish_label_event(job)
        raise

    job.status = "succeeded"
    job.stage = None
    job.completed_at = datetime.now(timezone.utc)
    db.commit()
    _publish_label_event(job)

    return job


def file_has_labels(db: Session, file_id: uuid.UUID) -> bool:
    return db.scalar(
        select(FileLabel.id).where(FileLabel.file_id == file_id).limit(1)
    ) is not None


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
