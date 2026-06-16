import logging
import uuid
from datetime import datetime, timezone

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.labeling.models import FileLabel, Label
from app.modules.rag.models import FileChunk
from app.shared.config import settings

logger = logging.getLogger(__name__)

# Confidence thresholds (per WF1 in docs/workflows.md).
# ≥ HIGH  → UI pre-selects the label
# ≥ DROP  → stored as "suggested", user sees it
# < DROP  → discarded
CONFIDENCE_THRESHOLD_DROP = 0.5
CONFIDENCE_THRESHOLD_HIGH = 0.75

# For short documents (≤ FULL_READ_CHUNK_LIMIT chunks ≈ ≤ 10 pages at
# CHUNK_SIZE=1000) we read everything — the full file fits comfortably in the
# prompt and gives the model more context.  For longer documents we fall back
# to the first MAX_LABEL_CHUNKS chunks (≈ first 1-2 pages), which is usually
# enough to identify document type.
MAX_LABEL_CHUNKS = 3
FULL_READ_CHUNK_LIMIT = 20  # ~10 pages at CHUNK_SIZE=1000 chars/chunk


# --------------------------------------------------------------------------- #
# LLM output schema
# --------------------------------------------------------------------------- #

class LabelCandidate(BaseModel):
    name: str = Field(description="Label name exactly as listed in the available labels")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0.0–1.0")


class LabelSuggestionOutput(BaseModel):
    labels: list[LabelCandidate] = Field(
        default_factory=list,
        description="Labels that apply to the document, each with a confidence score",
    )


_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document classification assistant. "
            "Select labels from the provided list that apply to this document. "
            "Do not invent labels that are not in the list.",
        ),
        (
            "human",
            "Available labels: {label_names}\n\n"
            "Document excerpt:\n{text}\n\n"
            "For each applicable label assign a confidence score between 0.0 and 1.0. "
            "Only include labels that clearly apply.",
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
# LLM label suggestion (Issue #8)
# --------------------------------------------------------------------------- #

def suggest_labels(
    db: Session,
    file_id: uuid.UUID,
    *,
    llm: BaseChatModel | None = None,
    max_chunks: int = MAX_LABEL_CHUNKS,
) -> list[FileLabel]:
    """Call the LLM to suggest labels for a file, then persist file_labels rows.

    Only labels with confidence >= CONFIDENCE_THRESHOLD_DROP are stored.
    Labels < CONFIDENCE_THRESHOLD_HIGH are still stored as "suggested" — the UI
    uses the confidence value itself to decide whether to pre-select them.

    The LLM client is injectable (llm parameter) so tests can pass a mock
    without hitting Ollama.

    Error handling:
    - httpx connection/timeout errors → re-raised (caller marks job failed, RQ retries)
    - All other errors (bad output, unknown labels, validation failure) → logged,
      returns [] so the job can still succeed (chunks are the valuable output)
    """
    labels = list(db.scalars(select(Label)))
    if not labels:
        return []

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

    chunks = all_chunks if len(all_chunks) <= FULL_READ_CHUNK_LIMIT else all_chunks[:max_chunks]

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
        # Ollama is unreachable — caller is responsible for marking the job
        # failed and letting RQ retry.
        raise
    except Exception as exc:
        logger.warning("suggest_labels: non-fatal LLM error for file %s: %s", file_id, exc)
        return []

    label_by_name = {lbl.name.lower(): lbl for lbl in labels}
    seen_ids: set[uuid.UUID] = set()
    file_labels: list[FileLabel] = []

    for candidate in output.labels:
        if candidate.confidence < CONFIDENCE_THRESHOLD_DROP:
            continue
        lbl = label_by_name.get(candidate.name.lower())
        if lbl is None:
            logger.debug("suggest_labels: LLM returned unknown label %r — skipping", candidate.name)
            continue
        if lbl.id in seen_ids:
            continue
        seen_ids.add(lbl.id)
        fl = FileLabel(
            file_id=file_id,
            label_id=lbl.id,
            source="llm",
            status="suggested",
            confidence=candidate.confidence,
        )
        db.add(fl)
        file_labels.append(fl)

    db.flush()
    return file_labels
