import logging
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.labeling.merge import append_augment_candidates, write_initial_candidates
from app.modules.labeling.models import FileLabel, Label
from app.modules.labeling.presets import OPTIONAL_LABELS, RECOMMENDED_LABELS
from app.modules.labeling.prompts import (
    AUGMENT_SUGGESTION_PROMPT,
    AugmentSuggestionOutput,
    INITIAL_SUGGESTION_PROMPT,
    LabelSuggestionOutput,
)
from app.modules.rag.models import FileChunk
from app.shared.config import settings

logger = logging.getLogger(__name__)

# UI hint only, not enforced here: the frontend pre-selects a label when its
# confidence is >= this value; everything else is still stored as
# "suggested" for the user to review. There is no server-side drop
# threshold -- every suggestion the LLM returns gets stored (the model is
# asked to only return confidence >= 0.25 in the prompt itself, but nothing
# here re-checks that on the way in; a prior CONFIDENCE_THRESHOLD_DROP = 0.0
# constant implied a stricter contract than actually existed and was removed
# as dead code, see #34).
CONFIDENCE_THRESHOLD_HIGH = 0.75


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


def _load_chunks_and_llm(
    db: Session,
    file_id: uuid.UUID,
    *,
    llm: BaseChatModel | None,
    max_chunks: int | None,
) -> tuple[list[Label], list[FileChunk], BaseChatModel]:
    """Shared scaffolding for suggest_labels / suggest_labels_augment.

    Ensures the label catalog exists, loads this file's chunks (ordered,
    capped to max_chunks), and resolves a default Ollama client if none was
    injected. Returns an empty chunks list when the file has no chunks yet --
    callers check for that themselves since the "no chunks" log message
    differs between initial and augment mode.
    """
    labels = _ensure_label_catalog(db)

    all_chunks = list(
        db.scalars(
            select(FileChunk)
            .where(FileChunk.file_id == file_id)
            .order_by(FileChunk.chunk_index)
        )
    )
    chunks = all_chunks if max_chunks is None else all_chunks[:max_chunks]

    if llm is None:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    return labels, chunks, llm


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

    No confidence-based filtering happens server-side — every suggestion the
    LLM returns is stored as "suggested". Labels >= CONFIDENCE_THRESHOLD_HIGH
    (0.75) are a UI hint for pre-selecting; the rest still show up for review.

    The LLM client is injectable so tests can pass a mock without hitting Ollama.

    Error handling (fail-loud):
    - Any LLM/parse error → logged and re-raised. run_label marks the job failed
      and records error_message; RQ retries transient failures (e.g. Ollama down).
      We do NOT swallow: a label job that produced nothing must look failed, not
      succeeded. Labeling is its own job, so failing it never undoes ingest/embed.
    """
    labels, chunks, llm = _load_chunks_and_llm(db, file_id, llm=llm, max_chunks=max_chunks)
    if not chunks:
        logger.warning("suggest_labels: no chunks found for file %s", file_id)
        return []

    structured_llm = llm.with_structured_output(LabelSuggestionOutput)
    messages = INITIAL_SUGGESTION_PROMPT.format_messages(
        label_names=", ".join(lbl.name for lbl in labels),
        text="\n\n".join(c.content for c in chunks),
    )

    try:
        output: LabelSuggestionOutput = structured_llm.invoke(messages)
    except Exception as exc:
        # Fail loud — see "Error handling" in the docstring.
        logger.warning("suggest_labels: LLM error for file %s: %s", file_id, exc)
        raise

    file_labels = write_initial_candidates(db, file_id, output, labels)

    logger.info(
        "suggest_labels: file %s → %d catalog picks, %d free-text suggestions",
        file_id,
        len([f for f in file_labels if f.label_id is not None]),
        len([f for f in file_labels if f.label_id is None]),
    )

    return file_labels


def suggest_labels_augment(
    db: Session,
    file_id: uuid.UUID,
    *,
    llm: BaseChatModel | None = None,
    max_chunks: int | None = None,
) -> list[FileLabel]:
    """Augment prompt: suggest NEW labels for a file that already has labels."""
    labels, chunks, llm = _load_chunks_and_llm(db, file_id, llm=llm, max_chunks=max_chunks)
    if not chunks:
        logger.warning("suggest_labels_augment: no chunks for file %s", file_id)
        return []

    existing = list(db.scalars(select(FileLabel).where(FileLabel.file_id == file_id)))
    confirmed = [fl.label_name for fl in existing if fl.status == "confirmed"]
    rejected = [fl.label_name for fl in existing if fl.status == "rejected"]
    all_existing = [fl.label_name for fl in existing]

    structured_llm = llm.with_structured_output(AugmentSuggestionOutput)
    messages = AUGMENT_SUGGESTION_PROMPT.format_messages(
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

    file_labels = append_augment_candidates(db, file_id, output, labels)

    logger.info(
        "suggest_labels_augment: file %s → %d new labels appended",
        file_id,
        len(file_labels),
    )
    return file_labels
