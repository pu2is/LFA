import inspect
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


def _invoke_or_raise(structured_llm, messages, *, file_id: uuid.UUID):
    """Fail loud: log then re-raise any LLM/parse error, never swallow it.

    The log prefix is the immediate caller's function name, read from the
    stack rather than passed in -- it can never drift out of sync with
    whoever's actually calling this.

    run_label marks the job failed and records error_message on this
    exception; RQ retries transient failures. A label job that produced
    nothing must look failed, not succeeded.
    """
    try:
        return structured_llm.invoke(messages)
    except Exception as exc:
        caller = inspect.stack()[1].function
        logger.warning("%s: LLM error for file %s: %s", caller, file_id, exc)
        raise


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

    No confidence scoring happens anywhere in this pipeline (see ADR-0001 D2) --
    every suggestion the LLM returns is stored as "suggested"; quality control
    is entirely the user's confirm/reject action, not a model-reported score.

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

    output: LabelSuggestionOutput = _invoke_or_raise(structured_llm, messages, file_id=file_id)

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

    output: AugmentSuggestionOutput = _invoke_or_raise(structured_llm, messages, file_id=file_id)

    file_labels = append_augment_candidates(db, file_id, output, labels)

    logger.info(
        "suggest_labels_augment: file %s → %d new labels appended",
        file_id,
        len(file_labels),
    )
    return file_labels
