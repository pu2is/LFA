import inspect
import logging
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.modules.jobs.models import Job
from app.modules.jobs.service import mark_progress
from app.modules.labeling.merge import (
    append_augment_candidates,
    select_kinds,
    write_tag_candidates,
    write_type_candidates,
)
from app.modules.labeling.models import FileLabel, Label, TagLabel, TypeLabelFile
from app.modules.labeling.presets import OPTIONAL_LABELS, RECOMMENDED_LABELS
from app.modules.labeling.prompts import (
    AUGMENT_SUGGESTION_PROMPT,
    AugmentSuggestionOutput,
    INITIAL_KIND_PROMPT,
    INITIAL_TAG_VALUES_PROMPT,
    INITIAL_TYPE_PROMPT,
    InitialKindSuggestionOutput,
    InitialTagValuesOutput,
    InitialTypeSuggestionOutput,
)
from app.modules.labeling.service import ensure_tag_kind_catalog, ensure_type_catalog
from app.modules.rag.models import FileChunk
from app.shared.config import settings

logger = logging.getLogger(__name__)

# ~1-2 pages (rag/service.py CHUNK_SIZE=1000 chars/chunk): classifying type/
# kinds/tags doesn't need the full document (see 01b-file-label-initial.md).
INITIAL_LABEL_MAX_CHUNKS = 5


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
    whoever's actually calling this. Assumes it's called DIRECTLY by
    suggest_labels/suggest_labels_augment (stack frame 1): if this ever
    gets called through an added layer (a decorator, a retry wrapper, a
    functools.partial), the logged name silently becomes that layer's
    name instead of the real caller's. Re-check this if that changes.

    run_label marks the job failed and records error_message on this
    exception; RQ retries transient failures. A label job that produced
    nothing must look failed, not succeeded.
    """
    try:
        return structured_llm.invoke(messages)
    except Exception as exc:
        caller = inspect.stack()[1].function  # direct caller only, see docstring
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
    job: Job,
    *,
    llm: BaseChatModel | None = None,
    max_chunks: int | None = INITIAL_LABEL_MAX_CHUNKS,
) -> tuple[list[TypeLabelFile], list[TagLabel]]:
    """Three-stage initial labeling (ADR-0001 D3, mode=initial): type -> kinds -> per-kind tags.

    One ChatOllama instance, one growing `messages` list: each call appends
    its own turn before the next runs, so later calls see earlier ones
    (conversation memory) without re-sending the document text every time.
    `job.stage` advances type -> kinds -> tags as the caller (run_label)
    already set it to "type" and called mark_running before this runs.

    Each stage writes AND commits immediately (write_type_candidates /
    write_tag_candidates) rather than just flushing: a mid-flow failure
    (e.g. Call 2, or a later Call-3 iteration) must leave earlier stages'
    suggestions in place, not roll them back — see
    docs/workflow/01b-file-label-initial.md.

    No confidence scoring (see ADR-0001 D2) -- everything lands as
    status=suggested; quality control is the user's confirm/reject action.

    Error handling (fail-loud): any LLM/parse error is logged and re-raised
    by _invoke_or_raise; run_label marks the job failed and records
    error_message on it.
    """
    types = ensure_type_catalog(db)
    kinds = ensure_tag_kind_catalog(db)

    all_chunks = list(
        db.scalars(
            select(FileChunk)
            .where(FileChunk.file_id == file_id)
            .order_by(FileChunk.chunk_index)
        )
    )
    chunks = all_chunks if max_chunks is None else all_chunks[:max_chunks]
    if not chunks:
        logger.warning("suggest_labels: no chunks found for file %s", file_id)
        return [], []

    if llm is None:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_ctx=settings.ollama_num_ctx,
        )

    text = "\n\n".join(c.content for c in chunks)
    messages: list[BaseMessage] = []

    # --- Call 1 (stage=type, set by run_label before calling this) ---
    type_llm = llm.with_structured_output(InitialTypeSuggestionOutput)
    messages.extend(
        INITIAL_TYPE_PROMPT.format_messages(type_names=", ".join(t.name for t in types), text=text)
    )
    type_output: InitialTypeSuggestionOutput = _invoke_or_raise(type_llm, messages, file_id=file_id)
    messages.append(AIMessage(content=type_output.model_dump_json()))

    type_label_files = write_type_candidates(db, file_id, type_output, types)

    # --- Call 2 (stage=kinds) ---
    job.stage = "kinds"
    mark_progress(db, job)

    kind_llm = llm.with_structured_output(InitialKindSuggestionOutput)
    messages.extend(INITIAL_KIND_PROMPT.format_messages(kind_names=", ".join(k.name for k in kinds)))
    kind_output: InitialKindSuggestionOutput = _invoke_or_raise(kind_llm, messages, file_id=file_id)
    messages.append(AIMessage(content=kind_output.model_dump_json()))

    chosen_kinds = select_kinds(kind_output, kinds)

    # --- Call 3 x N (stage=tags): one focused call per chosen kind ---
    job.stage = "tags"
    mark_progress(db, job)

    tag_labels: list[TagLabel] = []
    tag_llm = llm.with_structured_output(InitialTagValuesOutput)
    for kind in chosen_kinds:
        messages.extend(INITIAL_TAG_VALUES_PROMPT.format_messages(kind_name=kind.name))
        tag_output: InitialTagValuesOutput = _invoke_or_raise(tag_llm, messages, file_id=file_id)
        messages.append(AIMessage(content=tag_output.model_dump_json()))

        tag_labels.extend(write_tag_candidates(db, file_id, kind, tag_output))

    logger.info(
        "suggest_labels: file %s → %d type picks, %d tag values across %d kinds",
        file_id,
        len(type_label_files),
        len(tag_labels),
        len(chosen_kinds),
    )

    return type_label_files, tag_labels


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
