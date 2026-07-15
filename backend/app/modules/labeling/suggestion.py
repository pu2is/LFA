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
from app.modules.labeling.merge import select_kinds, write_tag_candidates, write_type_candidates
from app.modules.labeling.models import TagKind, TagLabel, TypeLabelFile
from app.modules.labeling.prompts import (
    AUGMENT_TAG_VALUES_PROMPT,
    INITIAL_KIND_PROMPT,
    INITIAL_TAG_VALUES_PROMPT,
    INITIAL_TYPE_PROMPT,
    InitialKindSuggestionOutput,
    InitialTypeSuggestionOutput,
    TagValuesOutput,
)
from app.modules.labeling.service import ensure_tag_kind_catalog, ensure_type_catalog
from app.modules.rag.models import FileChunk
from app.shared.config import settings

logger = logging.getLogger(__name__)

# ~1-2 pages (rag/service.py CHUNK_SIZE=1000 chars/chunk): classifying type/
# kinds/tags doesn't need the full document (see 01b-file-label-initial.md).
# Reused by augment (01c): "same as initial, only read the first few chunks".
INITIAL_LABEL_MAX_CHUNKS = 5

# >0 on purpose (ADR-0001 D4 f1 / 01c): augment's whole point is finding
# values Call 3 missed the first time; temperature=0 tends to rediscover the
# same values. Exact value TBD by real runs against qwen2.5:3b.
AUGMENT_TEMPERATURE = 0.7


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

    chunk_query = select(FileChunk).where(FileChunk.file_id == file_id).order_by(FileChunk.chunk_index)
    if max_chunks is not None:
        chunk_query = chunk_query.limit(max_chunks)
    chunks = list(db.scalars(chunk_query))
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
    tag_llm = llm.with_structured_output(TagValuesOutput)
    for kind in chosen_kinds:
        messages.extend(INITIAL_TAG_VALUES_PROMPT.format_messages(kind_name=kind.name))
        tag_output: TagValuesOutput = _invoke_or_raise(tag_llm, messages, file_id=file_id)
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
    max_chunks: int | None = INITIAL_LABEL_MAX_CHUNKS,
    temperature: float = AUGMENT_TEMPERATURE,
) -> list[TagLabel]:
    """Augment (ADR-0001 D4 f1, mode=augment): for each tag_kind this file
    already has values under, ask once more whether anything was missed.

    Which kinds to ask about is a DB query -- the distinct kind_id already
    present in this file's tag_labels -- never an LLM decision, so there is
    structurally no code path here that could write type_labels_files.

    Unlike suggest_labels's 3 calls, these per-kind calls do NOT share a
    growing message history with each other: each kind's gap-filling
    question is independent of any other kind's, so every call gets its own
    fresh system+human turn (re-sending the document excerpt each time,
    since there's no prior turn to lean on). jobs.stage stays "tags"
    throughout (no type/kinds stages) -- set by run_label before this runs.

    Append-only (see write_tag_candidates): existing confirmed/rejected/
    suggested rows are never touched, only genuinely new values are
    inserted. Each kind's call writes and commits immediately, so one
    kind's failure doesn't undo another's already-written values.

    temperature is injectable and defaults > 0: f1 is explicitly
    exploratory (finding what Call 3 missed), and temperature=0 tends to
    rediscover the same values as last time -- the accuracy/hallucination
    trade-off is a deliberate one (see docs/workflow/01c).
    """
    existing_tags = list(db.scalars(select(TagLabel).where(TagLabel.file_id == file_id)))
    kind_ids = {t.kind_id for t in existing_tags}
    if not kind_ids:
        logger.info("suggest_labels_augment: file %s has no existing tag_labels to augment", file_id)
        return []
    # Ordered by name: IN (...) doesn't guarantee row order, and each kind's
    # call is independent (no shared history), so a stable, predictable
    # iteration order matters for logs/debugging even though it's otherwise
    # not behaviorally significant.
    kinds = list(db.scalars(select(TagKind).where(TagKind.id.in_(kind_ids)).order_by(TagKind.name)))

    chunk_query = select(FileChunk).where(FileChunk.file_id == file_id).order_by(FileChunk.chunk_index)
    if max_chunks is not None:
        chunk_query = chunk_query.limit(max_chunks)
    chunks = list(db.scalars(chunk_query))
    if not chunks:
        logger.warning("suggest_labels_augment: no chunks for file %s", file_id)
        return []

    if llm is None:
        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=settings.ollama_num_ctx,
        )

    text = "\n\n".join(c.content for c in chunks)
    structured_llm = llm.with_structured_output(TagValuesOutput)

    tag_labels: list[TagLabel] = []
    for kind in kinds:
        kind_tags = [t for t in existing_tags if t.kind_id == kind.id]
        confirmed = [t.value for t in kind_tags if t.status == "confirmed"]
        rejected = [t.value for t in kind_tags if t.status == "rejected"]
        all_existing = [t.value for t in kind_tags]

        messages = AUGMENT_TAG_VALUES_PROMPT.format_messages(
            kind_name=kind.name,
            confirmed=", ".join(confirmed) or "(none)",
            rejected=", ".join(rejected) or "(none)",
            all_existing=", ".join(all_existing) or "(none)",
            text=text,
        )
        output: TagValuesOutput = _invoke_or_raise(structured_llm, messages, file_id=file_id)

        tag_labels.extend(
            write_tag_candidates(db, file_id, kind, output, existing_values=set(all_existing))
        )

    logger.info(
        "suggest_labels_augment: file %s → %d new tag values across %d existing kinds",
        file_id,
        len(tag_labels),
        len(kinds),
    )
    return tag_labels
