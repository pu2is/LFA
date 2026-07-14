"""LLM output schemas and prompt templates for label suggestion (initial; the
old single-call flat-schema augment approach is superseded by ADR-0001 D4 f1
below, which reuses TagValuesOutput)."""
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class CatalogCandidate(BaseModel):
    name: str = Field(description="Label name exactly as it appears in the available labels list")


class FreetextCandidate(BaseModel):
    name: str = Field(description="A specific label name you invented; use lowercase_with_underscores")


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


INITIAL_SUGGESTION_PROMPT = ChatPromptTemplate.from_messages(
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
            "Be generous — more labels help the user; they will confirm or reject each one.",
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
# Initial labeling, ADR-0001 D3 (mode=initial): three sequential structured
# calls sharing one growing message history -- type, then kinds, then one
# focused call per chosen kind for its tag values. Each schema is
# deliberately narrower than LabelSuggestionOutput above: type and kind are
# both closed catalogs (no free-text invention), so splitting the "which
# kind" decision (Call 2) from "what values" (Call 3) means Call 3 never has
# to also guess the kind -- see ADR-0001 D3.
# --------------------------------------------------------------------------- #

class InitialTypeCandidate(BaseModel):
    name: str = Field(description="Document type name exactly as it appears in the available types list")


class InitialTypeSuggestionOutput(BaseModel):
    types: list[InitialTypeCandidate] = Field(
        default_factory=list,
        description="At most 3 document types from the catalog that best describe this document",
    )


INITIAL_TYPE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document classification assistant helping organize a user's personal files. "
            "You will be asked a short sequence of related questions about the SAME document. "
            "Only choose from the options given in each question — never invent your own categories.",
        ),
        (
            "human",
            "Document excerpt:\n{text}\n\n"
            "Available document types: {type_names}\n\n"
            "Which of these types apply to this document? Choose at most 3, "
            "picking only names from the list above.",
        ),
    ]
)


class InitialKindCandidate(BaseModel):
    name: str = Field(description="Tag kind name exactly as it appears in the available kinds list")


class InitialKindSuggestionOutput(BaseModel):
    kinds: list[InitialKindCandidate] = Field(
        default_factory=list,
        description="Which tag kinds from the catalog are likely to have relevant information in this document",
    )


INITIAL_KIND_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Available tag categories: {kind_names}\n\n"
            "Given the document and the type(s) you just identified, which of these "
            "categories are likely to have relevant information in this document? "
            "Pick only names from the list above.",
        ),
    ]
)


class TagValuesOutput(BaseModel):
    """Shared by initial Call 3 and augment's per-kind call -- both just ask
    for a flat list of values under a kind fixed by the caller's loop."""

    values: list[str] = Field(
        default_factory=list,
        description="Every specific value for this tag kind mentioned in the document. Empty list if none appear.",
    )


INITIAL_TAG_VALUES_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Focus only on the category '{kind_name}'.\n"
            "List every specific '{kind_name}' value mentioned in the document "
            "(e.g. for 'person', list actual names). Return an empty list if none appear.",
        ),
    ]
)


# --------------------------------------------------------------------------- #
# Augment, ADR-0001 D4 f1 (mode=augment): one independent call per tag_kind
# this file already has values under (which kinds is a DB query, never the
# LLM -- see ADR-0001 D4 and 01c-file-label-augment.md). Unlike the initial
# flow's 3 calls, these do NOT share a growing message history with each
# other -- each kind's gap-filling question is unrelated to any other kind's,
# so every call gets its own fresh system+human turn (including re-sending
# the document excerpt, since there is no prior turn to lean on).
# --------------------------------------------------------------------------- #

AUGMENT_TAG_VALUES_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document classification assistant. "
            "The user already reviewed this document's '{kind_name}' values and wants to "
            "find any that were MISSED the first time.\n\n"
            "Rules:\n"
            "- DO NOT repeat any value from the existing list below.\n"
            "- DO NOT suggest near-synonyms of the rejected values.\n"
            "- Use the confirmed values as positive style references "
            "(the user likes this level of specificity).\n"
            "- Only suggest values you are confident about. If nothing new fits, return an empty list.",
        ),
        (
            "human",
            "Confirmed '{kind_name}' values (user likes these): {confirmed}\n"
            "Rejected '{kind_name}' values (avoid these and near-synonyms): {rejected}\n"
            "All existing '{kind_name}' values (do NOT repeat): {all_existing}\n\n"
            "Document excerpt:\n{text}\n\n"
            "Return only NEW '{kind_name}' values not already in the list above.",
        ),
    ]
)
