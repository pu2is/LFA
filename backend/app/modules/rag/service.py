import uuid

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.modules.files.models import File
from app.modules.rag.models import EMBEDDING_DIMENSIONS, FileChunk
from app.shared.config import settings

# Sized for the labeling step (#8), which only reads the first chunk or two --
# large enough to cover a full paragraph, small enough to leave room in the
# prompt. Overlap keeps sentences that straddle a chunk boundary searchable
# from either side once embeddings (Job2, #10) are added.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_and_store(db: Session, file_id: uuid.UUID, text: str) -> list[FileChunk]:
    """Split `text` into chunks and (re)persist them for `file_id`.

    Idempotent: deletes any chunks from a previous run before inserting the
    new ones, so reprocessing a file (e.g. a "modified" rescan, WF1b) never
    leaves stale or duplicate rows. Embeddings are left NULL -- Job2 (#10)
    backfills them later.
    """
    db.execute(delete(FileChunk).where(FileChunk.file_id == file_id))

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = [
        FileChunk(file_id=file_id, chunk_index=index, content=content)
        for index, content in enumerate(splitter.split_text(text))
    ]
    db.add_all(chunks)
    db.flush()
    return chunks


def embed_file(
    db: Session,
    file_id: uuid.UUID,
    embeddings: OllamaEmbeddings | None = None,
) -> None:
    """Backfill NULL embeddings for all chunks of `file_id` using bge-m3.

    Idempotent: only processes chunks where embedding IS NULL, so re-running
    after a partial failure never re-embeds already-completed chunks.
    The `embeddings` parameter is injectable so tests can pass a fake without
    hitting Ollama.
    """
    file = db.get(File, file_id)
    if file is None:
        raise ValueError(f"File {file_id} not found")

    if embeddings is None:
        embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embed_model,
        )

    file.embedding_status = "running"
    db.commit()

    try:
        chunks = db.scalars(
            select(FileChunk)
            .where(FileChunk.file_id == file_id, FileChunk.embedding.is_(None))
            .order_by(FileChunk.chunk_index)
        ).all()

        if chunks:
            vectors = embeddings.embed_documents([c.content for c in chunks])
            for chunk, vector in zip(chunks, vectors):
                chunk.embedding = vector
            db.commit()

        file.embedding_status = "done"
        db.commit()

    except Exception:
        file.embedding_status = "failed"
        db.commit()
        raise
