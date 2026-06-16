import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.modules.rag.models import FileChunk

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
