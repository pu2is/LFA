"""Document text extraction via LangChain loaders.

No database access here -- pure file I/O, mirroring the scans/discovery.py
pattern so that extraction logic can be tested and reasoned about in isolation.
"""

from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader

_LOADERS: dict[str, type] = {
    "pdf": PyPDFLoader,
    "docx": Docx2txtLoader,
}


def extract_text(path: Path, file_type: str) -> str:
    """Return the full text of a document using the LangChain loader for its type.

    Raises KeyError for unsupported file types (e.g. legacy .doc -- see #7).
    OCR fallback for scanned PDFs is also deferred to #7.
    """
    loader_cls = _LOADERS[file_type]
    pages = loader_cls(str(path)).load()
    return "\n".join(page.page_content for page in pages)
