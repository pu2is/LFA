"""Document text extraction via LangChain loaders.

No database access here -- pure file I/O, mirroring the scans/discovery.py
pattern so that extraction logic can be tested and reasoned about in isolation.
"""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pdf2image
import pytesseract
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader

# PDFs with fewer characters than this in their text layer are treated as
# scanned images and routed through the Tesseract OCR fallback.
OCR_THRESHOLD = 50


@dataclass
class ExtractionResult:
    text: str
    ocr_applied: bool = field(default=False)


_LOADERS: dict[str, type] = {
    "pdf": PyPDFLoader,
    "docx": Docx2txtLoader,
}


def _extract_doc(path: Path) -> str:
    """Convert a legacy .doc file to .docx via LibreOffice headless, then extract text."""
    if not shutil.which("soffice"):
        raise RuntimeError(
            "LibreOffice (soffice) not found on PATH. "
            "See dev-setup.md for installation instructions."
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = subprocess.run(
            ["soffice", "--headless", "--convert-to", "docx", "--outdir", tmpdir, str(path)],
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"LibreOffice failed to convert {path.name}: "
                f"{proc.stderr.decode(errors='replace')}"
            )
        docx_path = Path(tmpdir) / path.with_suffix(".docx").name
        if not docx_path.exists():
            raise RuntimeError(f"LibreOffice produced no output file for {path.name}")
        pages = Docx2txtLoader(str(docx_path)).load()
        return "\n".join(p.page_content for p in pages)


def _extract_pdf_with_ocr(path: Path) -> str:
    """OCR fallback for scanned PDFs using Tesseract (eng + deu + chi_sim)."""
    images = pdf2image.convert_from_path(str(path))
    texts = [pytesseract.image_to_string(img, lang="eng+deu+chi_sim") for img in images]
    return "\n".join(texts)


def extract_text(path: Path, file_type: str) -> ExtractionResult:
    """Return the full text of a document.

    For PDFs whose text layer is below OCR_THRESHOLD characters, falls back to
    Tesseract OCR and sets ocr_applied=True on the result.
    Raises RuntimeError when no usable text can be extracted after all fallbacks.
    Raises KeyError for unrecognised file_type values.
    """
    if file_type == "doc":
        text = _extract_doc(path)
        return ExtractionResult(text=text)

    loader_cls = _LOADERS.get(file_type)
    if loader_cls is None:
        raise KeyError(f"Unsupported file type: {file_type!r}")

    pages = loader_cls(str(path)).load()
    text = "\n".join(p.page_content for p in pages)

    if file_type == "pdf" and len(text.strip()) < OCR_THRESHOLD:
        ocr_text = _extract_pdf_with_ocr(path)
        if not ocr_text.strip():
            raise RuntimeError(
                f"No text could be extracted from {path.name}: "
                "text layer too short and OCR produced no output."
            )
        return ExtractionResult(text=ocr_text, ocr_applied=True)

    if not text.strip():
        raise RuntimeError(f"No text could be extracted from {path.name}.")

    return ExtractionResult(text=text)
