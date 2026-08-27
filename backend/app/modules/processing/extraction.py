"""Document text extraction via LangChain loaders.

No database access here -- pure file I/O, mirroring the scans/discovery.py
pattern so that extraction logic can be tested and reasoned about in isolation.
"""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import fitz  # pymupdf — bundles MuPDF statically, no external binary needed
import easyocr
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader

# PDFs with fewer characters than this in their text layer are treated as
# scanned images and routed through the EasyOCR fallback.
OCR_THRESHOLD = 50

# EasyOCR's ch_sim model is only compatible with en — cannot combine with de.
# Two Readers: Latin (en+de) primary, Chinese (ch_sim+en) fallback.
_latin_reader: easyocr.Reader | None = None
_chinese_reader: easyocr.Reader | None = None

# If Latin reader's average confidence is below this, try the Chinese reader.
CONFIDENCE_THRESHOLD = 0.5
# Even if confidence looks okay, if total text is shorter than this,
# the Latin reader probably missed most of the content — try Chinese too.
MIN_TEXT_GUARD = 10


def _get_latin_reader() -> easyocr.Reader:
    global _latin_reader
    if _latin_reader is None:
        _latin_reader = easyocr.Reader(["en", "de"], gpu=False)
    return _latin_reader


def _get_chinese_reader() -> easyocr.Reader:
    global _chinese_reader
    if _chinese_reader is None:
        _chinese_reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
    return _chinese_reader


@dataclass
class OcrResult:
    """Internal container for raw EasyOCR output (text + per-detection confidences)."""
    text: str
    confidences: Sequence[float]

    @property
    def avg_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0


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


def _ocr_pages(reader: easyocr.Reader, doc) -> OcrResult:
    page_texts: list[str] = []
    all_confidences: list[float] = []
    for page in doc:
        img_bytes = page.get_pixmap(dpi=300).tobytes("png")
        detections = reader.readtext(img_bytes)
        page_texts.append("\n".join(d[1] for d in detections))
        all_confidences.extend(d[2] for d in detections)
    return OcrResult(text="\n".join(page_texts), confidences=all_confidences)


def _needs_chinese_fallback(latin: OcrResult) -> bool:
    """Decide whether the Chinese reader should be tried.

    Triggers when the Latin reader is not confident about its output OR
    when it extracted very little text (low-text guard).
    """
    return latin.avg_confidence < CONFIDENCE_THRESHOLD or len(latin.text.strip()) < MIN_TEXT_GUARD


def _extract_pdf_with_ocr(path: Path) -> str:
    """OCR fallback for scanned PDFs using EasyOCR + PyMuPDF (no external binaries).

    Runs the Latin (en+de) reader first. If average confidence is below
    CONFIDENCE_THRESHOLD or the extracted text is shorter than MIN_TEXT_GUARD,
    also runs the Chinese (ch_sim+en) reader and keeps the stronger output.
    """
    doc = fitz.open(str(path))
    latin = _ocr_pages(_get_latin_reader(), doc)

    if not _needs_chinese_fallback(latin):
        doc.close()
        return latin.text

    chinese = _ocr_pages(_get_chinese_reader(), doc)
    doc.close()

    return chinese.text if len(chinese.text.strip()) > len(latin.text.strip()) else latin.text


def extract_text(path: Path, file_type: str, *, use_ocr: bool = True) -> ExtractionResult:
    """Return the full text of a document.

    For PDFs whose text layer is below OCR_THRESHOLD characters, falls back to
    EasyOCR and sets ocr_applied=True on the result -- unless use_ocr=False, in
    which case a thin text layer raises instead of escalating to OCR. Rescan's
    fuzzy-recovery text-signature step (ADR-0001b D3) passes use_ocr=False:
    Rescan must stay on deterministic/cheap matching and never upgrade to OCR.
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
        if not use_ocr:
            raise RuntimeError(f"No text layer available for {path.name} and OCR is disabled.")
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
