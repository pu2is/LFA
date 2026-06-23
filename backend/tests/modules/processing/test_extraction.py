"""Unit tests for processing/extraction.py.

LibreOffice is mocked so these tests run without any system-level installation.
EasyOCR/PyMuPDF OCR fallback is tested by patching _extract_pdf_with_ocr directly,
which avoids loading the ~200 MB EasyOCR model in a unit-test context.

Loader mocking note: _LOADERS is populated at import time with references to
the actual loader classes.  Patching the *name* PyPDFLoader in the module
namespace after the dict is built has no effect.  patch.dict on _LOADERS is
the correct approach.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.modules.processing import extraction
from app.modules.processing.extraction import (
    CONFIDENCE_THRESHOLD,
    MIN_TEXT_GUARD,
    OCR_THRESHOLD,
    ExtractionResult,
    OcrResult,
    _extract_doc,
    _needs_chinese_fallback,
    extract_text,
)


class TestNormalExtraction:
    def test_pdf_with_sufficient_text_returned_without_ocr(self, tmp_path):
        pdf = tmp_path / "text.pdf"
        pdf.write_bytes(b"fake")
        long_text = "A" * (OCR_THRESHOLD + 1)

        mock_cls = MagicMock()
        mock_cls.return_value.load.return_value = [MagicMock(page_content=long_text)]

        with patch.dict(extraction._LOADERS, {"pdf": mock_cls}):
            result = extract_text(pdf, "pdf")

        assert isinstance(result, ExtractionResult)
        assert result.ocr_applied is False
        assert result.text == long_text

    def test_docx_extraction_returns_text(self, tmp_path):
        docx = tmp_path / "doc.docx"
        docx.write_bytes(b"fake")
        content = "Some docx content here."

        mock_cls = MagicMock()
        mock_cls.return_value.load.return_value = [MagicMock(page_content=content)]

        with patch.dict(extraction._LOADERS, {"docx": mock_cls}):
            result = extract_text(docx, "docx")

        assert result.ocr_applied is False
        assert result.text == content

    def test_unsupported_file_type_raises_key_error(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_bytes(b"fake")
        with pytest.raises(KeyError, match="xyz"):
            extract_text(f, "xyz")


class TestOcrFallback:
    def test_ocr_triggered_when_text_layer_below_threshold(self, tmp_path):
        pdf = tmp_path / "scanned.pdf"
        pdf.write_bytes(b"fake")
        ocr_text = "B" * (OCR_THRESHOLD + 10)

        mock_cls = MagicMock()
        mock_cls.return_value.load.return_value = [MagicMock(page_content="hi")]

        with (
            patch.dict(extraction._LOADERS, {"pdf": mock_cls}),
            patch("app.modules.processing.extraction._extract_pdf_with_ocr") as mock_ocr,
        ):
            mock_ocr.return_value = ocr_text
            result = extract_text(pdf, "pdf")

        assert result.ocr_applied is True
        assert result.text == ocr_text

    def test_ocr_not_triggered_when_text_meets_threshold(self, tmp_path):
        pdf = tmp_path / "text.pdf"
        pdf.write_bytes(b"fake")
        enough_text = "C" * OCR_THRESHOLD

        mock_cls = MagicMock()
        mock_cls.return_value.load.return_value = [MagicMock(page_content=enough_text)]

        with (
            patch.dict(extraction._LOADERS, {"pdf": mock_cls}),
            patch("app.modules.processing.extraction._extract_pdf_with_ocr") as mock_ocr,
        ):
            result = extract_text(pdf, "pdf")
            mock_ocr.assert_not_called()

        assert result.ocr_applied is False


class TestFailureHandling:
    def test_raises_when_ocr_returns_only_whitespace(self, tmp_path):
        pdf = tmp_path / "unreadable.pdf"
        pdf.write_bytes(b"fake")

        mock_cls = MagicMock()
        mock_cls.return_value.load.return_value = [MagicMock(page_content="")]

        with (
            patch.dict(extraction._LOADERS, {"pdf": mock_cls}),
            patch("app.modules.processing.extraction._extract_pdf_with_ocr") as mock_ocr,
        ):
            mock_ocr.return_value = "   \n  "

            with pytest.raises(RuntimeError, match="No text could be extracted"):
                extract_text(pdf, "pdf")

    def test_raises_when_docx_yields_no_text(self, tmp_path):
        docx = tmp_path / "empty.docx"
        docx.write_bytes(b"fake")

        mock_cls = MagicMock()
        mock_cls.return_value.load.return_value = [MagicMock(page_content="")]

        with patch.dict(extraction._LOADERS, {"docx": mock_cls}):
            with pytest.raises(RuntimeError, match="No text could be extracted"):
                extract_text(docx, "docx")


class TestDocExtraction:
    def test_raises_when_libreoffice_not_on_path(self, tmp_path):
        doc = tmp_path / "legacy.doc"
        doc.write_bytes(b"fake")

        with patch("app.modules.processing.extraction.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="soffice"):
                _extract_doc(doc)

    def test_raises_when_libreoffice_conversion_fails(self, tmp_path):
        doc = tmp_path / "broken.doc"
        doc.write_bytes(b"fake")

        with (
            patch("app.modules.processing.extraction.shutil.which", return_value="/usr/bin/soffice"),
            patch("app.modules.processing.extraction.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr=b"conversion error")
            with pytest.raises(RuntimeError, match="LibreOffice failed"):
                _extract_doc(doc)

    def test_doc_file_type_routed_to_extract_doc(self, tmp_path):
        doc = tmp_path / "legacy.doc"
        doc.write_bytes(b"fake")

        with patch("app.modules.processing.extraction._extract_doc") as mock_extract:
            mock_extract.return_value = "Content extracted from legacy doc."
            result = extract_text(doc, "doc")

        mock_extract.assert_called_once_with(doc)
        assert result.text == "Content extracted from legacy doc."
        assert result.ocr_applied is False


class TestOcrResult:
    def test_avg_confidence_with_values(self):
        r = OcrResult(text="hello", confidences=[0.8, 0.6, 0.9])
        assert r.avg_confidence == pytest.approx(0.7667, abs=0.001)

    def test_avg_confidence_empty(self):
        r = OcrResult(text="", confidences=[])
        assert r.avg_confidence == 0.0


class TestNeedsChineseFallback:
    """Unit tests for the confidence-based fallback decision."""

    def test_high_confidence_long_text_skips_chinese(self):
        latin = OcrResult(text="A" * MIN_TEXT_GUARD, confidences=[0.9, 0.85, 0.88])
        assert _needs_chinese_fallback(latin) is False

    def test_low_confidence_triggers_chinese(self):
        latin = OcrResult(text="A" * 100, confidences=[0.2, 0.3, 0.1])
        assert _needs_chinese_fallback(latin) is True

    def test_short_text_triggers_chinese_even_with_high_confidence(self):
        latin = OcrResult(text="Hi", confidences=[0.95])
        assert _needs_chinese_fallback(latin) is True

    def test_no_detections_triggers_chinese(self):
        latin = OcrResult(text="", confidences=[])
        assert _needs_chinese_fallback(latin) is True

    def test_confidence_exactly_at_threshold_skips_chinese(self):
        latin = OcrResult(text="A" * MIN_TEXT_GUARD, confidences=[CONFIDENCE_THRESHOLD])
        assert _needs_chinese_fallback(latin) is False


class TestConfidenceBasedOcrFallback:
    """Integration-level tests for _extract_pdf_with_ocr with mocked readers."""

    def test_latin_high_confidence_does_not_load_chinese(self):
        latin_detections = [([0, 0, 1, 1], "Rechnung vom Mai", 0.92)]

        page = MagicMock()
        page.get_pixmap.return_value.tobytes.return_value = b"fake-png"
        doc = MagicMock()
        doc.__iter__ = lambda self: iter([page])

        with (
            patch("app.modules.processing.extraction.fitz.open", return_value=doc),
            patch("app.modules.processing.extraction._get_latin_reader") as mock_latin,
            patch("app.modules.processing.extraction._get_chinese_reader") as mock_chinese,
        ):
            mock_latin.return_value.readtext.return_value = latin_detections
            result = extraction._extract_pdf_with_ocr(Path("fake.pdf"))

        assert result == "Rechnung vom Mai"
        mock_chinese.assert_not_called()

    def test_latin_low_confidence_triggers_chinese_and_keeps_longer(self):
        latin_detections = [([0, 0, 1, 1], "??", 0.15)]
        chinese_detections = [([0, 0, 1, 1], "这是一份中文文件的内容", 0.88)]

        page = MagicMock()
        page.get_pixmap.return_value.tobytes.return_value = b"fake-png"
        doc = MagicMock()
        doc.__iter__ = lambda self: iter([page])

        with (
            patch("app.modules.processing.extraction.fitz.open", return_value=doc),
            patch("app.modules.processing.extraction._get_latin_reader") as mock_latin,
            patch("app.modules.processing.extraction._get_chinese_reader") as mock_chinese,
        ):
            mock_latin.return_value.readtext.return_value = latin_detections
            mock_chinese.return_value.readtext.return_value = chinese_detections
            result = extraction._extract_pdf_with_ocr(Path("fake.pdf"))

        assert result == "这是一份中文文件的内容"
        mock_chinese.assert_called_once()

    def test_latin_low_confidence_keeps_latin_when_longer(self):
        latin_detections = [([0, 0, 1, 1], "Some garbled text here", 0.3)]
        chinese_detections = [([0, 0, 1, 1], "短", 0.4)]

        page = MagicMock()
        page.get_pixmap.return_value.tobytes.return_value = b"fake-png"
        doc = MagicMock()
        doc.__iter__ = lambda self: iter([page])

        with (
            patch("app.modules.processing.extraction.fitz.open", return_value=doc),
            patch("app.modules.processing.extraction._get_latin_reader") as mock_latin,
            patch("app.modules.processing.extraction._get_chinese_reader") as mock_chinese,
        ):
            mock_latin.return_value.readtext.return_value = latin_detections
            mock_chinese.return_value.readtext.return_value = chinese_detections
            result = extraction._extract_pdf_with_ocr(Path("fake.pdf"))

        assert result == "Some garbled text here"
