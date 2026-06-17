"""Unit tests for processing/extraction.py.

LibreOffice is mocked so these tests run without any system-level installation.
EasyOCR/PyMuPDF OCR fallback is tested by patching _extract_pdf_with_ocr directly,
which avoids loading the ~200 MB EasyOCR model in a unit-test context.

Loader mocking note: _LOADERS is populated at import time with references to
the actual loader classes.  Patching the *name* PyPDFLoader in the module
namespace after the dict is built has no effect.  patch.dict on _LOADERS is
the correct approach.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.modules.processing import extraction
from app.modules.processing.extraction import (
    OCR_THRESHOLD,
    ExtractionResult,
    _extract_doc,
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
