"""Tests for PDF storage filename parsing."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from hr_breaker.services.pdf_storage import PDFStorage, sanitize_filename


class TestListAllWithLanguageSuffix:
    def _make_pdf(self, tmp_path: Path, name: str) -> Path:
        p = tmp_path / name
        p.write_bytes(b"%PDF-fake")
        return p

    def test_parse_filename_with_lang_suffix(self, tmp_path):
        """john_doe_acme_engineer_en.pdf should parse correctly."""
        self._make_pdf(tmp_path, "john_doe_acme_engineer_en.pdf")
        with patch("hr_breaker.services.pdf_storage.get_settings") as m:
            m.return_value = MagicMock(output_dir=tmp_path)
            storage = PDFStorage()
            records = storage.list_all()
        assert len(records) == 1
        r = records[0]
        assert r.first_name == "John"
        assert r.last_name == "Doe"
        assert "Acme" in r.company
        assert "Engineer" in r.job_title
        # Lang code should NOT be the entire job_title
        assert r.job_title != "En"

    def test_parse_filename_with_ru_suffix(self, tmp_path):
        """john_doe_acme_engineer_ru.pdf should parse the same metadata."""
        self._make_pdf(tmp_path, "john_doe_acme_engineer_ru.pdf")
        with patch("hr_breaker.services.pdf_storage.get_settings") as m:
            m.return_value = MagicMock(output_dir=tmp_path)
            storage = PDFStorage()
            records = storage.list_all()
        assert len(records) == 1
        r = records[0]
        assert r.first_name == "John"
        assert r.last_name == "Doe"
        assert "Engineer" in r.job_title

    def test_parse_old_filename_without_lang_suffix(self, tmp_path):
        """Backward compat: john_doe_acme_engineer.pdf (no lang) still works."""
        self._make_pdf(tmp_path, "john_doe_acme_engineer.pdf")
        with patch("hr_breaker.services.pdf_storage.get_settings") as m:
            m.return_value = MagicMock(output_dir=tmp_path)
            storage = PDFStorage()
            records = storage.list_all()
        assert len(records) == 1
        r = records[0]
        assert r.first_name == "John"
        assert r.last_name == "Doe"
