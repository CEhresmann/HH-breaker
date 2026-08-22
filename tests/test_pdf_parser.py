"""Tests for PDF text extraction."""

import tempfile
from pathlib import Path

import pytest

from hr_breaker.services.pdf_parser import extract_text_from_pdf


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a simple PDF using pymupdf for testing."""
    import fitz

    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "John Doe\nPython Developer\nExperience with Django")
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def test_extract_text_from_pdf(sample_pdf):
    text = extract_text_from_pdf(sample_pdf)
    assert "John Doe" in text
    assert "Python Developer" in text
    assert "Django" in text


def test_extract_text_multipage(tmp_path):
    """Test extraction from multi-page PDF."""
    import fitz

    pdf_path = tmp_path / "multipage.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((50, 50), "Page 1 content")

    page2 = doc.new_page()
    page2.insert_text((50, 50), "Page 2 content")

    doc.save(pdf_path)
    doc.close()

    text = extract_text_from_pdf(pdf_path)
    assert "Page 1 content" in text
    assert "Page 2 content" in text
