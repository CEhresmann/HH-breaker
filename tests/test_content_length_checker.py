"""Tests for ContentLengthChecker's render reuse/fallback behavior."""

from unittest.mock import MagicMock, patch

import pytest

from hr_breaker.filters.content_length import ContentLengthChecker
from hr_breaker.models import JobPosting, OptimizedResume, ResumeSource


@pytest.fixture
def source():
    return ResumeSource(content="John Doe\nPython dev")


@pytest.fixture
def job():
    return JobPosting(title="Backend Engineer", company="Acme", requirements=["Python"], keywords=["python"])


@pytest.mark.asyncio
async def test_reuses_prerendered_pdf_when_present(job, source):
    optimized = OptimizedResume(
        html="<div>Test</div>", source_checksum=source.checksum,
        pdf_bytes=b"%PDF-fake", page_count=1,
    )

    with patch("hr_breaker.filters.content_length.get_renderer") as mock_get_renderer:
        result = await ContentLengthChecker().evaluate(optimized, job, source)

    mock_get_renderer.assert_not_called()
    assert result.passed


@pytest.mark.asyncio
async def test_falls_back_to_render_when_page_count_missing(job, source):
    optimized = OptimizedResume(
        html="<div>Test</div>", source_checksum=source.checksum,
        pdf_bytes=b"%PDF-fake", page_count=None,
    )

    mock_render_result = MagicMock(page_count=1, pdf_bytes=b"%PDF-rendered")
    with patch("hr_breaker.filters.content_length.get_renderer") as mock_get_renderer:
        mock_get_renderer.return_value.render.return_value = mock_render_result
        result = await ContentLengthChecker().evaluate(optimized, job, source)

    mock_get_renderer.return_value.render.assert_called_once_with(optimized.html)
    assert result.passed
