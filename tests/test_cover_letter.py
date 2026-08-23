"""Tests for the cover letter agent wrapper (mocked - no real LLM calls)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hr_breaker.agents.cover_letter import SYSTEM_PROMPT, CoverLetter, write_cover_letter
from hr_breaker.models import JobPosting, OptimizedResume


def test_system_prompt_forbids_subject_line_and_greeting():
    """Regression test for a real observed defect: the model prefixed letters with
    a subject-like line ("Отклик на вакансию") and a stray "ping" opener."""
    assert "subject" in SYSTEM_PROMPT.lower()
    assert "greeting" in SYSTEM_PROMPT.lower()
    assert "ping" in SYSTEM_PROMPT.lower()  # explicitly called out as a forbidden example


@pytest.mark.asyncio
async def test_write_cover_letter_returns_stripped_text():
    optimized = OptimizedResume(html="<p>Jane Doe, Python dev</p>", source_checksum="abc", pdf_text="Jane Doe\nPython dev")
    job = JobPosting(title="Python Developer", company="Acme", requirements=["Python"], keywords=["python"])

    mock_result = MagicMock()
    mock_result.output = CoverLetter(text="  Dear Acme, I'd love to join.  ")

    with patch("hr_breaker.agents.cover_letter.get_cover_letter_agent") as mock_get_agent:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_get_agent.return_value = mock_agent

        text = await write_cover_letter(optimized, job)

    assert text == "Dear Acme, I'd love to join."
    mock_agent.run.assert_awaited_once()
    prompt = mock_agent.run.call_args[0][0]
    assert "Python Developer" in prompt
    assert "Acme" in prompt


@pytest.mark.asyncio
async def test_write_cover_letter_falls_back_to_job_language_when_none_given():
    optimized = OptimizedResume(html="<p>x</p>", source_checksum="abc", pdf_text="x")
    job = JobPosting(title="Dev", company="Acme", language_code="ru")

    mock_result = MagicMock()
    mock_result.output = CoverLetter(text="Здравствуйте")

    with patch("hr_breaker.agents.cover_letter.get_cover_letter_agent") as mock_get_agent:
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_get_agent.return_value = mock_agent

        text = await write_cover_letter(optimized, job)

    assert text == "Здравствуйте"
    prompt = mock_agent.run.call_args[0][0]
    assert "Russian (ru)" in prompt
