"""Tests for combined reviewer agent and LLMChecker filter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from hr_breaker.agents.combined_reviewer import (
    SCORE_WEIGHTS,
    CombinedReviewResult,
    compute_ats_score,
    pdf_to_image,
)
from hr_breaker.filters import LLMChecker, FilterRegistry
from hr_breaker.models import JobPosting, OptimizedResume, ResumeSource


@pytest.fixture
def source_resume():
    return ResumeSource(content="# John Doe\nSoftware Engineer")


@pytest.fixture
def job_posting():
    return JobPosting(
        title="Backend Engineer",
        company="Acme Corp",
        description="Looking for a backend engineer with Python experience.",
        requirements=["Python", "Django", "PostgreSQL"],
        keywords=["python", "django", "postgresql", "rest", "api"],
    )


@pytest.fixture
def optimized_resume(source_resume):
    return OptimizedResume(
        html='<header class="header"><h1 class="name">Test</h1></header>',
        source_checksum=source_resume.checksum,
    )


class TestCombinedReviewResult:
    def test_valid_result(self):
        result = CombinedReviewResult(
            looks_professional=True,
            visual_issues=[],
            visual_feedback="",
            keyword_score=0.8,
            experience_score=0.7,
            education_score=0.9,
            overall_fit_score=0.75,
            disqualified=False,
            ats_issues=[],
        )
        assert result.looks_professional
        assert result.keyword_score == 0.8

    def test_score_bounds(self):
        with pytest.raises(ValueError):
            CombinedReviewResult(
                looks_professional=True,
                visual_issues=[],
                visual_feedback="",
                keyword_score=1.5,  # Invalid
                experience_score=0.7,
                education_score=0.9,
                overall_fit_score=0.75,
                disqualified=False,
                ats_issues=[],
            )


class TestScoreAggregation:
    def test_weights_sum_to_one(self):
        total = sum(SCORE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_perfect_scores_yield_one(self):
        result = CombinedReviewResult(
            looks_professional=True,
            visual_issues=[],
            visual_feedback="",
            keyword_score=1.0,
            experience_score=1.0,
            education_score=1.0,
            overall_fit_score=1.0,
            disqualified=False,
            ats_issues=[],
        )
        assert abs(compute_ats_score(result) - 1.0) < 0.001

    def test_zero_scores_yield_zero(self):
        result = CombinedReviewResult(
            looks_professional=True,
            visual_issues=[],
            visual_feedback="",
            keyword_score=0.0,
            experience_score=0.0,
            education_score=0.0,
            overall_fit_score=0.0,
            disqualified=False,
            ats_issues=[],
        )
        assert compute_ats_score(result) == 0.0


class TestLLMChecker:
    def test_registered(self):
        assert "LLMChecker" in FilterRegistry.names()

    def test_priority(self):
        checker = LLMChecker()
        assert checker.priority == 5

    def test_threshold(self):
        checker = LLMChecker()
        assert checker.threshold == 0.7

    @pytest.mark.asyncio
    async def test_passes_when_professional_and_high_score(
        self, source_resume, job_posting, optimized_resume
    ):
        mock_result = CombinedReviewResult(
            looks_professional=True,
            visual_issues=[],
            visual_feedback="",
            keyword_score=0.9,
            experience_score=0.8,
            education_score=0.85,
            overall_fit_score=0.8,
            disqualified=False,
            ats_issues=[],
        )

        with patch("hr_breaker.filters.llm_checker.combined_review") as mock_review:
            mock_review.return_value = (mock_result, b"pdf", 1, [])

            checker = LLMChecker()
            result = await checker.evaluate(optimized_resume, job_posting, source_resume)

            assert result.passed
            assert result.score > 0.7

    @pytest.mark.asyncio
    async def test_fails_when_not_professional(
        self, source_resume, job_posting, optimized_resume
    ):
        mock_result = CombinedReviewResult(
            looks_professional=False,
            visual_issues=["Text overlapping"],
            visual_feedback="Fix the overlap",
            keyword_score=0.9,
            experience_score=0.8,
            education_score=0.85,
            overall_fit_score=0.8,
            disqualified=False,
            ats_issues=[],
        )

        with patch("hr_breaker.filters.llm_checker.combined_review") as mock_review:
            mock_review.return_value = (mock_result, b"pdf", 1, [])

            checker = LLMChecker()
            result = await checker.evaluate(optimized_resume, job_posting, source_resume)

            assert not result.passed
            assert result.score == 0.0
            assert "Text overlapping" in result.issues

    @pytest.mark.asyncio
    async def test_fails_when_low_ats_score(
        self, source_resume, job_posting, optimized_resume
    ):
        mock_result = CombinedReviewResult(
            looks_professional=True,
            visual_issues=[],
            visual_feedback="",
            keyword_score=0.3,
            experience_score=0.4,
            education_score=0.5,
            overall_fit_score=0.4,
            disqualified=False,
            ats_issues=["Missing required skills"],
        )

        with patch("hr_breaker.filters.llm_checker.combined_review") as mock_review:
            mock_review.return_value = (mock_result, b"pdf", 1, [])

            checker = LLMChecker()
            result = await checker.evaluate(optimized_resume, job_posting, source_resume)

            assert not result.passed
            assert result.score < 0.7

    @pytest.mark.asyncio
    async def test_fails_when_disqualified(
        self, source_resume, job_posting, optimized_resume
    ):
        mock_result = CombinedReviewResult(
            looks_professional=True,
            visual_issues=[],
            visual_feedback="",
            keyword_score=0.9,
            experience_score=0.9,
            education_score=0.9,
            overall_fit_score=0.9,
            disqualified=True,
            ats_issues=["Missing required degree"],
        )

        with patch("hr_breaker.filters.llm_checker.combined_review") as mock_review:
            mock_review.return_value = (mock_result, b"pdf", 1, [])

            checker = LLMChecker()
            result = await checker.evaluate(optimized_resume, job_posting, source_resume)

            assert not result.passed
            assert result.score > 0.7  # High score but still fails
            assert "Missing required degree" in result.issues

    @pytest.mark.asyncio
    async def test_includes_render_warnings(
        self, source_resume, job_posting, optimized_resume
    ):
        mock_result = CombinedReviewResult(
            looks_professional=True,
            visual_issues=[],
            visual_feedback="",
            keyword_score=0.9,
            experience_score=0.9,
            education_score=0.9,
            overall_fit_score=0.9,
            disqualified=False,
            ats_issues=[],
        )

        with patch("hr_breaker.filters.llm_checker.combined_review") as mock_review:
            mock_review.return_value = (mock_result, b"pdf", 1, ["Content overflow warning"])

            checker = LLMChecker()
            result = await checker.evaluate(optimized_resume, job_posting, source_resume)

            assert any("Content overflow" in issue for issue in result.issues)


class TestPdfToImage:
    def test_closes_fitz_document(self):
        """pdf_to_image should close the fitz doc to avoid resource leak."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"png_bytes"
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("hr_breaker.agents.combined_reviewer.fitz.open", return_value=mock_doc):
            pdf_to_image(b"fake pdf bytes")
            mock_doc.close.assert_called_once()
