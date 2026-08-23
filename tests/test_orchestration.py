"""Tests for orchestration module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from hr_breaker.models import (
    FilterResult,
    JobPosting,
    OptimizedResume,
    ResumeSource,
    ValidationResult,
)
from hr_breaker.orchestration import run_filters


@pytest.fixture
def source_resume():
    return ResumeSource(content="Test resume content")


@pytest.fixture
def job_posting():
    return JobPosting(
        title="Engineer",
        company="Test Corp",
        requirements=["Python"],
        keywords=["python"],
    )


@pytest.fixture
def optimized_resume(source_resume):
    return OptimizedResume(
        html="<div>Test</div>",
        source_checksum=source_resume.checksum,
        pdf_text="Test resume text",
    )


class TestRunFiltersParallel:
    @pytest.mark.asyncio
    async def test_parallel_handles_filter_exception(
        self, source_resume, job_posting, optimized_resume
    ):
        """If one filter raises, others should still return results."""
        good_result = FilterResult(
            filter_name="GoodFilter",
            passed=True,
            score=0.9,
            threshold=0.5,
            issues=[],
            suggestions=[],
        )

        class GoodFilter:
            name = "GoodFilter"
            priority = 1

            def __init__(self, **kwargs):
                pass

            async def evaluate(self, *args, **kwargs):
                return good_result

        class BadFilter:
            name = "BadFilter"
            priority = 2

            def __init__(self, **kwargs):
                pass

            async def evaluate(self, *args, **kwargs):
                raise RuntimeError("Filter crashed!")

        # Mock registry to return our test filters
        with patch("hr_breaker.orchestration.FilterRegistry.all") as mock_all:
            mock_all.return_value = [GoodFilter, BadFilter]

            # Should NOT raise, should return partial results
            validation = await run_filters(
                optimized_resume, job_posting, source_resume, parallel=True
            )

            # Should have results from working filter + error for crashed one
            assert len(validation.results) >= 1
            # The validation should not have passed (one filter crashed)
            # At minimum, the good result should be present
            good_results = [r for r in validation.results if r.filter_name == "GoodFilter"]
            assert len(good_results) == 1
            assert good_results[0].passed


class TestRunFiltersParallelTiered:
    @pytest.mark.asyncio
    async def test_remote_tier_fires_when_local_tier_passes(
        self, source_resume, job_posting, optimized_resume
    ):
        remote_calls = []

        class LocalFilter:
            name = "LocalFilter"
            priority = 1
            is_local = True

            def __init__(self, **kwargs):
                pass

            async def evaluate(self, *args, **kwargs):
                return FilterResult(filter_name="LocalFilter", passed=True, score=1.0, threshold=0.5)

        class RemoteFilter:
            name = "RemoteFilter"
            priority = 5
            is_local = False

            def __init__(self, **kwargs):
                pass

            async def evaluate(self, *args, **kwargs):
                remote_calls.append(1)
                return FilterResult(filter_name="RemoteFilter", passed=True, score=1.0, threshold=0.5)

        with patch("hr_breaker.orchestration.FilterRegistry.all") as mock_all:
            mock_all.return_value = [LocalFilter, RemoteFilter]

            validation = await run_filters(
                optimized_resume, job_posting, source_resume, parallel=True
            )

        assert len(remote_calls) == 1
        remote_results = [r for r in validation.results if r.filter_name == "RemoteFilter"]
        assert len(remote_results) == 1
        assert remote_results[0].skipped is False
        assert remote_results[0].passed

    @pytest.mark.asyncio
    async def test_remote_tier_skipped_when_local_tier_fails(
        self, source_resume, job_posting, optimized_resume
    ):
        remote_calls = []

        class LocalFilter:
            name = "LocalFilter"
            priority = 1
            is_local = True
            threshold = 1.0

            def __init__(self, **kwargs):
                pass

            async def evaluate(self, *args, **kwargs):
                return FilterResult(filter_name="LocalFilter", passed=False, score=0.0, threshold=1.0)

        class RemoteFilter:
            name = "RemoteFilter"
            priority = 5
            is_local = False
            threshold = 0.7

            def __init__(self, **kwargs):
                pass

            async def evaluate(self, *args, **kwargs):
                remote_calls.append(1)
                return FilterResult(filter_name="RemoteFilter", passed=True, score=1.0, threshold=0.7)

        with patch("hr_breaker.orchestration.FilterRegistry.all") as mock_all:
            mock_all.return_value = [LocalFilter, RemoteFilter]

            validation = await run_filters(
                optimized_resume, job_posting, source_resume, parallel=True
            )

        assert len(remote_calls) == 0  # never invoked - skipped instead

        remote_results = [r for r in validation.results if r.filter_name == "RemoteFilter"]
        assert len(remote_results) == 1
        assert remote_results[0].skipped is True
        assert remote_results[0].threshold == 0.7

        # The skipped remote filter reporting passed=True must not mask the local failure.
        assert validation.passed is False


