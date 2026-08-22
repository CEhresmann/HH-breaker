"""Tests for data models."""

from hr_breaker.models import (
    FilterResult,
    JobPosting,
    OptimizedResume,
    ResumeSource,
    ValidationResult,
)
from hr_breaker.models.resume_data import (
    ResumeData,
    ContactInfo,
    Experience,
    Education,
)


# --- ResumeSource Tests ---


def test_resume_source_checksum():
    source = ResumeSource(content="\\documentclass{article}")
    assert len(source.checksum) == 64  # SHA256 hex


def test_resume_source_checksum_stable():
    content = "test content"
    source1 = ResumeSource(content=content)
    source2 = ResumeSource(content=content)
    assert source1.checksum == source2.checksum


def test_resume_source_legacy_latex_field():
    """Test backward compatibility with old 'latex' field name."""
    # Old cache files have 'latex' instead of 'content'
    data = {"latex": "test content", "first_name": "John"}
    source = ResumeSource.model_validate(data)
    assert source.content == "test content"
    assert source.first_name == "John"


def test_resume_source_latex_property():
    """Test that .latex property returns content for backward compat."""
    source = ResumeSource(content="my resume")
    assert source.latex == "my resume"


def test_resume_source_content_field():
    """Test using new content field directly."""
    source = ResumeSource(content="# My Resume\n\nMarkdown content")
    assert source.content == "# My Resume\n\nMarkdown content"


# --- JobPosting Tests ---


def test_job_posting():
    job = JobPosting(
        title="Software Engineer",
        company="Acme Corp",
        requirements=["Python", "SQL"],
        keywords=["python", "postgresql"],
    )
    assert job.title == "Software Engineer"
    assert len(job.keywords) == 2


# --- FilterResult Tests ---


def test_filter_result():
    result = FilterResult(
        filter_name="test",
        passed=True,
        score=0.9,
        issues=[],
        suggestions=[],
    )
    assert result.passed
    assert result.score == 0.9


def test_filter_result_with_threshold():
    result = FilterResult(
        filter_name="test",
        passed=True,
        score=0.8,
        threshold=0.7,
        issues=[],
        suggestions=[],
    )
    assert result.threshold == 0.7


# --- ValidationResult Tests ---


def test_validation_result_passed():
    results = [
        FilterResult(filter_name="a", passed=True, score=0.9),
        FilterResult(filter_name="b", passed=True, score=0.8),
    ]
    validation = ValidationResult(results=results)
    assert validation.passed


def test_validation_result_failed():
    results = [
        FilterResult(filter_name="a", passed=True, score=0.9),
        FilterResult(filter_name="b", passed=False, score=0.4),
    ]
    validation = ValidationResult(results=results)
    assert not validation.passed


def test_validation_result_empty():
    validation = ValidationResult(results=[])
    assert validation.passed  # No failures = passed


# --- OptimizedResume Tests ---


def test_optimized_resume():
    data = ResumeData(
        contact=ContactInfo(name="John Doe", email="john@example.com"),
        skills=["Python", "SQL"],
    )
    optimized = OptimizedResume(
        data=data,
        iteration=1,
        changes=["Updated skills section"],
        source_checksum="abc123",
    )
    assert optimized.iteration == 1
    assert len(optimized.changes) == 1
    assert optimized.data.contact.name == "John Doe"


def test_optimized_resume_with_pdf():
    data = ResumeData(
        contact=ContactInfo(name="Jane", email="jane@example.com"),
    )
    optimized = OptimizedResume(
        data=data,
        source_checksum="def456",
        pdf_bytes=b"%PDF-1.4 fake",
        pdf_text="Jane\njane@example.com",
    )
    assert optimized.pdf_bytes is not None
    assert "Jane" in optimized.pdf_text


def test_optimized_resume_defaults():
    data = ResumeData(
        contact=ContactInfo(name="Test", email="test@example.com"),
    )
    optimized = OptimizedResume(data=data, source_checksum="xyz")
    assert optimized.iteration == 0
    assert optimized.changes == []
    assert optimized.pdf_bytes is None
    assert optimized.pdf_text is None
