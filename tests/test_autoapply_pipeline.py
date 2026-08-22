"""End-to-end wiring test for run_autoapply, with hh.ru/LLM calls mocked out."""

from unittest.mock import AsyncMock, patch

import pytest

from hr_breaker.autoapply.hh_client import Vacancy
from hr_breaker.autoapply.pipeline import run_autoapply
from hr_breaker.autoapply.state_store import AutoApplyStore
from hr_breaker.models import JobPosting, OptimizedResume, ResumeSource
from hr_breaker.models.feedback import ValidationResult


def _make_vacancy(vid="v1"):
    return Vacancy(
        id=vid, name="Python Developer", employer_name="Acme",
        url=f"https://hh.ru/vacancy/{vid}", description="Build things",
        key_skills=["Python"], area_name="Moscow", raw={"id": vid},
    )


@pytest.mark.asyncio
async def test_dry_run_tailors_but_does_not_apply(tmp_path):
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    output_dir = tmp_path / "pdfs"
    source = ResumeSource(content="Jane Doe\nPython developer")
    optimized = OptimizedResume(html="<p>tailored</p>", source_checksum=source.checksum, pdf_bytes=b"%PDF-fake", pdf_text="tailored resume text")

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock(return_value=(optimized, ValidationResult(results=[]), None))), \
         patch("hr_breaker.autoapply.pipeline.write_cover_letter", new=AsyncMock(return_value="Dear Acme...")):
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[_make_vacancy("v1")])
        mock_hh.apply_to_vacancy = AsyncMock()

        summary = await run_autoapply(
            triggers=["python"],
            resume_source=source,
            store=store,
            output_dir=output_dir,
            live=False,
        )

    assert summary.found == 1
    assert summary.tailored == 1
    assert summary.applied == 0
    assert summary.dry_run is True
    mock_hh.apply_to_vacancy.assert_not_awaited()

    row = store.get("v1")
    assert row["status"] == "ready"
    assert row["cover_letter"] == "Dear Acme..."
    assert (output_dir / row["pdf_path"].split("/")[-1]).exists()


@pytest.mark.asyncio
async def test_live_run_applies_and_respects_max_apply_cap(tmp_path):
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    source = ResumeSource(content="Jane Doe\nPython developer")
    optimized = OptimizedResume(html="<p>tailored</p>", source_checksum=source.checksum, pdf_bytes=b"%PDF-fake", pdf_text="tailored resume text")

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock(return_value=(optimized, ValidationResult(results=[]), None))), \
         patch("hr_breaker.autoapply.pipeline.write_cover_letter", new=AsyncMock(return_value="Dear Acme...")):
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[_make_vacancy("v1"), _make_vacancy("v2")])
        mock_hh.apply_to_vacancy = AsyncMock()

        summary = await run_autoapply(
            triggers=["python"],
            resume_source=source,
            store=store,
            output_dir=tmp_path / "pdfs",
            live=True,
            access_token="tok",
            hh_resume_id="resume-1",
            max_apply_per_run=1,
        )

    assert summary.tailored == 2
    assert summary.applied == 1
    assert summary.skipped_apply_cap == 1
    assert mock_hh.apply_to_vacancy.await_count == 1

    statuses = {store.get("v1")["status"], store.get("v2")["status"]}
    assert statuses == {"applied", "ready"}


@pytest.mark.asyncio
async def test_already_seen_vacancies_are_skipped(tmp_path):
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert("v1", "applied")
    source = ResumeSource(content="Jane Doe\nPython developer")

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock()) as mock_optimize:
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[_make_vacancy("v1")])

        summary = await run_autoapply(
            triggers=["python"], resume_source=source, store=store,
            output_dir=tmp_path / "pdfs", live=False,
        )

    assert summary.already_seen == 1
    assert summary.tailored == 0
    mock_optimize.assert_not_awaited()
