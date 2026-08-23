"""End-to-end wiring test for run_autoapply, with hh.ru/LLM calls mocked out."""

from unittest.mock import AsyncMock, patch

import pytest

from hr_breaker.autoapply.browser_apply import ApplyOutcome
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


class _FakeBrowserApplier:
    """Stand-in async context manager for BrowserApplier - avoids launching Playwright."""

    def __init__(self, apply_mock):
        self._apply_mock = apply_mock

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def apply(self, vacancy_id, cover_letter, resume_title=None):
        return await self._apply_mock(vacancy_id, cover_letter, resume_title=resume_title)


@pytest.mark.asyncio
async def test_dry_run_tailors_but_does_not_apply(tmp_path):
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    output_dir = tmp_path / "pdfs"
    source = ResumeSource(content="Jane Doe\nPython developer")
    optimized = OptimizedResume(html="<p>tailored</p>", source_checksum=source.checksum, pdf_bytes=b"%PDF-fake", pdf_text="tailored resume text")

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock(return_value=(optimized, ValidationResult(results=[]), None))), \
         patch("hr_breaker.autoapply.pipeline.write_cover_letter", new=AsyncMock(return_value="Dear Acme...")), \
         patch("hr_breaker.autoapply.pipeline.asyncio.sleep", new=AsyncMock()):
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[_make_vacancy("v1")])
        mock_hh.get_vacancy_detail = AsyncMock(return_value=_make_vacancy("v1"))
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

    apply_mock = AsyncMock(return_value=ApplyOutcome("applied"))

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.BrowserApplier", return_value=_FakeBrowserApplier(apply_mock)), \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock(return_value=(optimized, ValidationResult(results=[]), None))), \
         patch("hr_breaker.autoapply.pipeline.write_cover_letter", new=AsyncMock(return_value="Dear Acme...")), \
         patch("hr_breaker.autoapply.pipeline.asyncio.sleep", new=AsyncMock()):
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[_make_vacancy("v1"), _make_vacancy("v2")])
        mock_hh.get_vacancy_detail = AsyncMock(side_effect=lambda vid: _make_vacancy(vid))

        summary = await run_autoapply(
            triggers=["python"],
            resume_source=source,
            store=store,
            output_dir=tmp_path / "pdfs",
            live=True,
            max_apply_per_run=1,
        )

    assert summary.tailored == 2
    assert summary.applied == 1
    assert summary.skipped_apply_cap == 1
    assert apply_mock.await_count == 1

    statuses = {store.get("v1")["status"], store.get("v2")["status"]}
    assert statuses == {"applied", "ready"}


@pytest.mark.asyncio
async def test_live_run_marks_already_applied_as_skipped_not_failed(tmp_path):
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    source = ResumeSource(content="Jane Doe\nPython developer")
    optimized = OptimizedResume(html="<p>tailored</p>", source_checksum=source.checksum, pdf_bytes=b"%PDF-fake", pdf_text="tailored resume text")
    apply_mock = AsyncMock(return_value=ApplyOutcome("already_applied", "apply link not found"))

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.BrowserApplier", return_value=_FakeBrowserApplier(apply_mock)), \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock(return_value=(optimized, ValidationResult(results=[]), None))), \
         patch("hr_breaker.autoapply.pipeline.write_cover_letter", new=AsyncMock(return_value="Dear Acme...")), \
         patch("hr_breaker.autoapply.pipeline.asyncio.sleep", new=AsyncMock()):
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[_make_vacancy("v1")])
        mock_hh.get_vacancy_detail = AsyncMock(return_value=_make_vacancy("v1"))

        summary = await run_autoapply(
            triggers=["python"], resume_source=source, store=store,
            output_dir=tmp_path / "pdfs", live=True,
        )

    assert summary.applied == 0
    assert summary.failed == 0
    assert store.get("v1")["status"] == "skipped"


@pytest.mark.asyncio
async def test_live_run_records_browser_apply_error_as_failed(tmp_path):
    from hr_breaker.autoapply.browser_apply import BrowserApplyError

    store = AutoApplyStore(tmp_path / "state.sqlite3")
    source = ResumeSource(content="Jane Doe\nPython developer")
    optimized = OptimizedResume(html="<p>tailored</p>", source_checksum=source.checksum, pdf_bytes=b"%PDF-fake", pdf_text="tailored resume text")
    apply_mock = AsyncMock(side_effect=BrowserApplyError("no submit button found"))

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.BrowserApplier", return_value=_FakeBrowserApplier(apply_mock)), \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock(return_value=(optimized, ValidationResult(results=[]), None))), \
         patch("hr_breaker.autoapply.pipeline.write_cover_letter", new=AsyncMock(return_value="Dear Acme...")), \
         patch("hr_breaker.autoapply.pipeline.asyncio.sleep", new=AsyncMock()):
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[_make_vacancy("v1")])
        mock_hh.get_vacancy_detail = AsyncMock(return_value=_make_vacancy("v1"))

        summary = await run_autoapply(
            triggers=["python"], resume_source=source, store=store,
            output_dir=tmp_path / "pdfs", live=True,
        )

    assert summary.applied == 0
    assert summary.failed == 1
    assert store.get("v1")["status"] == "failed"


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


@pytest.mark.asyncio
async def test_bare_seen_vacancies_are_retried_not_skipped(tmp_path):
    """A vacancy left at "seen" (e.g. a prior run was interrupted mid-tailoring) must
    be retried, not treated as already handled."""
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert("v1", "seen")
    source = ResumeSource(content="Jane Doe\nPython developer")
    optimized = OptimizedResume(html="<p>tailored</p>", source_checksum=source.checksum, pdf_bytes=b"%PDF-fake", pdf_text="tailored resume text")

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock(return_value=(optimized, ValidationResult(results=[]), None))), \
         patch("hr_breaker.autoapply.pipeline.write_cover_letter", new=AsyncMock(return_value="Dear Acme...")), \
         patch("hr_breaker.autoapply.pipeline.asyncio.sleep", new=AsyncMock()):
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[_make_vacancy("v1")])
        mock_hh.get_vacancy_detail = AsyncMock(return_value=_make_vacancy("v1"))

        summary = await run_autoapply(
            triggers=["python"], resume_source=source, store=store,
            output_dir=tmp_path / "pdfs", live=False,
        )

    assert summary.already_seen == 0
    assert summary.tailored == 1
    assert store.get("v1")["status"] == "ready"


@pytest.mark.asyncio
async def test_failed_vacancies_are_retried_not_skipped(tmp_path):
    """A vacancy that failed last run (e.g. a transient timeout) must be retried,
    not treated as permanently resolved."""
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert("v1", "failed", error="TimeoutError")
    source = ResumeSource(content="Jane Doe\nPython developer")
    optimized = OptimizedResume(html="<p>tailored</p>", source_checksum=source.checksum, pdf_bytes=b"%PDF-fake", pdf_text="tailored resume text")

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock(return_value=(optimized, ValidationResult(results=[]), None))), \
         patch("hr_breaker.autoapply.pipeline.write_cover_letter", new=AsyncMock(return_value="Dear Acme...")), \
         patch("hr_breaker.autoapply.pipeline.asyncio.sleep", new=AsyncMock()):
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[_make_vacancy("v1")])
        mock_hh.get_vacancy_detail = AsyncMock(return_value=_make_vacancy("v1"))

        summary = await run_autoapply(
            triggers=["python"], resume_source=source, store=store,
            output_dir=tmp_path / "pdfs", live=False,
        )

    assert summary.already_seen == 0
    assert summary.tailored == 1
    assert store.get("v1")["status"] == "ready"


@pytest.mark.asyncio
async def test_search_paginates_until_max_new_or_short_page(tmp_path):
    """A full page (== per_page) with no new candidates must trigger a second search
    page; a short page (< per_page) must stop pagination."""
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert("v1", "applied")
    source = ResumeSource(content="Jane Doe\nPython developer")

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock()) as mock_optimize, \
         patch("hr_breaker.autoapply.pipeline.asyncio.sleep", new=AsyncMock()):
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(side_effect=[[_make_vacancy("v1")], []])

        summary = await run_autoapply(
            triggers=["python"], resume_source=source, store=store,
            output_dir=tmp_path / "pdfs", live=False, per_page=1, max_new=5,
        )

    assert summary.found == 1
    assert summary.already_seen == 1
    assert mock_hh.search_vacancies.await_count == 2
    first_call, second_call = mock_hh.search_vacancies.await_args_list
    assert first_call.kwargs["page"] == 0
    assert second_call.kwargs["page"] == 1
    mock_optimize.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_run_applies_to_previously_ready_vacancy_without_retailoring(tmp_path):
    """A vacancy tailored during an earlier dry run ("ready": cover letter + PDF
    already exist, never applied) must be picked up by a later --live run and
    applied using the saved cover letter - without re-running optimize_for_job."""
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert(
        "v1", "ready", title="Python Developer", company="Acme",
        cover_letter="Dear Acme, already written.", pdf_path="/tmp/v1.pdf",
    )
    apply_mock = AsyncMock(return_value=ApplyOutcome("applied"))

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.BrowserApplier", return_value=_FakeBrowserApplier(apply_mock)), \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock()) as mock_optimize:
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[])

        summary = await run_autoapply(
            triggers=["python"], resume_source=ResumeSource(content="Jane Doe"), store=store,
            output_dir=tmp_path / "pdfs", live=True,
        )

    mock_optimize.assert_not_awaited()  # no re-tailoring
    apply_mock.assert_awaited_once_with("v1", "Dear Acme, already written.", resume_title=None)
    assert summary.applied == 1
    assert store.get("v1")["status"] == "applied"


@pytest.mark.asyncio
async def test_live_run_retries_stale_failed_vacancy_with_existing_cover_letter(tmp_path):
    """A "failed" row from a stale attempt that already produced a cover letter +
    PDF (e.g. the old API-based apply path) must be retried for apply-only too,
    not just "ready" rows - otherwise it's only ever revisited by re-discovering
    it via search and redoing the whole tailoring pass for nothing."""
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert(
        "v1", "failed", title="Backend Engineer", company="Acme",
        cover_letter="Dear Acme, already written.", pdf_path="/tmp/v1.pdf",
        error="Apply to vacancy v1 failed (403): forbidden",
    )
    apply_mock = AsyncMock(return_value=ApplyOutcome("applied"))

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.BrowserApplier", return_value=_FakeBrowserApplier(apply_mock)), \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock()) as mock_optimize:
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[])

        summary = await run_autoapply(
            triggers=["python"], resume_source=ResumeSource(content="Jane Doe"), store=store,
            output_dir=tmp_path / "pdfs", live=True,
        )

    mock_optimize.assert_not_awaited()
    apply_mock.assert_awaited_once_with("v1", "Dear Acme, already written.", resume_title=None)
    assert summary.applied == 1
    assert store.get("v1")["status"] == "applied"


@pytest.mark.asyncio
async def test_live_run_ignores_failed_vacancy_without_cover_letter(tmp_path):
    """A "failed" row where tailoring itself failed (no cover letter/PDF) must NOT
    be picked up by the apply-only retry - it needs full re-tailoring, which only
    happens by re-discovering it via search."""
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert("v1", "failed", error="litellm.InternalServerError")
    apply_mock = AsyncMock()

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.BrowserApplier", return_value=_FakeBrowserApplier(apply_mock)):
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[])

        summary = await run_autoapply(
            triggers=["python"], resume_source=ResumeSource(content="Jane Doe"), store=store,
            output_dir=tmp_path / "pdfs", live=True,
        )

    apply_mock.assert_not_awaited()
    assert summary.applied == 0
    assert store.get("v1")["status"] == "failed"


@pytest.mark.asyncio
async def test_live_run_keeps_ready_status_on_failed_apply_retry(tmp_path):
    """A retry of a previously-ready vacancy that fails to apply must stay "ready"
    (not "failed") so it's retried again next run without re-tailoring."""
    from hr_breaker.autoapply.browser_apply import BrowserApplyError

    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert("v1", "ready", cover_letter="Dear Acme.", pdf_path="/tmp/v1.pdf")
    apply_mock = AsyncMock(side_effect=BrowserApplyError("no submit button"))

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.BrowserApplier", return_value=_FakeBrowserApplier(apply_mock)):
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[])

        summary = await run_autoapply(
            triggers=["python"], resume_source=ResumeSource(content="Jane Doe"), store=store,
            output_dir=tmp_path / "pdfs", live=True,
        )

    assert summary.applied == 0
    assert summary.failed == 1
    assert store.get("v1")["status"] == "ready"  # retryable next run, no re-tailor needed


@pytest.mark.asyncio
async def test_dry_run_does_not_retry_ready_vacancies(tmp_path):
    """Without --live there's no browser session - pending "ready" vacancies must
    be left untouched, not silently marked applied/skipped."""
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    store.upsert("v1", "ready", cover_letter="Dear Acme.", pdf_path="/tmp/v1.pdf")

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls:
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[])

        summary = await run_autoapply(
            triggers=["python"], resume_source=ResumeSource(content="Jane Doe"), store=store,
            output_dir=tmp_path / "pdfs", live=False,
        )

    assert summary.applied == 0
    assert store.get("v1")["status"] == "ready"


@pytest.mark.asyncio
async def test_iteration_event_emitted_during_tailoring(tmp_path):
    """on_iteration passed into optimize_for_job should surface as an "iteration"
    event via emit()/on_event, carrying iteration/max_iterations/status/elapsed/eta."""
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    source = ResumeSource(content="Jane Doe\nPython developer")
    optimized = OptimizedResume(html="<p>tailored</p>", source_checksum=source.checksum, pdf_bytes=b"%PDF-fake", pdf_text="tailored resume text")
    validation = ValidationResult(results=[])

    async def fake_optimize_for_job(*args, on_iteration=None, **kwargs):
        if on_iteration:
            on_iteration(0, optimized, validation)
        return optimized, validation, None

    events = []

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock(side_effect=fake_optimize_for_job)), \
         patch("hr_breaker.autoapply.pipeline.write_cover_letter", new=AsyncMock(return_value="Dear Acme...")), \
         patch("hr_breaker.autoapply.pipeline.asyncio.sleep", new=AsyncMock()):
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[_make_vacancy("v1")])
        mock_hh.get_vacancy_detail = AsyncMock(return_value=_make_vacancy("v1"))

        await run_autoapply(
            triggers=["python"], resume_source=source, store=store,
            output_dir=tmp_path / "pdfs", live=False, max_iterations=3,
            on_event=lambda event, data: events.append((event, data)),
        )

    iteration_events = [data for event, data in events if event == "iteration"]
    assert len(iteration_events) == 1
    assert iteration_events[0]["iteration"] == 1
    assert iteration_events[0]["max_iterations"] == 3
    assert iteration_events[0]["status"] == "PASS"
    assert "elapsed" in iteration_events[0]
    assert "eta" in iteration_events[0]


@pytest.mark.asyncio
async def test_excluded_text_filters_by_title(tmp_path):
    store = AutoApplyStore(tmp_path / "state.sqlite3")
    source = ResumeSource(content="Jane Doe\nPython developer")
    intern_vacancy = Vacancy(
        id="v2", name="Python Intern", employer_name="Acme",
        url="https://hh.ru/vacancy/v2", description="", key_skills=[], area_name=None, raw={},
    )

    with patch("hr_breaker.autoapply.pipeline.HHClient") as mock_hh_cls, \
         patch("hr_breaker.autoapply.pipeline.optimize_for_job", new=AsyncMock()) as mock_optimize:
        mock_hh = mock_hh_cls.return_value
        mock_hh.search_vacancies = AsyncMock(return_value=[intern_vacancy])

        summary = await run_autoapply(
            triggers=["python"], resume_source=source, store=store,
            output_dir=tmp_path / "pdfs", live=False, excluded_text="intern, junior",
        )

    assert summary.found == 1
    assert summary.tailored == 0
    assert store.seen("v2") is False  # filtered before ever being recorded
    mock_optimize.assert_not_awaited()
