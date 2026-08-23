import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from hr_breaker.cli import cli, live_progress
import hr_breaker.config as config_module
from hr_breaker.models.profile import DocumentExtraction
from hr_breaker.services.profile_store import ProfileStore
from hr_breaker.utils.optimization_telemetry import report_call_start, report_usage


@pytest.mark.asyncio
async def test_live_progress_start_and_done_cycle_does_not_crash_or_hang():
    """Smoke test for the background ticker task's lifecycle - enter, a real
    start->done event pair, then a clean exit (task cancelled, no leaks)."""
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_tokens=0, cache_write_tokens=0, requests=1)

    async with live_progress():
        report_call_start("Optimizer", "moonshot/kimi-k2.6")
        report_usage("Optimizer", "moonshot/kimi-k2.6", usage)
        await asyncio.sleep(0)


def _fake_run_summary():
    return SimpleNamespace(found=0, already_seen=0, tailored=0, failed=0, applied=0, skipped_apply_cap=0)


def test_autoapply_run_max_iterations_defaults_to_two(monkeypatch):
    mock_run = AsyncMock(return_value=_fake_run_summary())
    monkeypatch.setattr("hr_breaker.autoapply.run_autoapply", mock_run)

    result = CliRunner().invoke(cli, ["autoapply", "run", "-t", "python", "--profile", "some-id"])

    assert result.exit_code == 0, result.output
    assert mock_run.call_args.kwargs["max_iterations"] == 2


def test_autoapply_run_max_iterations_overridable(monkeypatch):
    mock_run = AsyncMock(return_value=_fake_run_summary())
    monkeypatch.setattr("hr_breaker.autoapply.run_autoapply", mock_run)

    result = CliRunner().invoke(
        cli, ["autoapply", "run", "-t", "python", "--profile", "some-id", "--max-iterations", "4"]
    )

    assert result.exit_code == 0, result.output
    assert mock_run.call_args.kwargs["max_iterations"] == 4


def test_profile_show_reports_empty_extraction(monkeypatch, tmp_path):
    monkeypatch.setenv("PROFILE_DIR", str(tmp_path / "profiles"))
    config_module.clear_settings_cache()
    try:
        store = ProfileStore()
        profile = store.create_profile("Candidate")
        doc = store.add_note(profile.id, title="Resume", content_text="raw text")

        with patch(
            "hr_breaker.agents.extractor.extract_document",
            new=AsyncMock(return_value=DocumentExtraction()),
        ):
            asyncio.run(store.extract_document_content(profile.id, doc.id))

        result = CliRunner().invoke(cli, ["profile", "show", profile.id])

        assert result.exit_code == 0
        assert "empty extraction" in result.output
        assert "[note, extracted]" not in result.output
    finally:
        config_module.clear_settings_cache()


def test_backfill_reports_empty_extraction_separately(monkeypatch, tmp_path):
    monkeypatch.setenv("PROFILE_DIR", str(tmp_path / "profiles"))
    config_module.clear_settings_cache()
    try:
        store = ProfileStore()
        profile = store.create_profile("Candidate")
        store.add_note(profile.id, title="Resume", content_text="raw text")

        with patch(
            "hr_breaker.agents.extractor.extract_document",
            new=AsyncMock(return_value=DocumentExtraction()),
        ):
            result = CliRunner().invoke(cli, ["backfill", "--profile", profile.id])

        assert result.exit_code == 0
        assert "Resume... empty" in result.output
        assert "Done: 0 extracted, 1 empty, 0 failed, 1 total" in result.output
    finally:
        config_module.clear_settings_cache()
