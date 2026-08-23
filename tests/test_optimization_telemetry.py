from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hr_breaker.utils.optimization_telemetry import (
    IterationETA,
    accumulate_usage_totals,
    report_usage,
    run_tracked_agent,
    telemetry_reporter,
    zero_usage_totals,
)


def test_iteration_eta_estimates_remaining_time_from_rolling_average(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "hr_breaker.utils.optimization_telemetry.time.monotonic", lambda: clock[0]
    )

    eta = IterationETA(max_iterations=4)  # constructed at clock=0.0

    clock[0] = 10.0
    elapsed0, eta0 = eta.tick(0)
    assert elapsed0 == 10.0
    assert eta0 == 30.0  # 3 remaining iterations * 10.0 avg

    clock[0] = 30.0
    elapsed1, eta1 = eta.tick(1)
    assert elapsed1 == 20.0
    assert eta1 == 30.0  # avg=15.0 * 2 remaining


def test_iteration_eta_is_zero_on_last_iteration(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "hr_breaker.utils.optimization_telemetry.time.monotonic", lambda: clock[0]
    )

    eta = IterationETA(max_iterations=2)
    clock[0] = 5.0
    eta.tick(0)
    clock[0] = 10.0
    _, eta1 = eta.tick(1)
    assert eta1 == 0.0


def test_accumulate_usage_totals_adds_usage_entry_values():
    totals = zero_usage_totals()
    updated = accumulate_usage_totals(
        totals,
        {
            "requests": 2,
            "input_tokens": 300,
            "output_tokens": 40,
            "cache_read_tokens": 120,
            "cache_write_tokens": 10,
        },
    )

    assert updated == {
        "requests": 2,
        "input_tokens": 300,
        "output_tokens": 40,
        "cache_read_tokens": 120,
        "cache_write_tokens": 10,
    }


@pytest.mark.asyncio
async def test_run_tracked_agent_reports_usage():
    captured = []
    usage = SimpleNamespace(
        requests=1,
        input_tokens=123,
        output_tokens=45,
        cache_read_tokens=67,
        cache_write_tokens=8,
    )
    result = SimpleNamespace(output="ok", usage=lambda: usage)
    agent = SimpleNamespace(
        model=SimpleNamespace(model_name="openai/gpt-5.3-codex"),
        run=AsyncMock(return_value=result),
    )

    with telemetry_reporter(captured.append):
        observed = await run_tracked_agent(agent, "hello", component="LLMChecker")

    assert observed is result
    agent.run.assert_called_once_with("hello")
    assert len(captured) == 2

    assert captured[0]["event"] == "start"
    assert captured[0]["component"] == "LLMChecker"
    assert captured[0]["model"] == "openai/gpt-5.3-codex"

    assert captured[1]["event"] == "done"
    assert captured[1]["component"] == "LLMChecker"
    assert captured[1]["model"] == "openai/gpt-5.3-codex"
    assert captured[1]["provider"] == "openai"
    assert captured[1]["requests"] == 1
    assert captured[1]["input_tokens"] == 123
    assert captured[1]["output_tokens"] == 45
    assert captured[1]["cache_read_tokens"] == 67
    assert captured[1]["cache_write_tokens"] == 8



def test_accumulate_usage_totals_skips_unavailable_usage_entries():
    totals = zero_usage_totals()
    updated = accumulate_usage_totals(
        totals,
        {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "usage_available": False,
        },
    )

    assert updated == totals


def test_report_usage_marks_zero_only_embedding_usage_unavailable():
    captured = []
    usage = SimpleNamespace(
        requests=0,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )

    with telemetry_reporter(captured.append):
        report_usage("VectorSimilarityMatcher", "gemini/gemini-embedding-2-preview", usage)

    assert len(captured) == 1
    assert captured[0]["usage_available"] is False
    assert captured[0]["requests"] == 0
    assert captured[0]["input_tokens"] == 0


def test_report_usage_keeps_zero_usage_available_for_non_embedding_models():
    captured = []
    usage = SimpleNamespace(
        requests=0,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )

    with telemetry_reporter(captured.append):
        report_usage("LLMChecker", "openai/gpt-5.3-codex", usage)

    assert len(captured) == 1
    assert captured[0]["usage_available"] is False


def test_report_usage_maps_prompt_completion_tokens_for_litellm_usage():
    captured = []
    usage = SimpleNamespace(
        prompt_tokens=691,
        completion_tokens=1071,
        cache_read_tokens=0,
        cache_write_tokens=0,
    )

    with telemetry_reporter(captured.append):
        report_usage("VectorSimilarityMatcher", "openai/text-embedding-3-small", usage)

    assert len(captured) == 1
    assert captured[0]["usage_available"] is True
    assert captured[0]["requests"] == 1
    assert captured[0]["input_tokens"] == 691
    assert captured[0]["output_tokens"] == 1071