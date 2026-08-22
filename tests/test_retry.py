import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm.exceptions import RateLimitError
from pydantic_ai.exceptions import ModelHTTPError

from hr_breaker.config import Settings
from hr_breaker.utils import retry as retry_module
from hr_breaker.utils.retry import is_retryable, run_with_retry


def test_settings_has_retry_fields():
    s = Settings()
    assert s.retry_max_attempts == 5
    assert s.retry_max_wait == 60.0


async def test_llm_max_concurrency_limits_simultaneous_calls(monkeypatch):
    monkeypatch.setattr(retry_module, "_semaphore", None)
    monkeypatch.setattr(
        retry_module, "get_settings",
        lambda: MagicMock(
            llm_max_concurrency=1, retry_max_attempts=5, retry_max_wait=60.0, llm_call_timeout=5.0,
        ),
    )

    in_flight = 0
    max_in_flight = 0

    async def func():
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return "ok"

    results = await asyncio.gather(*[run_with_retry(func) for _ in range(5)])
    assert results == ["ok"] * 5
    assert max_in_flight == 1


def test_is_retryable_429():
    exc = ModelHTTPError(status_code=429, model_name="test")
    assert is_retryable(exc) is True


def test_is_retryable_500():
    exc = ModelHTTPError(status_code=500, model_name="test")
    assert is_retryable(exc) is True


def test_is_retryable_502():
    exc = ModelHTTPError(status_code=502, model_name="test")
    assert is_retryable(exc) is True


def test_is_retryable_400_not_retryable():
    exc = ModelHTTPError(status_code=400, model_name="test")
    assert is_retryable(exc) is False


def test_is_retryable_unrelated_exception():
    assert is_retryable(ValueError("nope")) is False


def test_is_retryable_timeout():
    assert is_retryable(TimeoutError("timed out")) is True
    assert is_retryable(asyncio.TimeoutError()) is True


def test_is_retryable_litellm_rate_limit():
    exc = RateLimitError(
        message="rate limited",
        llm_provider="gemini",
        model="gemini/gemini-3-flash",
    )
    assert is_retryable(exc) is True


async def test_run_with_retry_succeeds_first_try():
    func = AsyncMock(return_value="ok")
    result = await run_with_retry(func, "arg1", key="val")
    assert result == "ok"
    func.assert_called_once_with("arg1", key="val")


async def test_run_with_retry_retries_on_429_then_succeeds():
    func = AsyncMock(
        side_effect=[
            ModelHTTPError(status_code=429, model_name="test"),
            "ok",
        ]
    )
    result = await run_with_retry(func, "arg1")
    assert result == "ok"
    assert func.call_count == 2


async def test_run_with_retry_exhausts_attempts():
    func = AsyncMock(
        side_effect=ModelHTTPError(status_code=429, model_name="test")
    )
    with pytest.raises(ModelHTTPError):
        await run_with_retry(func, "arg1", _max_attempts=2, _max_wait=0.01)


async def test_run_with_retry_retries_litellm_rate_limit():
    exc = RateLimitError(
        message="rate limited",
        llm_provider="gemini",
        model="gemini/gemini-3-flash",
    )
    func = AsyncMock(side_effect=[exc, "ok"])
    result = await run_with_retry(func, "arg1")
    assert result == "ok"
    assert func.call_count == 2


async def test_run_with_retry_times_out_and_retries_a_hanging_call(monkeypatch):
    monkeypatch.setattr(
        retry_module, "get_settings",
        lambda: MagicMock(
            llm_max_concurrency=8, retry_max_attempts=3, retry_max_wait=1.0, llm_call_timeout=0.05,
        ),
    )

    calls = {"n": 0}

    async def func():
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(10)  # exceeds llm_call_timeout - should be cancelled, not awaited fully
        return "ok"

    result = await run_with_retry(func)
    assert result == "ok"
    assert calls["n"] == 2


def test_is_retryable_false_for_litellm_none_type_config_error():
    exc = ModelHTTPError(
        status_code=500,
        model_name="openai/gpt-5.3-codex",
        body="litellm.APIConnectionError: APIConnectionError: OpenAIException - argument of type 'NoneType' is not iterable",
    )
    assert is_retryable(exc) is False