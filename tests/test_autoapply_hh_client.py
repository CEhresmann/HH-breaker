"""Tests for the hh.ru API client."""

from unittest.mock import patch

import pytest

from hr_breaker.autoapply.hh_client import HHApiError, HHClient, build_authorization_url


def test_build_authorization_url():
    url = build_authorization_url("cid", "https://example.com/cb", state="xyz")
    assert url.startswith("https://hh.ru/oauth/authorize?")
    assert "client_id=cid" in url
    assert "redirect_uri=https://example.com/cb" in url
    assert "state=xyz" in url


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text
        self.content = bool(json_data)

    def json(self):
        return self._json


class _FakeAsyncClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params=None, headers=None):
        self.calls.append(("GET", url, params, headers))
        return self._response

    async def post(self, url, data=None, headers=None):
        self.calls.append(("POST", url, data, headers))
        return self._response


@pytest.mark.asyncio
async def test_search_vacancies_parses_items_and_sends_headers():
    fake_items = {
        "items": [
            {
                "id": "123",
                "name": "Python Developer",
                "employer": {"name": "Acme"},
                "alternate_url": "https://hh.ru/vacancy/123",
                "snippet": {"requirement": "Know <b>Python</b> well"},
                "key_skills": [{"name": "Python"}, {"name": "SQL"}],
                "area": {"name": "Moscow"},
            }
        ]
    }
    fake_client = _FakeAsyncClient(_FakeResponse(200, fake_items))
    with patch("hr_breaker.autoapply.hh_client.httpx.AsyncClient", return_value=fake_client):
        client = HHClient()
        results = await client.search_vacancies("python", area="1")

    assert len(results) == 1
    v = results[0]
    assert v.id == "123"
    assert v.name == "Python Developer"
    assert v.employer_name == "Acme"
    assert v.key_skills == ["Python", "SQL"]
    assert "Python well" in v.description  # HTML stripped

    method, url, params, headers = fake_client.calls[0]
    assert method == "GET"
    assert url.endswith("/vacancies")
    assert params["text"] == "python"
    assert params["area"] == "1"
    assert "User-Agent" in headers
    assert "Authorization" not in headers  # search is public, no token set


@pytest.mark.asyncio
async def test_search_vacancies_raises_on_error_status():
    fake_client = _FakeAsyncClient(_FakeResponse(403, {}, text="forbidden"))
    with patch("hr_breaker.autoapply.hh_client.httpx.AsyncClient", return_value=fake_client):
        client = HHClient()
        with pytest.raises(HHApiError):
            await client.search_vacancies("python")


@pytest.mark.asyncio
async def test_apply_to_vacancy_requires_access_token():
    client = HHClient(access_token=None)
    with pytest.raises(HHApiError):
        await client.apply_to_vacancy("resume-1", "vacancy-1")


@pytest.mark.asyncio
async def test_apply_to_vacancy_sends_expected_payload_and_auth_header():
    fake_client = _FakeAsyncClient(_FakeResponse(200, {"id": "neg-1"}))
    with patch("hr_breaker.autoapply.hh_client.httpx.AsyncClient", return_value=fake_client):
        client = HHClient(access_token="tok-123")
        await client.apply_to_vacancy("resume-1", "vacancy-1", message="Hello")

    method, url, data, headers = fake_client.calls[0]
    assert method == "POST"
    assert url.endswith("/negotiations")
    assert data == {"resume_id": "resume-1", "vacancy_id": "vacancy-1", "message": "Hello"}
    assert headers["Authorization"] == "Bearer tok-123"


@pytest.mark.asyncio
async def test_get_my_resumes_requires_access_token():
    client = HHClient()
    with pytest.raises(HHApiError):
        await client.get_my_resumes()
