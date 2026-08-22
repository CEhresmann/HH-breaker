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
async def test_search_vacancies_parses_shard_and_drops_ads():
    fake_body = {
        "vacancySearchResult": {
            "vacancies": [
                {"@isAdv": True, "vacancyId": 1, "name": "Sponsored slot"},
                {
                    "vacancyId": 123,
                    "name": "Python Developer",
                    "company": {"name": "Acme"},
                    "area": {"name": "Moscow"},
                },
            ]
        }
    }
    fake_client = _FakeAsyncClient(_FakeResponse(200, fake_body))
    with patch("hr_breaker.autoapply.hh_client.httpx.AsyncClient", return_value=fake_client):
        client = HHClient()
        results = await client.search_vacancies("python", area="1")

    assert len(results) == 1  # the @isAdv entry is dropped
    v = results[0]
    assert v.id == "123"
    assert v.name == "Python Developer"
    assert v.employer_name == "Acme"
    assert v.area_name == "Moscow"
    assert v.url == "https://hh.ru/vacancy/123"

    method, url, params, headers = fake_client.calls[0]
    assert method == "GET"
    assert url.endswith("/shards/vacancy/search")
    assert params["text"] == "python"
    assert params["area"] == "1"
    assert headers["Accept"] == "application/json"
    assert "Authorization" not in headers  # search is public, no token set


@pytest.mark.asyncio
async def test_search_vacancies_clamps_per_page_to_100():
    fake_client = _FakeAsyncClient(_FakeResponse(200, {"vacancySearchResult": {"vacancies": []}}))
    with patch("hr_breaker.autoapply.hh_client.httpx.AsyncClient", return_value=fake_client):
        client = HHClient()
        await client.search_vacancies("python", per_page=500)

    _, _, params, _ = fake_client.calls[0]
    assert params["items_on_page"] == 100


@pytest.mark.asyncio
async def test_search_vacancies_raises_on_error_status():
    fake_client = _FakeAsyncClient(_FakeResponse(403, {}, text="forbidden"))
    with patch("hr_breaker.autoapply.hh_client.httpx.AsyncClient", return_value=fake_client):
        client = HHClient()
        with pytest.raises(HHApiError):
            await client.search_vacancies("python")


@pytest.mark.asyncio
async def test_get_vacancy_detail_parses_embedded_state():
    page_html = """<html><body>
    <template style="display:none" id="HH-Lux-InitialState">{&#34;vacancyView&#34;:{&#34;vacancyId&#34;:456,&#34;name&#34;:&#34;Backend Engineer&#34;,&#34;company&#34;:{&#34;name&#34;:&#34;Acme&#34;},&#34;area&#34;:{&#34;name&#34;:&#34;Moscow&#34;},&#34;description&#34;:&#34;&lt;p&gt;Know Python&lt;/p&gt;&#34;,&#34;keySkills&#34;:{&#34;keySkill&#34;:[&#34;Python&#34;,&#34;SQL&#34;]}}}</template>
    </body></html>"""
    fake_client = _FakeAsyncClient(_FakeResponse(200, text=page_html))
    with patch("hr_breaker.autoapply.hh_client.httpx.AsyncClient", return_value=fake_client):
        client = HHClient()
        v = await client.get_vacancy_detail("456")

    assert v.id == "456"
    assert v.name == "Backend Engineer"
    assert v.employer_name == "Acme"
    assert v.key_skills == ["Python", "SQL"]
    assert "Know Python" in v.description


@pytest.mark.asyncio
async def test_get_vacancy_detail_raises_when_state_missing():
    fake_client = _FakeAsyncClient(_FakeResponse(200, text="<html>no state here</html>"))
    with patch("hr_breaker.autoapply.hh_client.httpx.AsyncClient", return_value=fake_client):
        client = HHClient()
        with pytest.raises(HHApiError):
            await client.get_vacancy_detail("456")


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
