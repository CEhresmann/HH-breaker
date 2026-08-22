"""Async client for the hh.ru (HeadHunter) API: vacancy search, OAuth, applying.

Docs: https://github.com/hhru/api, schema at https://api.hh.ru/openapi/redoc.

`search_vacancies()` (`GET /vacancies`) is public, no auth required.
`apply_to_vacancy()` (`POST /negotiations`) is not in the public OpenAPI schema -
the resume_id/vacancy_id/message shape follows community usage. Validate against
a real vacancy before relying on it at scale.
"""

from dataclasses import dataclass
from typing import Any

import httpx

API_BASE = "https://api.hh.ru"
AUTHORIZE_URL = "https://hh.ru/oauth/authorize"
TOKEN_URL = "https://hh.ru/oauth/token"
REQUEST_TIMEOUT = 15.0

# hh.ru asks integrators to identify their app in the User-Agent header.
DEFAULT_USER_AGENT = "hr-breaker-autoapply/0.1 (personal use)"


class HHApiError(Exception):
    """Raised when the hh.ru API returns an error response."""


@dataclass
class Vacancy:
    """Minimal projection of an hh.ru vacancy search result."""

    id: str
    name: str
    employer_name: str
    url: str
    description: str  # plain text, HTML stripped
    key_skills: list[str]
    area_name: str | None
    raw: dict[str, Any]


def build_authorization_url(client_id: str, redirect_uri: str, state: str = "") -> str:
    """Build the URL to send the user to for the one-time OAuth consent step."""
    params = f"response_type=code&client_id={client_id}&redirect_uri={redirect_uri}"
    if state:
        params += f"&state={state}"
    return f"{AUTHORIZE_URL}?{params}"


class HHClient:
    """Async client for hh.ru vacancy search, applicant resumes, and applying."""

    def __init__(self, access_token: str | None = None, user_agent: str | None = None):
        self.access_token = access_token
        self.user_agent = user_agent or DEFAULT_USER_AGENT

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    async def exchange_code_for_token(
        self, client_id: str, client_secret: str, code: str, redirect_uri: str
    ) -> dict[str, Any]:
        """Exchange a one-time OAuth authorization code for access/refresh tokens."""
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                headers={"User-Agent": self.user_agent},
            )
        if resp.status_code >= 400:
            raise HHApiError(f"Token exchange failed ({resp.status_code}): {resp.text}")
        return resp.json()

    async def refresh_access_token(self, client_id: str, client_secret: str, refresh_token: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                },
                headers={"User-Agent": self.user_agent},
            )
        if resp.status_code >= 400:
            raise HHApiError(f"Token refresh failed ({resp.status_code}): {resp.text}")
        return resp.json()

    async def search_vacancies(
        self,
        text: str,
        area: str | None = None,
        excluded_text: str | None = None,
        per_page: int = 50,
        page: int = 0,
        professional_role: str | None = None,
    ) -> list[Vacancy]:
        """Search vacancies by full-text query. Public endpoint - no auth needed."""
        params: dict[str, Any] = {"text": text, "per_page": per_page, "page": page}
        if area:
            params["area"] = area
        if excluded_text:
            params["excluded_text"] = excluded_text
        if professional_role:
            params["professional_role"] = professional_role

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(f"{API_BASE}/vacancies", params=params, headers=self._headers())
        if resp.status_code >= 400:
            raise HHApiError(f"Vacancy search failed ({resp.status_code}): {resp.text}")

        items = resp.json().get("items", [])
        return [_parse_vacancy_summary(item) for item in items]

    async def get_vacancy_detail(self, vacancy_id: str) -> Vacancy:
        """Fetch full vacancy detail (search results only have a truncated snippet)."""
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(f"{API_BASE}/vacancies/{vacancy_id}", headers=self._headers())
        if resp.status_code >= 400:
            raise HHApiError(f"Fetching vacancy {vacancy_id} failed ({resp.status_code}): {resp.text}")
        return _parse_vacancy_summary(resp.json())

    async def get_my_resumes(self) -> list[dict]:
        """List the authenticated user's own resumes (needed to get a resume_id to apply with)."""
        if not self.access_token:
            raise HHApiError("get_my_resumes requires an access_token")
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(f"{API_BASE}/resumes/mine", headers=self._headers())
        if resp.status_code >= 400:
            raise HHApiError(f"Listing resumes failed ({resp.status_code}): {resp.text}")
        return resp.json().get("items", [])

    async def apply_to_vacancy(self, resume_id: str, vacancy_id: str, message: str | None = None) -> dict:
        """Submit an application ("отклик"). See module docstring."""
        if not self.access_token:
            raise HHApiError("apply_to_vacancy requires an access_token")
        data: dict[str, Any] = {"resume_id": resume_id, "vacancy_id": vacancy_id}
        if message:
            data["message"] = message
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(f"{API_BASE}/negotiations", data=data, headers=self._headers())
        if resp.status_code >= 400:
            raise HHApiError(f"Apply to vacancy {vacancy_id} failed ({resp.status_code}): {resp.text}")
        return resp.json() if resp.content else {}


def _strip_html(html: str) -> str:
    from hr_breaker.utils.html_text import extract_text_from_html

    return extract_text_from_html(html) if html else ""


def _parse_vacancy_summary(item: dict) -> Vacancy:
    employer = item.get("employer") or {}
    area = item.get("area") or {}
    key_skills = [s.get("name", "") for s in (item.get("key_skills") or []) if s.get("name")]
    description = item.get("description") or item.get("snippet", {}).get("requirement", "") or ""
    return Vacancy(
        id=str(item.get("id")),
        name=item.get("name", ""),
        employer_name=employer.get("name", ""),
        url=item.get("alternate_url", ""),
        description=_strip_html(description),
        key_skills=key_skills,
        area_name=area.get("name"),
        raw=item,
    )
