"""Async client for hh.ru (HeadHunter): vacancy search, OAuth, applying.

Docs: https://github.com/hhru/api, schema at https://api.hh.ru/openapi/redoc.

Since April 2026 the public `GET api.hh.ru/vacancies` collection returns 403
(behind DDoS-Guard) for unregistered apps - registration now requires a
verified employer account. `search_vacancies()` and `get_vacancy_detail()`
instead call the same internal endpoints the hh.ru website itself uses for
anonymous visitors (`hh.ru/shards/vacancy/search`, the `HH-Lux-InitialState`
JSON embedded in a vacancy's HTML page) - undocumented, subject to change
without notice, distinct from the versioned public API.

`apply_to_vacancy()` (`POST api.hh.ru/negotiations`) and `get_my_resumes()`
(`GET api.hh.ru/resumes/mine`) are also 403-blocked, confirmed live with a
fully valid, freshly-issued OAuth token - the block is not about token
validity. Kept here for reference/in case hh.ru's policy changes again; the
live apply path is `browser_apply.BrowserApplier`, which submits through a
real logged-in browser session instead.
"""

import html as html_module
import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

API_BASE = "https://api.hh.ru"
WEB_BASE = "https://hh.ru"
AUTHORIZE_URL = "https://hh.ru/oauth/authorize"
TOKEN_URL = "https://hh.ru/oauth/token"
REQUEST_TIMEOUT = 15.0

# hh.ru asks integrators to identify their app in the User-Agent header, for
# the official api.hh.ru endpoints (OAuth, negotiations).
DEFAULT_USER_AGENT = "hr-breaker-autoapply/0.1 (personal use)"

# The website's own shard/HTML endpoints expect an ordinary browser UA, not
# an app-identifying one.
_WEB_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_INITIAL_STATE_RE = re.compile(
    r'<template style="display:none" id="HH-Lux-InitialState">(.*?)</template>', re.DOTALL
)

# Standard hh.ru "Опыт работы" values for search_vacancies(experience=...).
EXPERIENCE_VALUES = ("noExperience", "between1And3", "between3And6", "moreThan6")


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

    def _web_headers(self, accept_json: bool = False) -> dict[str, str]:
        headers = {"User-Agent": _WEB_USER_AGENT}
        if accept_json:
            headers["Accept"] = "application/json"
        return headers

    async def search_vacancies(
        self,
        text: str,
        area: str | None = None,
        experience: str | None = None,
        per_page: int = 50,
        page: int = 0,
        professional_role: str | None = None,
    ) -> list[Vacancy]:
        """Search vacancies via the website's own search endpoint (see module docstring).

        Results are the compact search-listing shape - no description/key_skills.
        Call `get_vacancy_detail()` per vacancy to fill those in. `per_page` above 100
        is silently reset to 50 server-side, so it's clamped here to stay predictable.

        `experience` is one of the standard hh.ru values (noExperience, between1And3,
        between3And6, moreThan6) - confirmed live: the four values partition the
        unfiltered result count exactly, an invalid value is silently ignored.
        """
        params: dict[str, Any] = {
            "text": text, "items_on_page": min(per_page, 100), "page": page,
        }
        if area:
            params["area"] = area
        if experience:
            params["experience"] = experience
        if professional_role:
            params["professional_role"] = professional_role

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(
                f"{WEB_BASE}/shards/vacancy/search", params=params, headers=self._web_headers(accept_json=True)
            )
        if resp.status_code >= 400:
            raise HHApiError(f"Vacancy search failed ({resp.status_code}): {resp.text}")

        result = resp.json().get("vacancySearchResult", {})
        items = result.get("vacancies", [])
        return [_parse_shard_vacancy(item) for item in items if not item.get("@isAdv")]

    async def get_vacancy_detail(self, vacancy_id: str) -> Vacancy:
        """Fetch full vacancy detail (search results only have the compact shape)."""
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(f"{WEB_BASE}/vacancy/{vacancy_id}", headers=self._web_headers())
        if resp.status_code >= 400:
            raise HHApiError(f"Fetching vacancy {vacancy_id} failed ({resp.status_code}): {resp.text}")
        match = _INITIAL_STATE_RE.search(resp.text)
        if not match:
            raise HHApiError(f"Could not find HH-Lux-InitialState on vacancy {vacancy_id} page")
        state = json.loads(html_module.unescape(match.group(1)))
        return _parse_vacancy_view(state.get("vacancyView", {}))

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


def _extract_key_skills(raw_skills: Any) -> list[str]:
    if not raw_skills:
        return []
    if isinstance(raw_skills, dict):
        raw_skills = raw_skills.get("keySkill") or []
    names = []
    for s in raw_skills:
        if isinstance(s, dict):
            name = s.get("name") or s.get("@name")
            if name:
                names.append(name)
        elif isinstance(s, str):
            names.append(s)
    return names


def _parse_shard_vacancy(item: dict) -> Vacancy:
    """Parse one entry from GET hh.ru/shards/vacancy/search - compact, no description."""
    company = item.get("company") or {}
    area = item.get("area") or {}
    vacancy_id = str(item.get("vacancyId") or item.get("id") or "")
    return Vacancy(
        id=vacancy_id,
        name=item.get("name", ""),
        employer_name=company.get("name", ""),
        url=f"{WEB_BASE}/vacancy/{vacancy_id}",
        description="",
        key_skills=[],
        area_name=area.get("name"),
        raw=item,
    )


def _parse_vacancy_view(view: dict) -> Vacancy:
    """Parse the HH-Lux-InitialState.vacancyView object from a vacancy's HTML page."""
    company = view.get("company") or {}
    area = view.get("area") or {}
    vacancy_id = str(view.get("vacancyId") or "")
    return Vacancy(
        id=vacancy_id,
        name=view.get("name", ""),
        employer_name=company.get("name", ""),
        url=f"{WEB_BASE}/vacancy/{vacancy_id}",
        description=_strip_html(view.get("description") or ""),
        key_skills=_extract_key_skills(view.get("keySkills")),
        area_name=area.get("name"),
        raw=view,
    )
