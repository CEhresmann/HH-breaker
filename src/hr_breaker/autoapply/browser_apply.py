"""Real-browser application submission for hh.ru.

hh.ru's official apply endpoint (`POST api.hh.ru/negotiations`) is 403-blocked
(DDoS-Guard) for unregistered apps regardless of token validity - see
`hh_client` module docstring. This module submits applications through the
same UI a logged-in human uses instead, via a persistent Playwright browser
profile.

Login is a one-time interactive step (`login_interactively()`): a visible
browser opens, the user logs in by hand, and closing the window persists the
session (cookies/local storage) to `PROFILE_DIR` for later headless runs.

Selectors are based on hh.ru's markup as of 2026-08 (confirmed against a real
vacancy page, plus community-documented selectors from
https://habr.com/ru/articles/981764/) and are expected to need updates if
hh.ru changes its DOM. On an unexpected page state, `apply()` saves a
screenshot + the dialog's HTML to `debug_dir` and raises `BrowserApplyError`
with the paths, instead of guessing.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import Page, async_playwright

logger = logging.getLogger(__name__)

PROFILE_DIR = Path(".cache/hh_browser_profile")
DEBUG_DIR = Path("output/autoapply/browser_debug")
BASE_URL = "https://hh.ru"
NAV_TIMEOUT_MS = 20_000
DIALOG_TIMEOUT_MS = 8_000

_APPLY_LINK_SELECTOR = '[data-qa="vacancy-response-link-top"]'
_DIALOG_SELECTOR = '[role="dialog"]'
_LETTER_INPUT_SELECTOR = '[data-qa="vacancy-response-popup-form-letter-input"]'
_CLOSE_BUTTON_SELECTOR = '[data-qa="response-popup-close"]'
_SUCCESS_TEXT_RE = re.compile(r"отклик отправлен", re.IGNORECASE)
_SUBMIT_BUTTON_NAME_RE = re.compile(r"откликнут|отправ", re.IGNORECASE)
_BLOCKED_TITLE_RE = re.compile(r"ddos-guard|checking your browser|подтвердите, что вы человек", re.IGNORECASE)


class BrowserApplyError(Exception):
    """Raised when the browser-based apply flow can't complete."""


class CaptchaDetectedError(BrowserApplyError):
    """hh.ru showed a bot-check/CAPTCHA page. Never bypassed - solve it manually
    via `login_interactively()`, then retry."""


@dataclass
class ApplyOutcome:
    status: str  # "applied" | "already_applied"
    detail: str = ""


async def login_interactively(profile_dir: Path = PROFILE_DIR) -> None:
    """Open a visible browser for the user to log into hh.ru by hand. Closing the
    window persists the session to `profile_dir` for later headless runs."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        context = await pw.chromium.launch_persistent_context(str(profile_dir), headless=False)
        page = await context.new_page()
        await page.goto(f"{BASE_URL}/account/login?role=applicant")
        logger.info("Log into hh.ru in the opened browser window, then close it to save the session.")
        await context.wait_for_event("close", timeout=0)


class BrowserApplier:
    """Reusable logged-in browser session for submitting applications across a run."""

    def __init__(self, profile_dir: Path = PROFILE_DIR, headless: bool = True, debug_dir: Path = DEBUG_DIR):
        self.profile_dir = profile_dir
        self.headless = headless
        self.debug_dir = debug_dir
        self._pw = None
        self._context = None

    async def __aenter__(self) -> "BrowserApplier":
        if not self.profile_dir.exists():
            raise BrowserApplyError(
                f"No browser profile at {self.profile_dir} - run 'hr-breaker autoapply browser-login' first."
            )
        self._pw = await async_playwright().start()
        self._context = await self._pw.chromium.launch_persistent_context(
            str(self.profile_dir), headless=self.headless
        )
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._context is not None:
            await self._context.close()
        if self._pw is not None:
            await self._pw.stop()

    async def apply(self, vacancy_id: str, cover_letter: str, resume_title: str | None = None) -> ApplyOutcome:
        if self._context is None:
            raise BrowserApplyError("BrowserApplier used outside its `async with` block")
        page = await self._context.new_page()
        try:
            return await self._apply_on_page(page, vacancy_id, cover_letter, resume_title)
        finally:
            await page.close()

    async def _apply_on_page(
        self, page: Page, vacancy_id: str, cover_letter: str, resume_title: str | None
    ) -> ApplyOutcome:
        await page.goto(f"{BASE_URL}/vacancy/{vacancy_id}", timeout=NAV_TIMEOUT_MS)
        await self._raise_if_blocked(page, vacancy_id)

        apply_link = page.locator(_APPLY_LINK_SELECTOR).first
        if await apply_link.count() == 0:
            await self._dump_debug(page, vacancy_id, "no-apply-link")
            return ApplyOutcome("already_applied", "apply link not found on vacancy page")

        await apply_link.click()

        dialog = page.locator(_DIALOG_SELECTOR).first
        try:
            await dialog.wait_for(timeout=DIALOG_TIMEOUT_MS)
        except Exception:
            # Some vacancies apply directly without a cover-letter dialog.
            if await self._page_shows_success(page):
                return ApplyOutcome("applied")
            await self._dump_debug(page, vacancy_id, "no-dialog-after-click")
            raise BrowserApplyError(
                f"No response dialog appeared after clicking apply for vacancy {vacancy_id} "
                f"(debug dump saved under {self.debug_dir})"
            )

        if resume_title:
            await self._select_resume(dialog, resume_title)

        letter_input = dialog.locator(_LETTER_INPUT_SELECTOR).first
        if await letter_input.count() > 0 and cover_letter:
            await letter_input.fill(cover_letter)

        submit_button = dialog.get_by_role("button", name=_SUBMIT_BUTTON_NAME_RE).first
        if await submit_button.count() == 0:
            await self._dump_debug(page, vacancy_id, "no-submit-button")
            raise BrowserApplyError(
                f"Could not find a submit button in the response dialog for vacancy {vacancy_id} "
                f"(debug dump saved under {self.debug_dir})"
            )
        await submit_button.click()

        if not await self._page_shows_success(page, timeout=DIALOG_TIMEOUT_MS):
            await self._dump_debug(page, vacancy_id, "no-success-confirmation")
            raise BrowserApplyError(
                f"No success confirmation seen after submitting for vacancy {vacancy_id} "
                f"(debug dump saved under {self.debug_dir}) - it may still have gone through, check manually"
            )
        return ApplyOutcome("applied")

    async def _select_resume(self, dialog, resume_title: str) -> None:
        option = dialog.get_by_text(resume_title, exact=False).first
        if await option.count() > 0:
            await option.click()

    async def _page_shows_success(self, page: Page, timeout: int = 2_000) -> bool:
        try:
            await page.get_by_text(_SUCCESS_TEXT_RE).first.wait_for(timeout=timeout)
            return True
        except Exception:
            return False

    async def _raise_if_blocked(self, page: Page, vacancy_id: str) -> None:
        title = await page.title()
        if _BLOCKED_TITLE_RE.search(title or ""):
            await self._dump_debug(page, vacancy_id, "blocked")
            raise CaptchaDetectedError(
                f"hh.ru showed a bot-check page for vacancy {vacancy_id} (title: {title!r}) - "
                "solve it manually via 'hr-breaker autoapply browser-login', then retry."
            )

    async def _dump_debug(self, page: Page, vacancy_id: str, tag: str) -> None:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        base = self.debug_dir / f"{vacancy_id}_{tag}"
        try:
            await page.screenshot(path=str(base.with_suffix(".png")))
            base.with_suffix(".html").write_text(await page.content())
        except Exception:
            logger.exception(f"Failed to write debug dump for vacancy {vacancy_id}")
