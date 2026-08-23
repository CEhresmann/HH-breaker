"""Tests for the browser-based apply flow's branching logic, with Playwright's
Page/Locator faked out - no real browser is launched."""

import pytest

from hr_breaker.autoapply.browser_apply import (
    _APPLY_LINK_SELECTOR,
    _DIALOG_SELECTOR,
    _LETTER_INPUT_SELECTOR,
    BrowserApplier,
    BrowserApplyError,
    CaptchaDetectedError,
)


class _FakeLocator:
    def __init__(self, exists=True, children=None):
        self.exists = exists
        self._children = children or {}
        self.clicked = False
        self.filled_value = None

    async def count(self):
        return 1 if self.exists else 0

    async def click(self):
        self.clicked = True

    async def fill(self, value):
        self.filled_value = value

    async def wait_for(self, timeout=None):
        if not self.exists:
            raise TimeoutError("not found")

    @property
    def first(self):
        return self

    def locator(self, selector):
        return self._children.get(("locator", selector), _FakeLocator(exists=False))

    def get_by_role(self, role, name=None):
        return self._children.get(("role", role), _FakeLocator(exists=False))

    def get_by_text(self, pattern, exact=None):
        return self._children.get(("text", None), _FakeLocator(exists=False))


class _FakePage:
    def __init__(self, title_text="hh.ru", apply_link=None, dialog=None, success=True):
        self._title = title_text
        self._apply_link = apply_link if apply_link is not None else _FakeLocator(exists=False)
        self._dialog = dialog
        self._success_locator = _FakeLocator(exists=success)

    async def goto(self, url, timeout=None):
        pass

    async def title(self):
        return self._title

    def locator(self, selector):
        if selector == _APPLY_LINK_SELECTOR:
            return self._apply_link
        if selector == _DIALOG_SELECTOR:
            return self._dialog if self._dialog is not None else _FakeLocator(exists=False)
        return _FakeLocator(exists=False)

    def get_by_text(self, pattern, exact=None):
        return self._success_locator

    async def screenshot(self, path=None):
        pass

    async def content(self):
        return "<html></html>"


def _make_applier(tmp_path):
    applier = BrowserApplier.__new__(BrowserApplier)
    applier.debug_dir = tmp_path / "debug"
    return applier


@pytest.mark.asyncio
async def test_context_manager_raises_when_profile_dir_missing(tmp_path):
    applier = BrowserApplier(profile_dir=tmp_path / "no-such-profile")

    with pytest.raises(BrowserApplyError):
        await applier.__aenter__()


@pytest.mark.asyncio
async def test_apply_detects_captcha_page(tmp_path):
    applier = _make_applier(tmp_path)
    page = _FakePage(title_text="DDoS-Guard: checking your browser")

    with pytest.raises(CaptchaDetectedError):
        await applier._apply_on_page(page, "123", "letter", None)


@pytest.mark.asyncio
async def test_apply_returns_already_applied_when_no_apply_link(tmp_path):
    applier = _make_applier(tmp_path)
    page = _FakePage(apply_link=_FakeLocator(exists=False))

    outcome = await applier._apply_on_page(page, "123", "letter", None)

    assert outcome.status == "already_applied"


@pytest.mark.asyncio
async def test_apply_raises_when_no_dialog_appears_and_no_success(tmp_path):
    applier = _make_applier(tmp_path)
    page = _FakePage(apply_link=_FakeLocator(exists=True), dialog=_FakeLocator(exists=False), success=False)

    with pytest.raises(BrowserApplyError):
        await applier._apply_on_page(page, "123", "letter", None)


@pytest.mark.asyncio
async def test_apply_raises_when_dialog_has_no_submit_button(tmp_path):
    dialog = _FakeLocator(exists=True, children={
        ("locator", _LETTER_INPUT_SELECTOR): _FakeLocator(exists=True),
    })
    applier = _make_applier(tmp_path)
    page = _FakePage(apply_link=_FakeLocator(exists=True), dialog=dialog)

    with pytest.raises(BrowserApplyError):
        await applier._apply_on_page(page, "123", "letter", None)


@pytest.mark.asyncio
async def test_apply_happy_path_fills_letter_and_submits(tmp_path):
    letter = _FakeLocator(exists=True)
    submit = _FakeLocator(exists=True)
    dialog = _FakeLocator(exists=True, children={
        ("locator", _LETTER_INPUT_SELECTOR): letter,
        ("role", "button"): submit,
    })
    applier = _make_applier(tmp_path)
    page = _FakePage(apply_link=_FakeLocator(exists=True), dialog=dialog, success=True)

    outcome = await applier._apply_on_page(page, "123", "Dear hiring manager", None)

    assert outcome.status == "applied"
    assert letter.filled_value == "Dear hiring manager"
    assert submit.clicked is True
