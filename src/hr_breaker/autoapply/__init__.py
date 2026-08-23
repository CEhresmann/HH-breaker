"""hh.ru auto-apply pipeline: search vacancies by trigger keyword, tailor a resume +
cover letter per vacancy via hr-breaker's existing optimization loop, and optionally
submit the application through a real logged-in browser session.

See `pipeline.run_autoapply` for the entry point and its module docstring for an
important caveat about how hh.ru's apply flow relates to per-vacancy tailored PDFs.
"""

from .browser_apply import BrowserApplier, BrowserApplyError, CaptchaDetectedError, login_interactively
from .hh_client import HHApiError, HHClient, Vacancy, build_authorization_url
from .pipeline import AutoApplyRunSummary, run_autoapply
from .state_store import AutoApplyStore

__all__ = [
    "HHClient",
    "HHApiError",
    "Vacancy",
    "build_authorization_url",
    "run_autoapply",
    "AutoApplyRunSummary",
    "AutoApplyStore",
    "BrowserApplier",
    "BrowserApplyError",
    "CaptchaDetectedError",
    "login_interactively",
]
