"""Pre-render length estimation for fast fail on oversized content."""

import re
from dataclasses import dataclass

from hr_breaker.config import get_settings, logger

__all__ = [
    "LengthEstimate",
    "estimate_content_length",
]


@dataclass
class LengthEstimate:
    chars: int
    words: int
    lines: int
    overflow_chars: int  # How many chars over
    overflow_words: int


def estimate_content_length(html: str) -> LengthEstimate:
    """Estimate rendered content length from HTML without rendering."""
    settings = get_settings()

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", "", html)
    text = re.sub(r"\s+", " ", text).strip()

    chars = len(text)
    words = len(text.split())
    lines = chars // 85  # ~85 chars per line at 11pt Times

    return LengthEstimate(
        chars=chars,
        words=words,
        lines=lines,
        overflow_chars=max(0, chars - settings.resume_max_chars),
        overflow_words=max(0, words - settings.resume_max_words),
    )
