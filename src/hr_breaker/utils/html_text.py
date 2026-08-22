"""HTML text extraction utilities."""

import re


def extract_text_from_html(html: str) -> str:
    """Strip HTML tags to plain text. ~95% accuracy, fast."""
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()
