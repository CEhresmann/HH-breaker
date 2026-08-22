from abc import ABC, abstractmethod

from bs4 import BeautifulSoup

from hr_breaker.config import get_settings


class ScrapingError(Exception):
    """Raised when job scraping fails."""

    pass


class CloudflareBlockedError(ScrapingError):
    """Raised when blocked by Cloudflare or similar bot protection."""

    pass


class BaseScraper(ABC):
    """Base class for job posting scrapers."""

    name: str

    @abstractmethod
    def scrape(self, url: str) -> str:
        """Return job text or raise ScrapingError."""
        pass

    def is_cloudflare_blocked(self, html: str) -> bool:
        """Check if response is a Cloudflare challenge page."""
        indicators = [
            "Just a moment...",
            "cf-browser-verification",
            "challenge-platform",
            "_cf_chl_opt",
            "Checking your browser",
        ]
        return any(ind in html for ind in indicators)

    def extract_job_text(self, html: str) -> str:
        """Extract job posting text from HTML."""
        settings = get_settings()
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Try common job posting containers
        containers = [
            soup.find("div", class_=lambda x: x and "job" in x.lower()),
            soup.find("article"),
            soup.find("main"),
            soup.find("div", id=lambda x: x and "job" in x.lower()),
        ]

        for container in containers:
            if container:
                text = container.get_text(separator="\n", strip=True)
                if len(text) > settings.scraper_min_text_length:
                    return text

        # Fallback: get body text
        return soup.get_text(separator="\n", strip=True)
