import logging

from ..config import get_settings
from .scrapers.base import CloudflareBlockedError, ScrapingError
from .scrapers.httpx_scraper import HttpxScraper
from .scrapers.wayback_scraper import WaybackScraper
from .scrapers.playwright_scraper import PlaywrightScraper, PLAYWRIGHT_AVAILABLE

logger = logging.getLogger(__name__)

# Re-export for backwards compatibility
__all__ = ["scrape_job_posting", "ScrapingError", "CloudflareBlockedError"]


def scrape_job_posting(
    url: str,
    max_retries: int = 3,
    use_wayback: bool = True,
    use_playwright: bool = True,
) -> str:
    """
    Scrape job posting text from URL with fallback chain.

    Order: httpx -> wayback (skipped if cloudflare) -> playwright
    """
    settings = get_settings()
    errors: list[tuple[str, str]] = []
    cloudflare_blocked = False

    # 1. Try httpx (direct fetch)
    httpx_scraper = HttpxScraper(
        max_retries=max_retries,
        timeout=settings.scraper_httpx_timeout,
    )
    try:
        result = httpx_scraper.scrape(url)
        logger.info(f"Scraped {url} with httpx")
        return result
    except CloudflareBlockedError as e:
        cloudflare_blocked = True
        errors.append((httpx_scraper.name, str(e)))
        logger.warning(f"httpx blocked by Cloudflare for {url}")
    except ScrapingError as e:
        errors.append((httpx_scraper.name, str(e)))
        logger.warning(f"httpx failed for {url}: {e}")

    # 2. Try Wayback Machine (skip if Cloudflare blocked - unlikely to have snapshot)
    if use_wayback and not cloudflare_blocked:
        wayback_scraper = WaybackScraper(timeout=settings.scraper_wayback_timeout)
        try:
            result = wayback_scraper.scrape(url)
            logger.info(f"Scraped {url} via Wayback Machine")
            return result
        except ScrapingError as e:
            errors.append((wayback_scraper.name, str(e)))
            logger.warning(f"Wayback failed for {url}: {e}")
    elif use_wayback and cloudflare_blocked:
        logger.info("Skipping Wayback (Cloudflare site unlikely to have snapshot)")

    # 3. Try Playwright (browser)
    if use_playwright and PLAYWRIGHT_AVAILABLE:
        logger.warning(f"Trying Playwright browser for {url}...")
        playwright_scraper = PlaywrightScraper(timeout=settings.scraper_playwright_timeout)
        try:
            result = playwright_scraper.scrape(url)
            logger.warning(f"Scraped {url} with Playwright")
            return result
        except (ScrapingError, CloudflareBlockedError) as e:
            errors.append((playwright_scraper.name, str(e)))
            logger.warning(f"Playwright failed for {url}: {e}")
    elif use_playwright and not PLAYWRIGHT_AVAILABLE:
        errors.append(("playwright", "not installed"))

    # All methods failed
    methods_tried = ", ".join(f"{name}: {err}" for name, err in errors)
    raise ScrapingError(
        f"Failed to scrape {url}. Methods tried: [{methods_tried}]. "
        "Try pasting the job description text directly."
    )
