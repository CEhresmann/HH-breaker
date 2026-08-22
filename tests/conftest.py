"""Pytest configuration."""

import pytest
from dotenv import load_dotenv

# Load .env before running tests
load_dotenv()


@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for all async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
