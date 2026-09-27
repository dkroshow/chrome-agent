"""Test fixtures for chrome-agent.

Launches a real headless Chrome browser with CDP enabled for integration
tests. Uses the subprocess-based launcher (no Playwright dependency).
"""

import asyncio
import os
import shutil
import signal
from pathlib import Path

import pytest

from chrome_agent.launcher import launch_browser

FIXTURE_PATH = Path(__file__).parent / "fixture.html"
CDP_PORT = 9333  # Use a non-default port to avoid conflicts


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def browser_session(event_loop, tmp_path_factory):
    """Launch a headless browser with CDP for the entire test session.

    Uses an isolated registry so test runs never touch the real registry at
    /tmp/chrome-agent/registry.json. Provides a browser running on CDP_PORT;
    tests that need CDP access use CDPClient or get_ws_url directly.
    """
    reg_path = str(tmp_path_factory.mktemp("registry") / "registry.json")
    result = event_loop.run_until_complete(
        launch_browser(
            port_override=CDP_PORT,
            headless=True,
            pin_to_desktop=False,
            registry_path=reg_path,
        )
    )
    yield result
    # Teardown: kill the browser and remove its session directory.
    try:
        os.kill(result.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    shutil.rmtree(result.user_data_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def fixture_url():
    """URL to the test fixture page."""
    return f"file://{FIXTURE_PATH.resolve()}"


@pytest.fixture(scope="session", autouse=True)
def sweep_code_sign_clones_left_by_tests():
    """Tests kill Chrome on purpose; remove the code-sign clones those kills leave.

    Only entries that appear during this test session and that no process
    holds at the end are removed. Nothing that existed before is touched.
    """
    from chrome_agent.utils import code_sign_clone_snapshot, remove_unheld_clone_dirs

    before = code_sign_clone_snapshot()
    yield
    remove_unheld_clone_dirs(code_sign_clone_snapshot() - before, timeout=60)
