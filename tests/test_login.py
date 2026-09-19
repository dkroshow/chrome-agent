"""Tests for login-check / login.

A local server plays a site with a sign-in: /app is the site, /api/me is an
authenticated read (200 or 401), /signin sets the session cookie. A second
hostname for the same server (localhost vs 127.0.0.1) plays the identity
provider the site redirects to. Everything runs headless in isolated roots.
"""

import asyncio
import http.server
import json
import os
import subprocess
import sys
import threading

import pytest

from chrome_agent import launcher
from chrome_agent.cdp_client import get_targets
from chrome_agent.launcher import find_chrome_binary, launch_browser
from chrome_agent.login import ERROR, NEEDS_LOGIN, OK, classify, login_check, same_site, wait_for_login
from chrome_agent.registry import stop

PORT = 9362

PROBE = """
fetch('/api/me').then(r => r.status === 200
  ? r.json().then(j => ({status: 'ok', room: j.room}))
  : {status: 'needs_login', http: r.status})
"""

pytestmark = pytest.mark.skipif(
    find_chrome_binary() is None, reason="Chrome/Chromium not installed"
)


class _Site(http.server.BaseHTTPRequestHandler):
    redirect_to_idp = False
    late_js_redirect_ms = None
    api_calls = 0

    def do_GET(self):  # noqa: N802
        signed_in = "sid=synthetic" in self.headers.get("Cookie", "")
        if self.path == "/signin":
            self._send(200, "signed in", cookie="sid=synthetic; Max-Age=86400; Path=/")
        elif self.path == "/app" and not signed_in and _Site.late_js_redirect_ms is not None:
            port = self.server.server_address[1]
            self._send(200, "<html><body>page<script>setTimeout(() => { location.href = "
                       f"'http://localhost:{port}/idp' }}, {_Site.late_js_redirect_ms})</script></body></html>")
        elif self.path == "/api/me":
            _Site.api_calls += 1
            self._send(200, json.dumps({"room": "alpha"})) if signed_in else self._send(401, "{}")
        elif self.path == "/app" and not signed_in and _Site.redirect_to_idp:
            port = self.server.server_address[1]
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{port}/idp")
            self.end_headers()
        else:
            self._send(200, "<html><body>page</body></html>")

    def _send(self, code, body, cookie=None):
        self.send_response(code)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def site():
    _Site.redirect_to_idp = False
    _Site.late_js_redirect_ms = None
    _Site.api_calls = 0
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def browser(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROME_AGENT_PROFILE_ROOT", str(tmp_path / "profiles"))
    session_root = tmp_path / "sessions"
    session_root.mkdir()
    monkeypatch.setattr(launcher, "_SESSION_ROOT", str(session_root))
    registry = str(tmp_path / "registry.json")
    info = asyncio.run(launch_browser(
        port_override=PORT, headless=True, pin_to_desktop=False,
        registry_path=registry, profile="site",
    ))
    yield info
    stop(info.name, registry_path=registry)


def test_classify():
    assert classify(True) == (OK, {})
    assert classify("needs_login") == (NEEDS_LOGIN, {})
    assert classify({"status": "ok", "room": "a"}) == (OK, {"room": "a"})
    assert classify({"status": "weird"})[0] == ERROR
    assert classify(None)[0] == ERROR
    assert same_site("http://a.test/x", "http://a.test:81/y")
    assert not same_site("http://a.test/", "http://login.b.test/")


def test_check_reports_needs_login_then_ok(browser, site):
    def site_tabs():
        return {t["id"] for t in get_targets(port=PORT) if t["url"].startswith(site)}

    assert site_tabs() == set()
    result = asyncio.run(login_check(port=PORT, url=f"{site}/app", probe=PROBE))
    assert (result.status, result.exit_code, result.detail) == (NEEDS_LOGIN, 2, {"http": 401})

    asyncio.run(login_check(port=PORT, url=f"{site}/signin", probe="true"))
    result = asyncio.run(login_check(port=PORT, url=f"{site}/app", probe=PROBE))
    assert (result.status, result.exit_code, result.detail) == (OK, 0, {"room": "alpha"})
    # The probe can prove "the right room", not only "signed in".
    wrong_room = PROBE.replace("status: 'ok'", "status: j.room === 'beta' ? 'ok' : 'error'")
    assert asyncio.run(login_check(port=PORT, url=f"{site}/app", probe=wrong_room)).status == ERROR
    # Every check closed the tab it opened and touched no other.
    assert site_tabs() == set()


def test_redirect_off_site_is_needs_login_and_probe_not_run(browser, site):
    _Site.redirect_to_idp = True
    result = asyncio.run(login_check(
        port=PORT, url=f"{site}/app", probe="(() => { throw new Error('must not run') })()",
    ))
    assert result.status == NEEDS_LOGIN and "localhost" in result.url
    assert result.detail == {"reason": "redirected off site"}


def test_late_script_redirect_is_never_ok_or_error(browser, site):
    """The page finishes loading and only then redirects to sign-in from script.

    Chrome delays timers in background tabs by an unpredictable second or
    more, so whether the probe gets to run before the redirect is a race by
    nature. What must hold either way: the answer is needs_login -- from the
    redirect, or from the probe's own authenticated read -- never ok, never
    error.
    """
    for delay_ms in (0, 300, 700, 1500):
        _Site.late_js_redirect_ms = delay_ms
        result = asyncio.run(login_check(port=PORT, url=f"{site}/app", probe=PROBE))
        assert result.status == NEEDS_LOGIN, (delay_ms, result)

    # A probe that would wrongly say "ok" is overruled when the tab leaves the
    # site while it runs.
    _Site.late_js_redirect_ms = 300
    optimistic = "new Promise(r => setTimeout(() => r(true), 4000))"
    result = asyncio.run(login_check(port=PORT, url=f"{site}/app", probe=optimistic, settle=0.1))
    assert result.status == NEEDS_LOGIN and "localhost" in result.url


def test_probe_errors_and_timeouts_are_errors(browser, site):
    boom = asyncio.run(login_check(port=PORT, url=f"{site}/app", probe="nope.nope"))
    assert boom.status == ERROR and boom.exit_code == 3
    hang = asyncio.run(login_check(
        port=PORT, url=f"{site}/app", probe="new Promise(() => {})", timeout=1.5,
    ))
    assert (hang.status, hang.detail) == (ERROR, {"reason": "timed out"})
    nobody = asyncio.run(login_check(port=9363, url=f"{site}/app", probe="true"))
    assert nobody.status == ERROR


def test_wait_for_login_returns_once_signed_in(browser, site):
    async def scenario():
        waiter = asyncio.create_task(wait_for_login(
            port=PORT, url=f"{site}/app", probe=PROBE, timeout=30, poll_interval=0.3,
        ))
        await asyncio.sleep(1.0)
        assert not waiter.done()
        # "The person signs in": done here from a second tab of the same profile.
        await login_check(port=PORT, url=f"{site}/signin", probe="true")
        return await asyncio.wait_for(waiter, timeout=15)

    assert asyncio.run(scenario()).status == OK


def test_wait_for_login_times_out_as_needs_login(browser, site):
    result = asyncio.run(wait_for_login(
        port=PORT, url=f"{site}/app", probe=PROBE, timeout=1.5, poll_interval=0.3,
    ))
    assert (result.status, result.exit_code) == (NEEDS_LOGIN, 2)


_CLI = """
import sys
from chrome_agent import cli, launcher, registry
registry.REGISTRY_PATH = launcher.REGISTRY_PATH = sys.argv[1]
launcher._SESSION_ROOT = sys.argv[2]
registry.BASE_PORT, registry.MAX_PORT = 9364, 9368
sys.argv = ["chrome-agent"] + sys.argv[3:]
cli.main()
"""


@pytest.fixture
def cli(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    registry = str(tmp_path / "registry.json")
    env = dict(os.environ, CHROME_AGENT_PROFILE_ROOT=str(tmp_path / "profiles"))

    def start(*args):
        return subprocess.Popen(
            [sys.executable, "-c", _CLI, registry, str(sessions), *args],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=str(tmp_path),
        )

    def run(*args):
        proc = start(*args)
        out, err = proc.communicate(timeout=90)
        return subprocess.CompletedProcess(args, proc.returncode, out, err)

    run.start, run.registry, run.root = start, registry, tmp_path
    yield run
    for name in json.loads(open(registry).read() or "{}") if os.path.exists(registry) else []:
        run("stop", name)


def test_cli_check_on_stopped_profile_launches_checks_and_stops(cli, site):
    first = cli("login-check", "--profile", "cli-site", "--site", f"{site}/app", "--probe-expr", PROBE)
    assert first.returncode == 2, first.stderr
    assert json.loads(first.stdout)["status"] == "needs_login"
    # It stopped the browser it started, and the profile is still there.
    assert json.loads(open(cli.registry).read()) == {}
    assert os.path.isdir(cli.root / "profiles" / "cli-site")


def test_cli_argument_errors_use_the_json_exit_3_contract(cli, site):
    cases = [
        ("login-check", "--profile", "p", "--site", f"{site}/app"),
        ("login-check", "x-01", "--profile", "p", "--site", f"{site}/app", "--probe-expr", "true"),
        ("login-check", "--bogus"),
        ("login", "--profile", "p", "--site", f"{site}/app", "--probe-expr", "true", "--nope", "1"),
        ("login-check", "--profile", "p", "--probe-expr", "true"),
    ]
    for args in cases:
        result = cli(*args)
        assert result.returncode == 3, (args, result.stderr)
        assert json.loads(result.stdout)["status"] == "error"


def test_concurrent_cli_checks_on_one_stopped_profile(cli, site):
    """The first check must not stop the browser out from under the second."""
    slow = "new Promise(r => setTimeout(() => r(false), 3000))"
    procs = [
        cli.start("login-check", "--profile", "shared", "--site", f"{site}/app", "--probe-expr", PROBE),
        cli.start("login-check", "--profile", "shared", "--site", f"{site}/app", "--probe-expr", slow),
    ]
    results = [p.communicate(timeout=120) for p in procs]
    assert [p.returncode for p in procs] == [2, 2], results
    assert json.loads(open(cli.registry).read()) == {}


def test_repeated_concurrent_checks_never_leave_a_browser(cli, site):
    """Whichever check runs last stops the browser: nothing stays registered."""
    for _ in range(4):
        procs = [
            cli.start("login-check", "--profile", "shared", "--site", f"{site}/app", "--probe-expr", PROBE)
            for _ in range(3)
        ]
        results = [p.communicate(timeout=120) for p in procs]
        assert [p.returncode for p in procs] == [2, 2, 2], results
        assert json.loads(open(cli.registry).read()) == {}


def test_cli_login_refuses_headless_instance(cli, site):
    launched = json.loads(cli("launch", "--headless", "--profile", "hidden").stdout)
    result = cli("login", "--profile", "hidden", "--site", f"{site}/app",
                 "--probe-expr", "true", "--timeout", "5")
    assert result.returncode == 3 and "headless" in json.loads(result.stdout)["reason"]
    by_name = cli("login", launched["name"], "--site", f"{site}/app", "--probe-expr", "true")
    assert by_name.returncode == 3
    # An entry written by an older build has no headless field: still refused.
    registry = json.loads(open(cli.registry).read())
    del registry[launched["name"]]["headless"]
    open(cli.registry, "w").write(json.dumps(registry))
    legacy = cli("login", "--profile", "hidden", "--site", f"{site}/app",
                 "--probe-expr", "true", "--timeout", "5")
    assert legacy.returncode == 3 and "headless" in json.loads(legacy.stdout)["reason"]
    # The refusal left the running browser alone.
    assert launched["name"] in json.loads(open(cli.registry).read())
