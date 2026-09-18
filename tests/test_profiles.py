"""Tests for persistent profiles (--profile / --profile-dir).

The integration tests drive a real headless Chrome against a local HTTP server
that hands out a persistent cookie, so "the login survived" is judged by the
server's answer, not by a directory still existing. Every test isolates the
registry, the profile root and the temporary session root under tmp_path: no
test here can touch a real profile or the shared /tmp/chrome-agent root.
"""

import asyncio
import http.server
import json
import os
import subprocess
import sys
import threading

import pytest

from chrome_agent import launcher, profiles
from chrome_agent.cdp_client import CDPClient, get_ws_url
from chrome_agent.launcher import cleanup_sessions, find_chrome_binary, launch_browser
from chrome_agent.profiles import ProfileError
from chrome_agent.registry import (
    _load_registry,
    _save_registry,
    cleanup,
    deregister,
    enumerate_instances,
    lookup,
    register,
    stop,
)

PORT_A = 9351
PORT_B = 9352

needs_chrome = pytest.mark.skipif(
    find_chrome_binary() is None, reason="Chrome/Chromium not installed"
)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Isolated profile root, session root and registry for one test."""
    root = tmp_path / "profiles"
    session_root = tmp_path / "sessions"
    session_root.mkdir()
    monkeypatch.setenv("CHROME_AGENT_PROFILE_ROOT", str(root))
    monkeypatch.setattr(launcher, "_SESSION_ROOT", str(session_root))
    return {
        "root": str(root),
        "session_root": str(session_root),
        "registry": str(tmp_path / "registry.json"),
        "tmp": tmp_path,
    }


# ---------------------------------------------------------------------------
# Names and directories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["work", "x-main", "a", "deal.npp_2", "0day"])
def test_valid_names(name):
    assert profiles.validate_name(name) == name


@pytest.mark.parametrize(
    "name", ["", "Work", "../x", "a/b", ".hidden", "-lead", "a b", "x" * 64, "*"]
)
def test_invalid_names(name):
    with pytest.raises(ProfileError):
        profiles.validate_name(name)


def test_resolve_named_creates_private_dir(isolated):
    resolved = profiles.resolve_named("work")
    assert resolved.name == "work"
    assert resolved.path == os.path.join(isolated["root"], "work")
    assert os.path.isdir(resolved.path)
    if os.name == "posix":
        assert os.stat(resolved.path).st_mode & 0o777 == 0o700
    assert profiles.list_profiles() == ["work"]


def test_resolve_named_without_create(isolated):
    with pytest.raises(ProfileError, match="no such profile"):
        profiles.resolve_named("missing", create=False)


def test_resolve_named_refuses_symlink(isolated):
    os.makedirs(isolated["root"])
    outside = isolated["tmp"] / "outside"
    outside.mkdir()
    os.symlink(outside, os.path.join(isolated["root"], "sneaky"))
    with pytest.raises(ProfileError, match="symlink"):
        profiles.resolve_named("sneaky")
    assert "sneaky" not in profiles.list_profiles()


def test_resolve_dir_creates_and_accepts(isolated):
    target = isolated["tmp"] / "mine"
    resolved = profiles.resolve_dir(str(target), session_root=isolated["session_root"])
    assert resolved.name is None
    assert resolved.path == os.path.realpath(target)


def test_resolve_dir_refusals(isolated):
    session_root = isolated["session_root"]
    real = isolated["tmp"] / "real"
    real.mkdir()
    link = isolated["tmp"] / "link"
    os.symlink(real, link)
    with pytest.raises(ProfileError, match="symlink"):
        profiles.resolve_dir(str(link), session_root=session_root)
    with pytest.raises(ProfileError, match="temporary session root"):
        profiles.resolve_dir(os.path.join(session_root, "x"), session_root=session_root)
    with pytest.raises(ProfileError, match="managed profile root"):
        profiles.resolve_dir(os.path.join(isolated["root"], "x"), session_root=session_root)
    everyday = profiles._everyday_chrome_dirs()[0]
    with pytest.raises(ProfileError, match="everyday browser profile"):
        profiles.resolve_dir(os.path.join(everyday, "Default"), session_root=session_root)
    # None of the refusals created anything.
    assert not os.path.exists(os.path.join(session_root, "x"))
    assert not os.path.exists(os.path.join(isolated["root"], "x"))


def test_conflicting_options_rejected(isolated):
    async def go(**kwargs):
        return await launch_browser(
            headless=True, pin_to_desktop=False, registry_path=isolated["registry"], **kwargs
        )

    with pytest.raises(ProfileError, match="either"):
        asyncio.run(go(profile="a", profile_dir=str(isolated["tmp"] / "d")))
    for raw in (["--user-data-dir=/tmp/elsewhere"], ["--user-data-dir", "/tmp/elsewhere"]):
        with pytest.raises(ProfileError, match="--user-data-dir"):
            asyncio.run(go(profile="a", extra_args=raw))


# ---------------------------------------------------------------------------
# Registry: no deletion path ever receives a persistent directory
# ---------------------------------------------------------------------------


def _register_dead_persistent(isolated, name="keep"):
    """A registry entry for a persistent profile whose browser is gone."""
    resolved = profiles.resolve_named(name)
    marker = os.path.join(resolved.path, "marker")
    with open(marker, "w") as f:
        f.write("login state")
    info = register(
        working_dir="/home/user/proj",
        pid=2**22 + 12345,  # not a live process
        browser_version="test",
        user_data_dir=resolved.path,
        port_override=PORT_B,
        registry_path=isolated["registry"],
        pid_start="never",
        persistent=True,
        profile=name,
    )
    return info, resolved.path, marker


def test_registry_entry_hides_dir_from_deleters(isolated):
    info, path, _ = _register_dead_persistent(isolated)
    entry = _load_registry(isolated["registry"])[info.name]
    # Older chrome-agent versions delete entry["user_data_dir"]; it must be empty.
    assert entry["user_data_dir"] == ""
    assert entry["profile_dir"] == path
    assert entry["profile"] == "keep"
    looked_up = lookup(info.name, registry_path=isolated["registry"])
    assert looked_up.persistent and looked_up.profile == "keep"
    assert looked_up.user_data_dir == path


def test_cleanup_keeps_persistent_dir(isolated):
    info, _, marker = _register_dead_persistent(isolated)
    assert cleanup(registry_path=isolated["registry"]) == [info.name]
    assert os.path.exists(marker)
    cleanup_sessions(registry_path=isolated["registry"])
    assert os.path.exists(marker)


def test_deregister_keeps_persistent_dir(isolated):
    info, _, marker = _register_dead_persistent(isolated)
    assert deregister(info.name, registry_path=isolated["registry"]) is True
    assert os.path.exists(marker)


def test_stop_of_dead_instance_keeps_persistent_dir(isolated):
    info, _, marker = _register_dead_persistent(isolated)
    stop(info.name, registry_path=isolated["registry"])
    assert os.path.exists(marker)
    assert enumerate_instances(registry_path=isolated["registry"]) == []


def test_disposable_dir_still_deleted(isolated):
    """The default, temporary behaviour is unchanged."""
    session_dir = os.path.join(isolated["session_root"], "session-x")
    os.makedirs(session_dir)
    info = register(
        working_dir="/home/user/proj", pid=2**22 + 12346, browser_version="test",
        user_data_dir=session_dir, port_override=PORT_B,
        registry_path=isolated["registry"], pid_start="never",
    )
    assert cleanup(registry_path=isolated["registry"]) == [info.name]
    assert not os.path.exists(session_dir)


def test_tampered_entry_cannot_redirect_deletion(isolated):
    """An entry carrying BOTH keys is still treated as persistent."""
    info, path, marker = _register_dead_persistent(isolated)
    registry = _load_registry(isolated["registry"])
    registry[info.name]["user_data_dir"] = path
    _save_registry(registry, isolated["registry"])
    cleanup(registry_path=isolated["registry"])
    assert os.path.exists(marker)


# ---------------------------------------------------------------------------
# Real browser: a login survives stop, cleanup and relaunch
# ---------------------------------------------------------------------------


class _LoginServer(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        cookie = self.headers.get("Cookie", "")
        self.send_response(200)
        if self.path == "/login":
            self.send_header("Set-Cookie", "sid=synthetic; Max-Age=86400; Path=/")
            body = "logged-in"
        else:
            body = "in" if "sid=synthetic" in cookie else "out"
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def login_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LoginServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


async def _visit(port: int, url: str) -> str:
    """Navigate the first tab to url and return the page text."""
    async with CDPClient(ws_url=get_ws_url(port=port)) as cdp:
        await cdp.send(method="Page.enable")
        loaded = asyncio.Event()
        cdp.on("Page.loadEventFired", lambda _params: loaded.set())
        await cdp.send(method="Page.navigate", params={"url": url})
        await asyncio.wait_for(loaded.wait(), timeout=15)
        result = await cdp.send(
            method="Runtime.evaluate",
            params={"expression": "document.body.innerText", "returnByValue": True},
        )
        return result["result"]["value"].strip()


async def _launch(isolated, port, **kwargs):
    return await launch_browser(
        port_override=port, headless=True, pin_to_desktop=False,
        working_dir="/home/user/proj", registry_path=isolated["registry"], **kwargs
    )


def _stop(isolated, name):
    # stop() runs its own event loop, so it cannot be called from inside one.
    return stop(name, registry_path=isolated["registry"])


@needs_chrome
def test_login_survives_stop_cleanup_relaunch(isolated, login_server):
    first = asyncio.run(_launch(isolated, PORT_A, profile="synthetic"))
    try:
        assert first.persistent and first.profile == "synthetic" and not first.reused
        assert first.user_data_dir == os.path.join(isolated["root"], "synthetic")
        assert asyncio.run(_visit(PORT_A, f"{login_server}/whoami")) == "out"
        asyncio.run(_visit(PORT_A, f"{login_server}/login"))
        assert asyncio.run(_visit(PORT_A, f"{login_server}/whoami")) == "in"
    finally:
        _stop(isolated, first.name)
    assert os.path.isdir(first.user_data_dir)
    assert cleanup_sessions(registry_path=isolated["registry"]) == []
    assert os.path.isdir(first.user_data_dir)

    second = asyncio.run(_launch(isolated, PORT_A, profile="synthetic"))
    try:
        assert second.pid != first.pid
        assert asyncio.run(_visit(PORT_A, f"{login_server}/whoami")) == "in"
    finally:
        _stop(isolated, second.name)


@needs_chrome
def test_default_launch_is_still_temporary(isolated, login_server):
    info = asyncio.run(_launch(isolated, PORT_A))
    try:
        assert not info.persistent
        assert info.user_data_dir.startswith(isolated["session_root"])
        asyncio.run(_visit(PORT_A, f"{login_server}/login"))
    finally:
        _stop(isolated, info.name)
    assert not os.path.exists(info.user_data_dir)


@needs_chrome
def test_two_profiles_are_isolated_and_relaunch_reuses(isolated, login_server):
    a = asyncio.run(_launch(isolated, PORT_A, profile="one"))
    b = None
    try:
        asyncio.run(_visit(PORT_A, f"{login_server}/login"))
        b = asyncio.run(_launch(isolated, PORT_B, profile="two"))
        assert asyncio.run(_visit(PORT_B, f"{login_server}/whoami")) == "out"

        # Launching a running profile hands back the same instance.
        again = asyncio.run(_launch(isolated, PORT_A, profile="one"))
        assert again.reused and again.name == a.name and again.pid == a.pid
        with pytest.raises(ProfileError, match="already running"):
            asyncio.run(_launch(isolated, PORT_B + 1, profile="one"))
    finally:
        _stop(isolated, a.name)
        if b is not None:
            _stop(isolated, b.name)


@needs_chrome
def test_profile_dir_is_used_and_never_deleted(isolated, login_server):
    mine = str(isolated["tmp"] / "caller-owned")
    info = asyncio.run(_launch(isolated, PORT_A, profile_dir=mine))
    try:
        assert info.persistent and info.profile is None
        assert info.user_data_dir == os.path.realpath(mine)
        asyncio.run(_visit(PORT_A, f"{login_server}/login"))
    finally:
        _stop(isolated, info.name)
    assert os.path.isdir(mine)
    again = asyncio.run(_launch(isolated, PORT_A, profile_dir=mine))
    try:
        assert asyncio.run(_visit(PORT_A, f"{login_server}/whoami")) == "in"
    finally:
        _stop(isolated, again.name)
    assert os.path.isdir(mine)


@needs_chrome
def test_login_survives_process_kill(isolated, login_server):
    """A killed browser (no clean shutdown) leaves the profile in place.

    Only state Chrome had already written survives a kill, so the login is
    established and cleanly stopped first; the kill happens on a later run.
    """
    first = asyncio.run(_launch(isolated, PORT_A, profile="crashy"))
    asyncio.run(_visit(PORT_A, f"{login_server}/login"))
    _stop(isolated, first.name)

    second = asyncio.run(_launch(isolated, PORT_A, profile="crashy"))
    os.kill(second.pid, 9)
    for _ in range(50):
        if not launcher.process_is_running(pid=second.pid):
            break
        asyncio.run(asyncio.sleep(0.1))
    # The next launch prunes the dead entry; the profile must survive that.
    third = asyncio.run(_launch(isolated, PORT_A, profile="crashy"))
    try:
        assert third.pid != second.pid
        assert asyncio.run(_visit(PORT_A, f"{login_server}/whoami")) == "in"
    finally:
        _stop(isolated, third.name)


# ---------------------------------------------------------------------------
# CLI: profiles list / path / remove
# ---------------------------------------------------------------------------


def _cli(isolated, *args):
    env = dict(os.environ, CHROME_AGENT_PROFILE_ROOT=isolated["root"])
    return subprocess.run(
        [sys.executable, "-m", "chrome_agent", *args],
        capture_output=True, text=True, env=env,
    )


def test_cli_profiles_list_path_remove(isolated):
    profiles.resolve_named("alpha")
    profiles.resolve_named("beta")

    listed = _cli(isolated, "profiles", "list")
    assert [row["name"] for row in json.loads(listed.stdout)] == ["alpha", "beta"]
    assert _cli(isolated, "profiles", "path").stdout.strip() == isolated["root"]
    assert _cli(isolated, "profiles", "path", "alpha").stdout.strip().endswith("alpha")

    refused = _cli(isolated, "profiles", "remove", "alpha")
    assert refused.returncode == 1 and "--yes" in refused.stderr
    assert os.path.isdir(os.path.join(isolated["root"], "alpha"))

    removed = _cli(isolated, "profiles", "remove", "alpha", "--yes")
    assert removed.returncode == 0
    assert not os.path.exists(os.path.join(isolated["root"], "alpha"))
    assert os.path.isdir(os.path.join(isolated["root"], "beta"))

    missing = _cli(isolated, "profiles", "remove", "nope", "--yes")
    assert missing.returncode == 1 and "no such profile" in missing.stderr
    bad = _cli(isolated, "profiles", "remove", "../beta", "--yes")
    assert bad.returncode == 1 and os.path.isdir(os.path.join(isolated["root"], "beta"))


def test_remove_refuses_profile_held_by_live_browser(isolated):
    """Chrome's own lock naming a live process blocks removal."""
    resolved = profiles.resolve_named("held")
    os.symlink(f"somehost-{os.getpid()}", os.path.join(resolved.path, "SingletonLock"))
    result = _cli(isolated, "profiles", "remove", "held", "--yes")
    assert result.returncode == 1 and "in use" in result.stderr
    assert os.path.isdir(resolved.path)
