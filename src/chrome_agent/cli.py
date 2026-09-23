"""CLI entry point for chrome-agent.

Routes to operational commands (launch, status, attach, help, cleanup)
and one-shot CDP method calls (<instance> Domain.method '{"params": ...}').

Iteration 2: instance name routing, target specifiers, attach mode.

Usage: chrome-agent <command> [args...]
"""

import asyncio
import json
import os
import sys


# Operational commands -- checked first during routing
OPERATIONAL_COMMANDS = {"launch", "profiles", "login", "login-check", "status", "attach", "help", "cleanup", "stop", "guide", "completions"}


# Target-selection flags and the resolution each one forces. Bare --target maps
# to None: "decide by shape", handled in one place by attach.resolve_target
# rather than guessed separately at each call site.
TARGET_FLAGS = {
    "--target": None,
    "--target-id": "id",
    "--target-index": "index",
    "--url": "url",
}


def _extract_flags(argv: list[str]) -> tuple[list[str], str | None, str | None]:
    """Extract the target-selection flags from argv before routing.

    Returns (remaining_args, target_spec, target_by), where target_by is "id",
    "index" or "url" for an explicit flag and None for bare --target (resolved
    by shape when the targets are known). Flags can appear anywhere in argv.
    """
    remaining = []
    seen: list[tuple[str, str]] = []
    i = 0
    while i < len(argv):
        if argv[i] in TARGET_FLAGS and i + 1 < len(argv):
            seen.append((argv[i], argv[i + 1]))
            i += 2
        else:
            remaining.append(argv[i])
            i += 1

    if len(seen) > 1:
        names = ", ".join(flag for flag, _ in seen)
        print(f"Error: specify only one target selector (got {names})", file=sys.stderr)
        sys.exit(1)

    if not seen:
        return remaining, None, None

    flag, spec = seen[0]
    return remaining, spec, TARGET_FLAGS[flag]


def _print_guide(args: list[str]) -> None:
    """Print the bundled agent guide (AGENTS.md), or just its path.

    The guide ships inside the package, so it is available from any install
    without a checkout. `--path` prints the file location instead of its
    contents, which is usually what an agent wants: reading the file with its
    own tools beats paging 20+ KB through stdout.
    """
    from importlib.resources import files

    guide = files("chrome_agent").joinpath("AGENTS.md")
    if "--path" in args:
        print(guide)
    else:
        print(guide.read_text(encoding="utf-8"), end="")


def _run_completions(args: list[str]) -> None:
    """Print shell completions, or the live data the completion draws on.

    `zsh` prints the completion function (ship it to a directory on $fpath as
    _chrome-agent, or source it after compinit). `instances` prints one
    `name:description` line per registered instance -- the format zsh's
    _describe consumes -- and is what the completion calls on every Tab, so the
    names offered are the ones actually registered rather than a snapshot.
    """
    if not args:
        print("Error: completions requires a shell or data name", file=sys.stderr)
        print("Usage: chrome-agent completions <zsh | instances>", file=sys.stderr)
        sys.exit(1)

    what = args[0]

    if what == "zsh":
        from importlib.resources import files

        script = files("chrome_agent").joinpath("completions.zsh")
        print(script.read_text(encoding="utf-8"), end="")
        return

    if what in ("methods", "events"):
        _print_protocol_completions(
            kind="commands" if what == "methods" else "events",
            instance_name=args[1] if len(args) > 1 else None,
        )
        return

    if what == "instances":
        from .instance_status import get_instance_status

        for status in get_instance_status():
            if not status.alive:
                description = f"port {status.port} -- DEAD"
            else:
                count = len(status.targets)
                description = f"port {status.port} -- {count} tab{'' if count == 1 else 's'}"
            print(f"{status.name}:{description}")
        return

    print(f"Error: unknown completions target: {what}", file=sys.stderr)
    print(
        "Usage: chrome-agent completions <zsh | instances | methods | events> [<instance>]",
        file=sys.stderr,
    )
    sys.exit(1)


def _print_protocol_completions(*, kind: str, instance_name: str | None) -> None:
    """Print `Domain.member:description` lines for the live protocol.

    Read from the running browser rather than a bundled list, so the candidates
    match the protocol *this* Chrome implements -- including surface newer than
    any snapshot shipped with chrome-agent. Any live instance answers, since the
    schema is identical across instances of the same browser.

    The result is cached on disk, keyed by the browser version the registry
    already records -- so a hit needs no browser contact at all, and a Chrome
    upgrade invalidates it by changing the key. The fetch itself is only ~7 ms
    and would not justify a cache for Tab alone; what does is that zsh runs
    completion on every *keystroke* when autosuggestions use the completion
    strategy, so an uncached lookup means a process spawn and an HTTP round
    trip per character typed.

    Prints nothing and exits 0 when no browser is reachable -- at Tab time the
    right answer is no candidates, not an error in the middle of a command line.
    """
    from .protocol import fetch_protocol_schema

    resolved = _resolve_instance_for_protocol(instance_name=instance_name)
    if resolved is None:
        return
    port, browser_version = resolved

    cache = _protocol_cache_path(browser_version=browser_version, kind=kind)
    if cache is not None and cache.exists():
        try:
            sys.stdout.write(cache.read_text(encoding="utf-8"))
            return
        except OSError:
            pass  # unreadable cache is not a reason to fail; re-fetch below

    try:
        schema = fetch_protocol_schema(port=port)
    except (ConnectionError, RuntimeError, OSError):
        return

    lines = []
    for domain in schema.get("domains", []):
        name = domain.get("domain", "")
        for member in domain.get(kind, []):
            # _describe splits each line on its FIRST colon, so colons inside a
            # description are harmless; newlines are not -- a CDP description
            # can run to several lines, and each would read as its own bogus
            # candidate. Join rather than truncate to the first line: CDP wraps
            # its prose at arbitrary points, so a first-line cut ends mid
            # sentence and reads as a bug in the menu.
            description = " ".join((member.get("description") or "").split())
            lines.append(f"{name}.{member.get('name', '')}:{description}")

    text = "".join(f"{line}\n" for line in lines)
    sys.stdout.write(text)

    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            # Write via a temp file in the same directory and rename, so a
            # completion racing this one never reads a half-written cache.
            temp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
            temp.write_text(text, encoding="utf-8")
            os.replace(temp, cache)
        except OSError:
            pass  # a cache we cannot write is not an error worth surfacing


def _resolve_instance_for_protocol(
    *, instance_name: str | None
) -> tuple[int, str] | None:
    """Resolve to (port, browser_version), or None when nothing can answer.

    Any live instance will do when none is named: the protocol schema is a
    property of the browser build, not of the instance.
    """
    from .registry import enumerate_instances, lookup

    try:
        if instance_name is not None:
            info = lookup(instance_name=instance_name)
            return (info.port, info.browser_version) if info.alive else None
        for info in enumerate_instances():
            if info.alive:
                return info.port, info.browser_version
    except Exception:
        return None
    return None


def _protocol_cache_path(*, browser_version: str, kind: str):
    """Where the protocol completions for this browser build are cached.

    None when the registry has no version to key on -- better to re-fetch every
    time than to serve one browser's protocol under another's name.
    """
    import re
    from pathlib import Path

    if not browser_version:
        return None
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", browser_version)
    return Path(base) / "chrome-agent" / f"protocol-{safe}-{kind}.txt"


def _print_static_usage() -> None:
    """Print static usage when no browser is available for protocol listing."""
    print("chrome-agent -- CLI for AI agents to control Chrome via CDP\n")
    print("Usage: chrome-agent <command> [args...]\n")
    print("Operational commands:")
    print("  launch [--port PORT] [--fingerprint PATH] [--headless] [--no-window-border] [-- CHROME_ARGS]  Launch Chrome")
    print("         [--profile NAME | --profile-dir PATH]           Keep logins: use a persistent profile")
    print("  profiles [list | path [NAME] | clone SOURCE NEW | remove NAME --yes]  Manage persistent named profiles")
    print("  login-check (<instance> | --profile NAME | --profile-dir PATH) --site URL (--probe FILE | --probe-expr JS)")
    print("                                                         Is the site signed in? exit 0 yes, 2 needs login, 3 error")
    print("  login (same options) [--timeout SECONDS]               Open the site visibly; wait until a person has signed in")
    print("  status [<instance>]                                    List instances and targets")
    print("  attach <instance> [+Event ...] [TARGET]                Attach for events")
    print("  help [<instance>] [Domain | Domain.method]             Protocol discovery")
    print("  stop <instance> [TARGET]                               Stop a browser, or close one tab")
    print("  cleanup                                                Remove stale instances")
    print("  guide [--path]                                         Print this tool's agent guide")
    print("  completions <zsh|instances|methods|events> [<instance>]  Shell completion and its data")
    print()
    print("  --version, -V                                          Show version and exit")
    print()
    print("Instance patterns (quote them -- the shell expands an unquoted glob first):")
    print("  '<glob>'               Any instance argument may be a glob (*, ?, [abc])")
    print("                         status and stop act on every match;")
    print("                         attach, help and one-shots require it to match exactly one")
    print()
    print("Target selectors (TARGET -- pick one; usable on attach, stop and one-shots):")
    print("  --target SPEC          Tab index if SPEC is fewer than 8 digits, else a target-id prefix")
    print("  --target-id ID         Always a target-id prefix (the `id` or `full_id` from status)")
    print("  --target-index N       Always the 1-based index from status")
    print("  --url SUBSTRING        The tab whose URL contains SUBSTRING")
    print()
    print("CDP one-shot commands:")
    print("  <instance> Domain.method '{\"param\": \"value\"}'         Send a single CDP command")
    print("  Domain.method '{\"param\": \"value\"}'                    (auto-selects instance)")
    print()
    print("Examples:")
    print("  chrome-agent launch --headless")
    print("  chrome-agent launch --profile work                      # logins survive stop and restart")
    print("  chrome-agent status")
    print("  chrome-agent attach mysite-01 +Page.loadEventFired")
    print("  chrome-agent mysite-01 Page.navigate '{\"url\": \"https://example.com\"}'")
    print("  chrome-agent help Page.navigate")
    print("  chrome-agent stop 'mysite-*'")


async def _run_launch(args: list[str]) -> None:
    """Launch a browser with CDP enabled."""
    from .launcher import BrowserNotFoundError, launch_browser
    from .profiles import ProfileError

    profile = None
    profile_dir = None
    fingerprint_path = None
    headless = False
    port_override = None
    window_border = True
    extra_args = []
    i = 0
    while i < len(args):
        if args[i] == "--":
            # Everything after -- is passed through to Chrome
            extra_args = args[i + 1:]
            break
        elif args[i] == "--fingerprint" and i + 1 < len(args):
            fingerprint_path = args[i + 1]
            i += 2
        elif args[i] == "--profile" and i + 1 < len(args):
            profile = args[i + 1]
            i += 2
        elif args[i] == "--profile-dir" and i + 1 < len(args):
            profile_dir = args[i + 1]
            i += 2
        elif args[i] == "--headless":
            headless = True
            i += 1
        elif args[i] == "--no-window-border":
            window_border = False
            i += 1
        elif args[i] == "--port" and i + 1 < len(args):
            try:
                port_override = int(args[i + 1])
            except ValueError:
                print(f"Error: invalid port: {args[i + 1]}", file=sys.stderr)
                sys.exit(1)
            i += 2
        else:
            print(f"Error: unknown launch option: {args[i]}", file=sys.stderr)
            sys.exit(1)

    try:
        result = await launch_browser(
            port_override=port_override,
            fingerprint=fingerprint_path,
            headless=headless,
            extra_args=extra_args,
            window_border=window_border,
            profile=profile,
            profile_dir=profile_dir,
        )
    except (BrowserNotFoundError, ProfileError, RuntimeError, TimeoutError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if sys.stdout.isatty():
        print(f"Browser {'already running' if result.reused else 'launched'}: {result.name}")
        print(f"  Port:    {result.port}")
        print(f"  PID:     {result.pid}")
        print(f"  Version: {result.browser_version}")
        if result.persistent:
            print(f"  Profile: {result.profile or result.user_data_dir}")
    else:
        payload = {
            "name": result.name,
            "port": result.port,
            "pid": result.pid,
            "browser_version": result.browser_version,
        }
        if result.persistent:
            payload["profile"] = result.profile
            payload["profile_dir"] = result.user_data_dir
            payload["reused"] = result.reused
        print(json.dumps(payload))


async def _run_login(args: list[str], wait: bool) -> None:
    """login-check / login: judge sign-in state with a caller-supplied probe."""
    from .launcher import BrowserNotFoundError, launch_browser
    from .login import ERROR, LoginResult, login_check, wait_for_login
    from .profiles import ProfileError
    from .registry import InstanceNotFoundError, resolve_instance_name, lookup, stop

    command = "login" if wait else "login-check"
    # --site, not --url: --url is already the global "pick the tab whose URL
    # contains..." target selector and is consumed before command routing.
    opts = {"--profile": None, "--profile-dir": None, "--site": None,
            "--probe": None, "--probe-expr": None, "--timeout": None}
    instance = None
    unknown = None
    i = 0
    while i < len(args):
        if args[i] in opts and i + 1 < len(args):
            opts[args[i]] = args[i + 1]
            i += 2
        elif not args[i].startswith("-") and instance is None:
            instance = args[i]
            i += 1
        else:
            unknown = args[i]
            break

    def fail(message: str) -> None:
        result = LoginResult(ERROR, url=opts["--site"] or "", detail={"reason": message})
        print(result.to_json() if not sys.stdout.isatty() else f"error: {message}")
        sys.exit(result.exit_code)

    if unknown is not None:
        fail(f"unknown {command} option: {unknown}")
    targets = [v for v in (instance, opts["--profile"], opts["--profile-dir"]) if v]
    if len(targets) != 1:
        fail("name exactly one of: <instance>, --profile NAME, --profile-dir PATH")
    if not opts["--site"]:
        fail("--site URL is required")
    if bool(opts["--probe"]) == bool(opts["--probe-expr"]):
        fail("give exactly one of --probe FILE or --probe-expr JS")
    try:
        probe = opts["--probe-expr"] or open(opts["--probe"]).read()
        timeout = float(opts["--timeout"]) if opts["--timeout"] else (600.0 if wait else 30.0)
    except (OSError, ValueError) as exc:
        fail(str(exc))

    async def run(info) -> LoginResult:
        if wait:
            if info.headless:
                fail(f"instance {info.name} is headless, so nobody can sign in to it; stop it and run login again")
            return await wait_for_login(port=info.port, url=opts["--site"], probe=probe, timeout=timeout)
        return await login_check(port=info.port, url=opts["--site"], probe=probe, timeout=timeout)

    try:
        if instance is not None:
            info = lookup(resolve_instance_name(instance))
            if not info.alive:
                fail(f"instance {info.name} is not running")
            result = await run(info)
        else:
            # A check on a profile nobody is running starts a headless browser
            # and stops it again; waiting for a person needs a visible window.
            # The use lock makes that start-check-stop one unit per profile:
            # without it a second check reuses the first one's browser and has
            # it stopped underneath it.
            from .launcher import _SESSION_ROOT, _async_flock
            from .profiles import lock_path, require_gui_session, resolve_dir, resolve_named
            # Before resolving: resolving creates the profile and lock
            # directories, and a refused launch must leave nothing behind.
            require_gui_session()
            resolved = (
                resolve_named(opts["--profile"]) if opts["--profile"]
                else resolve_dir(opts["--profile-dir"], session_root=_SESSION_ROOT)
            )
            use_lock = _async_flock(lock_path(resolved.path) + ".use")
            if wait:
                async with use_lock:
                    info = await launch_browser(
                        profile=opts["--profile"], profile_dir=opts["--profile-dir"],
                    )
                result = await run(info)
            else:
                async with use_lock:
                    result, info = await _checked_headless(
                        opts=opts, run=run, timeout=timeout, launch_browser=launch_browser, stop=stop,
                    )
    except (InstanceNotFoundError, BrowserNotFoundError, ProfileError, RuntimeError, TimeoutError) as exc:
        fail(str(exc))
    result.detail.setdefault("instance", info.name)

    if sys.stdout.isatty():
        reason = result.detail.get("reason")
        print(f"{result.status}: {result.url}" + (f" ({reason})" if reason else ""))
    else:
        print(result.to_json())
    sys.exit(result.exit_code)


async def _checked_headless(*, opts, run, timeout, launch_browser, stop):
    """Start a headless browser, run the check, stop the browser -- bounded.

    ``timeout`` is a wall-clock bound over launch, connect and probe together.
    A profile heavy with extensions can bring up a headless Chrome whose CDP
    endpoint never answers; without this bound the command hung for minutes
    with no output. On expiry the browser this call started is stopped and
    the result is an error, never a hang.
    """
    from .login import ERROR, LoginResult

    info = None
    launch_done = asyncio.Event()

    async def go():
        nonlocal info
        info = await launch_browser(
            headless=True, profile=opts["--profile"], profile_dir=opts["--profile-dir"],
        )
        launch_done.set()
        return await run(info)

    try:
        result = await asyncio.wait_for(go(), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        phase = "probe" if launch_done.is_set() else "headless launch"
        result = LoginResult(ERROR, url=opts["--site"], detail={"reason": f"timed out during {phase} after {timeout:g}s"})
    if info is not None and not info.reused and not _has_open_pages(info.port):
        if not await _shutdown_own_browser(info):
            result.detail["cleanup"] = (
                f"browser for {info.name} on port {info.port} did not exit; run: chrome-agent stop {info.name}"
            )
    if info is None:
        # Launch itself timed out or failed: nothing registered to stop.
        return result, _Anon()
    return result, info


class _Anon:
    name = "(not launched)"


async def _shutdown_own_browser(info) -> bool:
    """Stop the browser THIS command started, without ever blocking forever.

    ``registry.stop`` talks to the browser and waits on it; a hung Chrome
    (seen with a headless profile whose extensions wedged its DevTools
    server) never answers, so that path cannot be used from a bounded
    command. This one asks politely once with a short bound, then signals
    only processes that carry the exact profile directory and port this
    launch passed, and finally drops the registry entry. Returns True when
    nothing of the browser is left.
    """
    from .cdp_client import CDPClient, get_ws_url
    from .utils import browser_processes, kill_browser_processes
    from .registry import _port_is_listening, deregister

    async def polite_close():
        ws_url = get_ws_url(port=info.port, target_type="browser")
        async with CDPClient(ws_url=ws_url) as cdp:
            await cdp.send(method="Browser.close")

    try:
        await asyncio.wait_for(polite_close(), timeout=5)
    except Exception:
        pass

    def gone() -> bool:
        # A plain socket check, not an HTTP request: a wedged DevTools server
        # accepts connections and then never answers.
        return not _port_is_listening(info.port) and not browser_processes(
            user_data_dir=info.user_data_dir, port=info.port,
        )

    for signal_number in (None, 15, 9, 9):
        for _ in range(10):
            if gone():
                deregister(info.name)
                return True
            await asyncio.sleep(0.2)
        if signal_number is not None:
            kill_browser_processes(user_data_dir=info.user_data_dir, port=info.port, signal_number=signal_number)
    deregister(info.name)
    return gone()


def _has_open_pages(port: int) -> bool:
    """Whether anyone has opened a real page in this browser.

    Guards the stop after a check: if something else launched this profile in
    the meantime it got this same browser back, and is now using it.
    """
    from .cdp_client import get_targets
    try:
        targets = get_targets(port=port)
    except ConnectionError:
        return False
    return any(
        t.get("type") == "page" and t.get("url", "").startswith(("http://", "https://", "file://"))
        for t in targets
    )


def _run_profiles(args: list[str]) -> None:
    """Manage persistent named profiles: list, path, remove."""
    from .profiles import (
        ProfileError,
        launch_lock,
        list_profiles,
        profile_root,
        remove_named,
        resolve_named,
        singleton_holder_pid,
    )
    from .registry import find_by_profile_dir
    from .utils import process_is_running

    action = args[0] if args else "list"
    try:
        if action == "list" and len(args) <= 1:
            names = list_profiles()
            if sys.stdout.isatty():
                print("\n".join(names) if names else "No profiles. Create one with: chrome-agent launch --profile NAME")
                return
            rows = []
            for name in names:
                running = find_by_profile_dir(resolve_named(name, create=False).path)
                rows.append({"name": name, "instance": running.name if running else None})
            print(json.dumps(rows))
        elif action == "path" and len(args) <= 2:
            print(resolve_named(args[1], create=False).path if len(args) == 2 else profile_root())
        elif action == "clone" and len(args) == 3:
            from .profiles import clone_named, resolve_named as _resolve
            source, new = args[1], args[2]
            src = _resolve(source, create=False)
            with launch_lock(src.path):
                running = find_by_profile_dir(src.path)
                if running is not None:
                    raise ProfileError(f"profile {source!r} is in use by instance {running.name}; stop it first")
                path = clone_named(source, new)
            if sys.stdout.isatty():
                print(f"Cloned profile {source} -> {new}")
                print(f"  Path: {path}")
            else:
                print(json.dumps({"source": source, "profile": new, "profile_dir": path}))
        elif action == "remove" and len(args) >= 2:
            name = args[1]
            if args[2:] != ["--yes"]:
                print(
                    f"Error: removing profile {name!r} deletes its logins and data. "
                    f"Re-run with --yes to confirm: chrome-agent profiles remove {name} --yes",
                    file=sys.stderr,
                )
                sys.exit(1)
            resolved = resolve_named(name, create=False)
            with launch_lock(resolved.path):
                running = find_by_profile_dir(resolved.path)
                if running is not None:
                    raise ProfileError(
                        f"profile {name!r} is in use by instance {running.name}; stop it first"
                    )
                holder = singleton_holder_pid(resolved.path)
                if holder is not None and process_is_running(pid=holder):
                    raise ProfileError(
                        f"profile {name!r} is in use by a browser (pid {holder}); close it first"
                    )
                remove_named(name)
            print(f"Removed profile {name}")
        else:
            print("Usage: chrome-agent profiles [list | path [NAME] | clone SOURCE NEW | remove NAME --yes]", file=sys.stderr)
            sys.exit(1)
    except ProfileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _run_status(args: list[str]) -> None:
    """List running browser instances and their targets."""
    from .instance_status import (
        format_status_json,
        format_status_text,
        get_instance_status,
    )
    from .registry import InstanceNotFoundError

    instance_name = args[0] if args else None

    try:
        statuses = get_instance_status(instance_name=instance_name)
    except InstanceNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not statuses and instance_name is None:
        print("No instances registered. Launch one with: chrome-agent launch")
        return

    if sys.stdout.isatty():
        print(format_status_text(statuses))
    else:
        print(format_status_json(statuses))


async def _run_attach(args: list[str], target_spec: str | None, target_by: str | None) -> None:
    """Attach to a browser instance for event observation."""
    from .attach import run_attach

    if not args:
        print("Error: attach requires an instance name", file=sys.stderr)
        print("Usage: chrome-agent attach <instance> [+Event ...]", file=sys.stderr)
        sys.exit(1)

    from .registry import (
        AmbiguousInstanceError,
        InstanceNotFoundError,
        resolve_instance_name,
    )

    try:
        instance_name = resolve_instance_name(name_or_pattern=args[0])
    except (AmbiguousInstanceError, InstanceNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    subscriptions = [arg[1:] for arg in args[1:] if arg.startswith("+")]

    try:
        await run_attach(
            instance_name=instance_name,
            subscriptions=subscriptions,
            target_spec=target_spec,
            target_by=target_by,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _run_help(args: list[str]) -> None:
    """Protocol discovery / help.

    Disambiguation: if the first arg exists in the registry, treat it
    as an instance name. Otherwise treat it as a domain query.
    """
    from .protocol import discover_protocol

    if not args:
        try:
            discover_protocol()
        except ConnectionError:
            _print_static_usage()
        return

    # Try to disambiguate: is args[0] an instance name or a domain query?
    instance_name = None
    query = None

    from .registry import AmbiguousInstanceError, resolve_instance_name

    try:
        # Resolves a literal name or a single-match glob; a pattern matching
        # several instances is an error rather than a silent pick, even though
        # the protocol schema is identical across them.
        instance_name = resolve_instance_name(name_or_pattern=args[0])
        query = args[1] if len(args) > 1 else None
    except AmbiguousInstanceError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        # Not in registry -- treat as domain query
        query = args[0]

    try:
        discover_protocol(instance_name=instance_name, query=query)
    except ConnectionError:
        # A query was given (a Domain/method to look up), but no browser could
        # answer it. Emit a clear, actionable error instead of silently falling
        # through to the generic usage banner.
        if instance_name:
            print(f"Error: browser for instance '{instance_name}' is not responding", file=sys.stderr)
        else:
            print(
                "Error: no running browser to query for protocol help. "
                "Start one with: chrome-agent launch",
                file=sys.stderr,
            )
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _run_stop(args: list[str], target_spec: str | None, target_by: str | None) -> None:
    """Stop one or more browser instances, or close a specific tab.

    The instance argument may be a glob pattern, in which case every matching
    instance is stopped -- the matched names are printed first, so a broad
    pattern leaves a record of what it swept up. A target selector closes one
    tab, which is only meaningful against a single browser, so it is refused
    when the pattern matches several.
    """
    from .registry import InstanceNotFoundError, resolve_instance_names, stop

    if not args:
        print("Error: stop requires an instance name", file=sys.stderr)
        print("Usage: chrome-agent stop <instance> [--target SPEC | --target-id ID | --target-index N | --url SUBSTRING]", file=sys.stderr)
        sys.exit(1)

    try:
        matched = resolve_instance_names(name_or_pattern=args[0])
    except InstanceNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if len(matched) > 1:
        if target_spec is not None:
            names = ", ".join(matched)
            print(
                f"Error: a target selector closes one tab, but pattern "
                f"'{args[0]}' matches {len(matched)} instances: {names}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Pattern '{args[0]}' matched {len(matched)} instances:")
        print(f"  {', '.join(matched)}")
        failures = 0
        for name in matched:
            try:
                print(stop(instance_name=name))
            except Exception as exc:
                print(f"Error stopping {name}: {exc}", file=sys.stderr)
                failures += 1
        if failures:
            sys.exit(1)
        return

    instance_name = matched[0]

    # If a target specifier was provided, resolve it to a target ID
    resolved_target_id = None
    if target_spec is not None:
        from .attach import resolve_target
        from .cdp_client import CDPClient, get_ws_url
        from .registry import lookup

        try:
            info = lookup(instance_name=instance_name)
        except InstanceNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        async def _get_targets():
            browser_ws = get_ws_url(port=info.port, target_type="browser")
            async with CDPClient(ws_url=browser_ws) as cdp:
                result = await cdp.send(method="Target.getTargets")
                return sorted(
                    (t for t in result.get("targetInfos", []) if t.get("type") == "page"),
                    key=lambda t: t.get("targetId", ""),
                )

        import asyncio
        try:
            page_targets = asyncio.run(_get_targets())
        except (ConnectionError, RuntimeError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        try:
            resolved_target_id = resolve_target(
                page_targets=page_targets,
                target_spec=target_spec,
                target_by=target_by,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    try:
        result = stop(instance_name=instance_name, target_id=resolved_target_id)
        print(result)
    except InstanceNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def _run_cleanup() -> None:
    """Clean up stale instances and session directories."""
    from .launcher import cleanup_sessions

    removed = cleanup_sessions()
    if removed:
        print(f"Cleaned up {len(removed)} stale instance(s): {', '.join(removed)}")
    else:
        print("No stale instances found")


async def _run_cdp_one_shot(
    instance_name: str | None,
    method: str,
    params_str: str | None,
    target_spec: str | None,
    target_by: str | None,
) -> None:
    """Send a single CDP command via browser-level WS + Target.attachToTarget."""
    from .attach import AmbiguousTargetError, TargetNotFoundError
    from .cdp_client import CDPClient, get_ws_url
    from .errors import CDPError

    # Resolve instance
    if instance_name is not None:
        from .registry import (
            AmbiguousInstanceError,
            InstanceNotFoundError,
            lookup,
            resolve_instance_name,
        )
        try:
            resolved = resolve_instance_name(name_or_pattern=instance_name)
            info = lookup(instance_name=resolved)
        except (AmbiguousInstanceError, InstanceNotFoundError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        port = info.port
    else:
        # Default instance resolution: auto-select single live instance
        from .registry import enumerate_instances
        instances = enumerate_instances()
        live = [i for i in instances if i.alive]
        if len(live) == 0:
            print("Error: no instances registered. Launch one with: chrome-agent launch", file=sys.stderr)
            sys.exit(1)
        elif len(live) > 1:
            names = ", ".join(i.name for i in live)
            print(f"Error: multiple instances running. Specify one: {names}", file=sys.stderr)
            sys.exit(1)
        port = live[0].port

    # Parse params
    params = None
    if params_str is not None:
        try:
            params = json.loads(params_str)
        except json.JSONDecodeError as exc:
            print(f"Error: invalid JSON parameters: {exc}", file=sys.stderr)
            sys.exit(1)
        if not isinstance(params, dict):
            print("Error: parameters must be a JSON object", file=sys.stderr)
            sys.exit(1)

    # Connect to browser-level WebSocket
    try:
        browser_ws_url = get_ws_url(port=port, target_type="browser")
    except (ConnectionError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        async with CDPClient(ws_url=browser_ws_url) as cdp:
            # Resolve target
            targets_result = await cdp.send(method="Target.getTargets")
            page_targets = sorted(
                (t for t in targets_result.get("targetInfos", [])
                 if t.get("type") == "page"),
                key=lambda t: t.get("targetId", ""),
            )

            if not page_targets:
                print("Error: no page targets in browser", file=sys.stderr)
                sys.exit(1)

            from .attach import resolve_target
            target_id = resolve_target(
                page_targets=page_targets,
                target_spec=target_spec,
                target_by=target_by,
            )

            # Create isolated session
            session_result = await cdp.send(
                method="Target.attachToTarget",
                params={"targetId": target_id, "flatten": True},
            )
            session_id = session_result["sessionId"]

            try:
                result = await cdp.send(
                    method=method,
                    params=params,
                    session_id=session_id,
                )
                print(json.dumps(result, indent=2))
            finally:
                try:
                    await cdp.send(
                        method="Target.detachFromTarget",
                        params={"sessionId": session_id},
                    )
                except Exception:
                    pass

    except (AmbiguousTargetError, TargetNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except CDPError as exc:
        print(f"CDP error {exc.code}: {exc.message}", file=sys.stderr)
        sys.exit(1)
    except ConnectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """CLI entry point."""
    # Phase 0: Extract the target-selection flags before routing
    args, target_spec, target_by = _extract_flags(sys.argv[1:])

    if args and args[0] in ("--version", "-V"):
        from . import __version__
        print(f"chrome-agent {__version__}")
        sys.exit(0)

    if not args or args[0] in ("-h", "--help"):
        _print_static_usage()
        sys.exit(0)

    command = args[0]
    rest = args[1:]

    # Route operational commands first
    if command in OPERATIONAL_COMMANDS:
        if command == "launch":
            asyncio.run(_run_launch(args=rest))
        elif command == "profiles":
            _run_profiles(rest)
        elif command in ("login", "login-check"):
            asyncio.run(_run_login(rest, wait=command == "login"))
        elif command == "status":
            _run_status(args=rest)
        elif command == "attach":
            asyncio.run(_run_attach(args=rest, target_spec=target_spec, target_by=target_by))
        elif command == "help":
            _run_help(args=rest)
        elif command == "stop":
            _run_stop(args=rest, target_spec=target_spec, target_by=target_by)
        elif command == "cleanup":
            _run_cleanup()
        elif command == "guide":
            _print_guide(args=rest)
        elif command == "completions":
            _run_completions(args=rest)
        return

    # Disambiguate "instance name" vs "bare Domain.method":
    #   - Registered instance names (e.g. from a directory basename like
    #     "aroundchicago.tech-01") may contain dots, so a naive "." check
    #     misroutes them as CDP methods.
    #   - Resolve by checking the registry first. If the first arg matches a
    #     known instance, route as instance. Otherwise, apply the
    #     Domain.method heuristic (PascalCase domain + dot + camelCase method).
    #   - A glob pattern is always an instance argument: method names are not
    #     globbable, so a wildcard is unambiguous intent to select instances.
    from .registry import enumerate_instances, is_pattern

    known_instances = {i.name for i in enumerate_instances()}
    is_known_instance = command in known_instances or is_pattern(command)
    looks_like_method = (
        "." in command
        and command.count(".") == 1
        and command.split(".")[0].isidentifier()
        and command.split(".")[0][:1].isupper()
    )

    if not is_known_instance and looks_like_method:
        method = command
        params_str = rest[0] if rest else None
        asyncio.run(_run_cdp_one_shot(
            instance_name=None,
            method=method,
            params_str=params_str,
            target_spec=target_spec,
            target_by=target_by,
        ))
        return

    # Otherwise: first arg is instance name, second should be a CDP method
    instance_name = command
    if not rest or "." not in rest[0]:
        print(f"Error: expected Domain.method after instance name '{instance_name}'", file=sys.stderr)
        print("Usage: chrome-agent <instance> Domain.method '{\"params\"}'", file=sys.stderr)
        sys.exit(1)

    method = rest[0]
    params_str = rest[1] if len(rest) > 1 else None
    asyncio.run(_run_cdp_one_shot(
        instance_name=instance_name,
        method=method,
        params_str=params_str,
        target_spec=target_spec,
        target_by=target_by,
    ))
