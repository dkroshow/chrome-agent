"""Shared utilities for chrome-agent.

Functions in this module are used by multiple feature modules and must
not have dependencies on other chrome-agent modules to avoid circular imports.
"""

import os
import subprocess
import sys


def process_is_running(pid: int) -> bool:
    """Check if a process with the given PID is running.

    Uses signal 0 (existence check without killing).
    Returns True if the process exists, False if it does not.
    Returns True on PermissionError (process exists but we can't signal it).
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def process_is_ours(pid: int, expected_start: str | None = None) -> bool:
    """Whether ``pid`` is a live process belonging to this user -- and, when
    an ``expected_start`` token is given, the same process originally
    recorded, not a later occupant of a recycled PID.

    chrome-agent launches Chrome as the invoking user, so a PID this user
    cannot signal is never one of our browsers. That is why this differs from
    ``process_is_running``, which deliberately treats PermissionError as
    "running": here PermissionError means "running but not ours". The
    distinction matters for PIDs recorded from inside a PID-namespaced
    sandbox (e.g. an agent CLI's bwrap): the namespace-local PID aliases to
    an unrelated host process -- often a root kernel thread -- which must
    never be treated as, or signalled as, our browser.
    """
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    if expected_start is not None:
        actual = process_start_time(pid=pid)
        if actual is not None and actual != expected_start:
            return False
    return True


def process_start_time(pid: int) -> str | None:
    """Opaque start-time identity token for a live process, or None.

    Two processes that ever shared a PID are told apart by start time, so
    (pid, start-token) is a durable process identity across PID reuse.
    Linux: field 22 of ``/proc/<pid>/stat`` (clock ticks since boot).
    Elsewhere: ``ps -o lstart=`` where available. Returns None when
    undeterminable -- callers treat that as "no identity evidence", never as
    a mismatch.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
        # Fields after the (comm) -- which may itself contain spaces/parens --
        # start at field 3; starttime is field 22, i.e. index 19 here.
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        pass
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = result.stdout.strip()
        return value or None
    except Exception:
        return None


def process_argv(pid: int) -> list[str] | None:
    """The exact argument vector of ``pid``, or None when it cannot be read.

    Flattened ``ps`` text is ambiguous: arguments are space-joined without
    quoting, so a path containing spaces -- or one that itself contains
    ``--no-first-run`` -- cannot be told apart from a different argument
    sequence. The kernel keeps the real vector: ``kern.procargs2`` on macOS,
    ``/proc/<pid>/cmdline`` on Linux.
    """
    if sys.platform == "linux":
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read()
        except OSError:
            return None
        return _parse_linux_cmdline(raw)
    if sys.platform == "darwin":
        return _darwin_procargs(pid)
    return None


def _parse_linux_cmdline(raw: bytes) -> list[str] | None:
    """NUL-separated argv, or None when it is not unambiguous.

    Chrome on Linux rewrites its argv into one space-joined entry. Splitting
    that on spaces would recreate exactly the ambiguity this function exists
    to remove, so a single entry containing spaces is reported as unknown.
    """
    parts = [p.decode(errors="replace") for p in raw.split(b"\0") if p]
    if len(parts) == 1 and " " in parts[0]:
        return None
    return parts or None


def _darwin_procargs(pid: int) -> list[str] | None:
    import ctypes
    import ctypes.util

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    except OSError:
        return None
    CTL_KERN, KERN_PROCARGS2 = 1, 49
    mib = (ctypes.c_int * 3)(CTL_KERN, KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0 or size.value < 4:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
        return None
    raw = buf.raw[:size.value]
    argc = int.from_bytes(raw[:4], sys.byteorder)
    rest = raw[4:]
    # exec path, NUL-terminated, then NUL padding, then argv[0..argc-1].
    rest = rest[rest.index(b"\0"):].lstrip(b"\0") if b"\0" in rest else b""
    args = rest.split(b"\0")
    return [a.decode(errors="replace") for a in args[:argc]]


def _candidate_pids() -> list[int]:
    if sys.platform == "linux":
        try:
            return [int(e) for e in os.listdir("/proc") if e.isdigit()]
        except OSError:
            return []
    try:
        out = subprocess.run(["ps", "-axo", "pid="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(t) for t in out.split() if t.isdigit()]


def browser_processes(user_data_dir: str, port: int) -> list[int]:
    """PIDs of our user's browser processes launched with exactly these two
    arguments, ``--user-data-dir=<dir>`` and ``--remote-debugging-port=<port>``,
    compared as whole argv entries. Helper processes (``--type=``) excluded.

    The PID chrome-agent recorded is not always the browser: on macOS the
    launched process can hand off to a child and exit, so a kill of the
    recorded PID alone leaves the real browser running. Returns [] where the
    argument vector cannot be read; never guesses from flattened text.
    """
    dir_arg, port_arg = f"--user-data-dir={user_data_dir}", f"--remote-debugging-port={port}"
    found = []
    for pid in _candidate_pids():
        argv = process_argv(pid)
        if not argv or dir_arg not in argv or port_arg not in argv:
            continue
        if any(a.startswith("--type=") for a in argv):
            continue
        if process_is_ours(pid=pid):
            found.append(pid)
    return found


def kill_browser_processes(user_data_dir: str, port: int, signal_number: int = 9) -> list[int]:
    """Signal every process ``browser_processes`` finds. Returns the PIDs hit."""
    hit = []
    for pid in browser_processes(user_data_dir=user_data_dir, port=port):
        try:
            os.kill(pid, signal_number)
            hit.append(pid)
        except ProcessLookupError:
            pass
    return hit


# --- macOS code-sign clones -------------------------------------------------
# At startup Chrome on macOS clones its own app bundle (APFS clonefile) under
# the user's temp area so code-signature checks keep working if the app is
# updated in place while running. A normal shutdown deletes the clone; a
# killed Chrome cannot, and Chrome never sweeps old ones. Every forced kill
# chrome-agent performs would therefore leave a 2 GiB-apparent directory
# behind. chrome-agent attributes the clone to its browser at launch (the one
# entry that appears while it starts, under the launch lock) and removes it
# after a forced kill once nothing holds it.

_CLONE_DIRNAME = "com.google.Chrome.code_sign_clone"


def code_sign_clone_root() -> str | None:
    """Chrome's clone directory for this user (macOS), or None."""
    if sys.platform != "darwin":
        return None
    try:
        tmp = subprocess.run(["getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not tmp:
        return None
    root = os.path.join(os.path.dirname(tmp.rstrip("/")), "X", _CLONE_DIRNAME)
    return root if os.path.isdir(root) else None


def code_sign_clone_snapshot() -> set[str]:
    root = code_sign_clone_root()
    if not root:
        return set()
    try:
        return {os.path.join(root, e) for e in os.listdir(root) if e.startswith("code_sign_clone.")}
    except OSError:
        return set()


def chrome_main_pids() -> set[int]:
    """PIDs of browser main processes (Chrome/Chromium binaries, no --type=)."""
    found = set()
    for pid in _candidate_pids():
        argv = process_argv(pid)
        if not argv:
            continue
        # The browser binary itself, not helpers such as chrome_crashpad_handler
        # that live under the same bundle path.
        name = os.path.basename(argv[0])
        if name in ("Google Chrome", "Google Chrome Beta", "Google Chrome Canary", "Chromium",
                    "chrome", "google-chrome", "google-chrome-stable", "chromium", "chromium-browser") \
                and not any(a.startswith("--type=") for a in argv):
            found.add(pid)
    return found


def new_code_sign_clone(before: set[str], mains_before: set[int], own_pids,
                        wait: float = 0.0) -> str | None:
    """The clone entry our browser created, or None when that is not certain.

    Chrome creates the clone shortly after startup, sometimes after its CDP
    port is already up, so callers at launch pass a short ``wait``. Certainty
    means: exactly one new entry appeared, and no browser main process other
    than ours started in the same window. ``own_pids`` is the launched pid
    plus every process carrying our exact launch arguments (on macOS the
    launched process can hand off to a child); a callable is re-evaluated at
    check time. Otherwise None: better to leak one directory than to delete a
    clone that is not ours.
    """
    import time

    if code_sign_clone_root() is None:
        return None
    deadline = time.monotonic() + wait
    while True:
        new = code_sign_clone_snapshot() - before
        if len(new) > 1:
            return None
        if len(new) == 1:
            ours = set(own_pids() if callable(own_pids) else own_pids)
            others = chrome_main_pids() - mains_before - ours
            return None if others else new.pop()
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.1)


def remove_unheld_clone_dirs(dirs, gone, timeout: float = 10.0) -> list[str]:
    """Delete attributed clone directories once ``gone()`` says our browser is out.

    ``gone`` is the caller's cheap check that the browser the clone belongs to
    has fully exited (recorded pid dead and no process carrying its exact
    launch arguments). The clone was attributed to that browser at launch, so
    nothing else uses it. No lsof: on a loaded machine one lsof call takes
    ~10 s, which is what made the earlier version miss its deadlines.
    ``timeout`` bounds the wait for ``gone()``.
    """
    import shutil
    import time

    root = code_sign_clone_root()
    pending = {d for d in dirs if d and root and os.path.dirname(d) == root and os.path.isdir(d)}
    if not pending:
        return []
    deadline = time.monotonic() + timeout
    while not gone():
        if time.monotonic() >= deadline:
            return []
        time.sleep(0.2)
    removed = []
    for d in pending:
        shutil.rmtree(d, ignore_errors=True)
        if not os.path.exists(d):
            removed.append(d)
    return removed


def _reap_if_child(pid: int) -> None:
    """Collect ``pid`` if it is a dead child of this process.

    A killed child stays visible to ``kill(pid, 0)`` as a zombie until its
    parent waits for it. Paths that hold no Popen handle (the login-check
    fail-safe, which killed a browser launched in this same process) must
    still reap, or the zombie looks alive for the whole cleanup window.
    """
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass  # not our child, or already reaped
    except OSError:
        pass


def _is_zombie(pid: int) -> bool:
    try:
        stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return stat.startswith("Z")


def browser_gone(pid: int, user_data_dir: str, port: int):
    """A ``gone`` predicate for ``remove_unheld_clone_dirs``."""
    def check() -> bool:
        _reap_if_child(pid)
        main_gone = not process_is_running(pid=pid) or _is_zombie(pid)
        live = [p for p in browser_processes(user_data_dir=user_data_dir, port=port) if not _is_zombie(p)]
        return main_gone and not live
    return check
