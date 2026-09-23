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
        # Chrome rewrites its argv into one space-joined string on Linux; in
        # that case split on whitespace (upstream accepts that ambiguity).
        parts = [p.decode(errors="replace") for p in raw.split(b"\0") if p]
        return parts[0].split() if len(parts) == 1 and " " in parts[0] else parts
    if sys.platform == "darwin":
        return _darwin_procargs(pid)
    return None


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
