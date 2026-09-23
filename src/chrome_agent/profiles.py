"""Persistent browser profiles.

A default launch uses a throwaway session directory that is deleted when the
browser stops. A *persistent* launch keeps its Chrome profile -- and with it
website sessions and installed extensions -- across stop, process exit, crash
and cleanup. There are two ways to ask for one:

- ``--profile NAME``: a profile chrome-agent manages, stored under the
  per-user profile root. Only ``profiles remove NAME --yes`` ever deletes it.
- ``--profile-dir PATH``: a directory the caller owns. chrome-agent uses it
  and never deletes it, under any command.

Chrome owns everything inside the directory. chrome-agent never reads or
exports authentication or browsing data, and never copies it except as a
whole-profile copy on the same machine through ``profiles clone``. The one
thing it reads is the target of Chrome's ``SingletonLock`` (a host name and
PID), to tell whether a browser is using the profile.
"""

import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Lowercase only: macOS and Windows filesystems are case-insensitive by
# default, so "Work" and "work" would silently share one directory.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")

_PROFILE_ROOT_ENV = "CHROME_AGENT_PROFILE_ROOT"
_LOCKS_DIRNAME = ".locks"
_ALLOW_NO_GUI_ENV = "CHROME_AGENT_ALLOW_NO_GUI_SESSION"


class ProfileError(Exception):
    """A persistent profile request that cannot be honoured safely."""


@dataclass
class ResolvedProfile:
    """A persistent profile directory, validated and ready to launch on."""
    path: str
    name: str | None = None  # None for a caller-owned --profile-dir


def profile_root() -> str:
    """The per-user directory that holds managed named profiles."""
    override = os.environ.get(_PROFILE_ROOT_ENV)
    if override:
        return os.path.abspath(os.path.expanduser(override))
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")
    return os.path.join(base, "chrome-agent", "profiles")


def validate_name(name: str) -> str:
    """Return ``name`` if it is a legal profile name, else raise ProfileError."""
    if not _NAME_RE.match(name or ""):
        raise ProfileError(
            f"invalid profile name {name!r}: use lowercase letters, digits, "
            "'.', '_' or '-', starting with a letter or digit (max 63 characters)"
        )
    return name


def _owned_by_us(path: str) -> bool:
    if not hasattr(os, "getuid"):
        return True  # Windows: no uid model; ACLs are left to the platform
    return os.lstat(path).st_uid == os.getuid()


def _make_private_dir(path: str) -> None:
    os.makedirs(path, mode=0o700, exist_ok=True)


def _is_within(path: str, parent: str) -> bool:
    path = os.path.realpath(path)
    parent = os.path.realpath(parent)
    return path == parent or path.startswith(parent + os.sep)


def _everyday_chrome_dirs() -> list[str]:
    """Default user-data directories of the user's own everyday browsers.

    Driving these over CDP would expose the user's real browsing identity, and
    current Chrome refuses remote debugging on its default directory anyway.
    """
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        base = os.path.join(home, "Library", "Application Support")
        return [
            os.path.join(base, "Google", "Chrome"),
            os.path.join(base, "Google", "Chrome Beta"),
            os.path.join(base, "Google", "Chrome Canary"),
            os.path.join(base, "Chromium"),
        ]
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return [
            os.path.join(base, "Google", "Chrome", "User Data"),
            os.path.join(base, "Chromium", "User Data"),
        ]
    config = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    return [
        os.path.join(config, "google-chrome"),
        os.path.join(config, "google-chrome-beta"),
        os.path.join(config, "chromium"),
        os.path.join(home, "snap", "chromium", "common", "chromium"),
    ]


def require_gui_session() -> None:
    """On macOS, refuse a persistent launch unless run from the desktop session.

    Chrome encrypts cookies with a key held in the login keychain. A process
    started outside the GUI ("Aqua") session -- over SSH, from some daemons --
    may be unable to use that keychain. Chrome then may not be able to decrypt
    the profile's saved sessions: the browser can come up signed out, and
    there is a risk the saved logins are lost. LaunchAgents, Terminal and
    desktop apps run inside the GUI session.

    Fails closed: anything other than a positive "Aqua" answer refuses,
    including a missing or failing ``launchctl``. The override is explicit.
    """
    if sys.platform != "darwin" or os.environ.get(_ALLOW_NO_GUI_ENV) == "1":
        return
    try:
        result = subprocess.run(
            ["launchctl", "managername"], capture_output=True, text=True, timeout=5,
        )
        manager = result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        manager = ""
    if manager == "Aqua":
        return
    found = f"session type {manager!r}" if manager else "session type could not be determined"
    raise ProfileError(
        f"refusing to open a persistent profile outside the macOS desktop session "
        f"({found}): Chrome may not reach the login keychain there, and the "
        f"profile's saved logins would be at risk. Launch from the desktop session "
        f"(Terminal, a LaunchAgent), or set {_ALLOW_NO_GUI_ENV}=1 to override."
    )


def resolve_named(name: str, create: bool = True) -> ResolvedProfile:
    """Resolve ``--profile NAME`` to its managed directory.

    Creates the directory (owner-only) on first use when ``create`` is true.
    Refuses a symlinked or foreign-owned profile directory: the root is ours,
    so either means something else put it there.
    """
    validate_name(name)
    root = profile_root()
    path = os.path.join(root, name)
    if os.path.islink(path):
        raise ProfileError(f"profile {name!r} is a symlink; refusing to use it")
    if not os.path.exists(path):
        if not create:
            raise ProfileError(f"no such profile: {name}")
        _make_private_dir(root)
        _make_private_dir(path)
    if not os.path.isdir(path):
        raise ProfileError(f"profile {name!r} is not a directory")
    if not _owned_by_us(path):
        raise ProfileError(f"profile {name!r} is not owned by the current user")
    if not _is_within(path, root):
        raise ProfileError(f"profile {name!r} resolves outside the profile root")
    return ResolvedProfile(path=path, name=name)


def resolve_dir(path: str, session_root: str) -> ResolvedProfile:
    """Resolve ``--profile-dir PATH`` to a validated caller-owned directory.

    The directory is created (owner-only) when missing. chrome-agent never
    deletes it. Refused: a symlink, a directory owned by another user, the
    user's everyday Chrome profile, anything inside the temporary session root
    (which cleanup sweeps), and anything inside the managed profile root (use
    ``--profile`` for those).
    """
    if not path:
        raise ProfileError("--profile-dir requires a path")
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.islink(path):
        raise ProfileError(f"--profile-dir {path} is a symlink; pass the real directory")
    for everyday in _everyday_chrome_dirs():
        if _is_within(path, everyday):
            raise ProfileError(
                f"--profile-dir {path} is inside an everyday browser profile "
                f"({everyday}); use a dedicated directory"
            )
    if _is_within(path, session_root):
        raise ProfileError(
            f"--profile-dir {path} is inside the temporary session root "
            f"({session_root}), which cleanup deletes; choose another location"
        )
    if _is_within(path, profile_root()):
        raise ProfileError(
            f"--profile-dir {path} is inside the managed profile root; use --profile NAME"
        )
    if not os.path.exists(path):
        _make_private_dir(path)
    if not os.path.isdir(path):
        raise ProfileError(f"--profile-dir {path} is not a directory")
    if not _owned_by_us(path):
        raise ProfileError(f"--profile-dir {path} is not owned by the current user")
    return ResolvedProfile(path=os.path.realpath(path), name=None)


def list_profiles() -> list[str]:
    """Names of the managed profiles that exist, sorted."""
    root = profile_root()
    if not os.path.isdir(root):
        return []
    return sorted(
        entry for entry in os.listdir(root)
        if _NAME_RE.match(entry)
        and os.path.isdir(os.path.join(root, entry))
        and not os.path.islink(os.path.join(root, entry))
    )


def lock_path(profile_path: str) -> str:
    # Keyed on the resolved path so a named profile and a --profile-dir that
    # reach the same directory contend for one lock. Kept outside the profile:
    # chrome-agent writes nothing into a directory Chrome owns.
    digest = hashlib.sha256(os.path.realpath(profile_path).encode()).hexdigest()[:32]
    return os.path.join(profile_root(), _LOCKS_DIRNAME, f"{digest}.lock")


@contextlib.contextmanager
def launch_lock(profile_path: str):
    """Serialize launches and removal of one profile directory.

    Held from the in-use check until the new instance is registered, so two
    concurrent launches of one profile cannot both pass the check. Advisory
    ``flock``; on platforms without ``fcntl`` (Windows) this is a no-op and
    Chrome's own profile singleton is the only guard.
    """
    try:
        import fcntl
    except ImportError:
        yield
        return
    lock_file = lock_path(profile_path)
    _make_private_dir(os.path.dirname(lock_file))
    fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def singleton_holder_pid(profile_path: str) -> int | None:
    """PID recorded in the directory's Chrome ``SingletonLock``, if any.

    Chrome writes the lock as a symlink whose target is ``<hostname>-<pid>``.
    Read-only: chrome-agent never creates, edits or unlinks Chrome's locks.
    """
    lock_file = os.path.join(profile_path, "SingletonLock")
    try:
        target = os.readlink(lock_file)
    except OSError:
        return None
    try:
        return int(target.rsplit("-", 1)[-1])
    except ValueError:
        return None


# Chrome's per-instance lock and IPC files. A copy must not carry them: a
# stale SingletonLock would make the clone believe another Chrome holds it.
_NEVER_COPY = {"SingletonLock", "SingletonSocket", "SingletonCookie", "lockfile", "RunningChromeVersion"}


def clone_named(source: str, new: str) -> str:
    """Copy managed profile ``source`` to a new managed profile ``new``.

    A profile is the unit of sign-in: a fresh one starts signed out, and only a
    person can change that. Cloning a profile that already holds a Chrome
    sign-in gives a new, separately usable profile that starts from the same
    saved state -- "save as" for profiles. Everything Chrome saved is copied
    (sessions, extensions, settings), which is the one place chrome-agent
    copies profile data; it still reads none of it. The copy is on this
    machine, so Chrome's credential store can open it; a site may still ask
    for a fresh login when it treats the clone as a new device.

    Refused while a browser holds the source; the caller must hold
    ``launch_lock`` for both directories. Returns the new profile's path.
    """
    src = resolve_named(source, create=False)
    validate_name(new)
    dest_path = os.path.join(profile_root(), new)
    if os.path.lexists(dest_path):
        raise ProfileError(f"profile {new!r} already exists")
    holder = singleton_holder_pid(src.path)
    if holder is not None and _pid_running(holder):
        raise ProfileError(f"profile {source!r} is in use by a browser (pid {holder}); stop it first")
    for dirpath, dirnames, filenames in os.walk(src.path):
        for entry in dirnames + filenames:
            if os.path.islink(os.path.join(dirpath, entry)) and entry not in _NEVER_COPY:
                # Chrome writes no symlinks besides its lock files. One here
                # means something outside Chrome touched the profile; copying
                # its target could pull in a directory outside the profile.
                raise ProfileError(
                    f"profile {source!r} contains an unexpected symlink "
                    f"({os.path.relpath(os.path.join(dirpath, entry), src.path)}); refusing to clone"
                )

    def ignore(directory: str, names: list[str]) -> set[str]:
        return {n for n in names if n in _NEVER_COPY}

    # A private, unique staging directory: names beginning with '.' are not
    # valid profile names, so this can never collide with or destroy a profile.
    staging = tempfile.mkdtemp(prefix=f".clone-{new}-", dir=profile_root())
    tmp_path = os.path.join(staging, new)
    try:
        shutil.copytree(src.path, tmp_path, symlinks=True, ignore=ignore, copy_function=shutil.copy2)
        os.chmod(tmp_path, 0o700)
        os.rename(tmp_path, dest_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return dest_path


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_named(name: str) -> str:
    """Delete a managed named profile. The only deletion path for profiles.

    The caller must already hold ``launch_lock`` for the profile and have
    established that no browser is using it. Returns the removed path.

    Refused on Windows: there the launch lock is a no-op and Chrome's profile
    lock is not the readable symlink it is on POSIX, so "no browser is using
    it" cannot be established. Deleting under a running browser would remove
    part of a profile before a locked file stopped it.
    """
    if sys.platform == "win32":
        raise ProfileError(
            "profiles remove is not supported on Windows: chrome-agent cannot "
            "verify the profile is idle. Close the browser and delete "
            f"{os.path.join(profile_root(), name)} yourself."
        )
    resolved = resolve_named(name, create=False)
    shutil.rmtree(resolved.path)
    return resolved.path
