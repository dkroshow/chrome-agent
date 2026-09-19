"""Persistent browser profiles.

A default launch uses a throwaway session directory that is deleted when the
browser stops. A *persistent* launch keeps its Chrome profile -- and with it
website sessions and installed extensions -- across stop, process exit, crash
and cleanup. There are two ways to ask for one:

- ``--profile NAME``: a profile chrome-agent manages, stored under the
  per-user profile root. Only ``profiles remove NAME --yes`` ever deletes it.
- ``--profile-dir PATH``: a directory the caller owns. chrome-agent uses it
  and never deletes it, under any command.

Chrome owns everything inside the directory. chrome-agent never reads, copies
or exports authentication or browsing data. The one thing it reads is the
target of Chrome's ``SingletonLock`` (a host name and PID), to tell whether a
browser is using the profile.
"""

import contextlib
import hashlib
import os
import re
import shutil
import sys
from dataclasses import dataclass

# Lowercase only: macOS and Windows filesystems are case-insensitive by
# default, so "Work" and "work" would silently share one directory.
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")

_PROFILE_ROOT_ENV = "CHROME_AGENT_PROFILE_ROOT"
_LOCKS_DIRNAME = ".locks"


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


def _lock_path(profile_path: str) -> str:
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
    lock_file = _lock_path(profile_path)
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
