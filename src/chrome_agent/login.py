"""Login checks: is this browser signed in to a site, judged by a probe.

A *probe* is a JavaScript expression supplied by the caller. It runs in a page
of the site and answers the only question that matters -- "does the site treat
this browser as signed in?" -- typically with a small authenticated read
(``fetch`` inherits the page's session). chrome-agent knows nothing about any
site; the probe carries all of that.

The probe evaluates (promises are awaited) to one of:

- ``true`` / ``"ok"``                      -> signed in
- ``false`` / ``"needs_login"``            -> a person must sign in
- ``{"status": "ok" | "needs_login" | "error", ...}``  (extra keys are kept)

Anything else, an exception, or a timeout is an error. Results map to exit
codes 0 (ok), 2 (needs_login) and 3 (error), so callers can branch without
parsing output.

Nothing here reads, stores or logs credentials or cookies. A one-time code a
person relays is typed with ordinary CDP input commands, outside this module.
"""

import asyncio
import json
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .cdp_client import CDPClient, get_ws_url
from .errors import CDPError

OK = "ok"
NEEDS_LOGIN = "needs_login"
ERROR = "error"
EXIT_CODES = {OK: 0, NEEDS_LOGIN: 2, ERROR: 3}


@dataclass
class LoginResult:
    status: str
    url: str = ""          # where the page ended up (login pages redirect)
    detail: dict = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]

    def to_json(self) -> str:
        return json.dumps({"status": self.status, "url": self.url, **self.detail})


def classify(value) -> tuple[str, dict]:
    """Map a probe's return value to (status, detail)."""
    if value is True or value == OK:
        return OK, {}
    if value is False or value == NEEDS_LOGIN:
        return NEEDS_LOGIN, {}
    if isinstance(value, dict) and value.get("status") in EXIT_CODES:
        detail = {k: v for k, v in value.items() if k != "status"}
        return value["status"], detail
    return ERROR, {"reason": "unrecognized probe result", "value": repr(value)[:200]}


def same_site(url_a: str, url_b: str) -> bool:
    """Whether two URLs share a host -- i.e. the login redirects have ended."""
    return urlsplit(url_a).hostname == urlsplit(url_b).hostname


class _Tab:
    """One tab this module opened, driven over the browser-level connection."""

    def __init__(self, cdp: CDPClient, target_id: str, session_id: str):
        self.cdp, self.target_id, self.session_id = cdp, target_id, session_id

    @classmethod
    async def open(cls, cdp: CDPClient, url: str, background: bool) -> "_Tab":
        # background=True never activates the tab or raises the window: a
        # check must not steal focus from whatever the machine is doing.
        created = await cdp.send(
            method="Target.createTarget",
            params={"url": url, "background": background},
        )
        attached = await cdp.send(
            method="Target.attachToTarget",
            params={"targetId": created["targetId"], "flatten": True},
        )
        return cls(cdp, created["targetId"], attached["sessionId"])

    async def evaluate(self, expression: str, timeout: float):
        result = await asyncio.wait_for(
            self.cdp.send(
                method="Runtime.evaluate",
                params={"expression": expression, "awaitPromise": True, "returnByValue": True},
                session_id=self.session_id,
            ),
            timeout=timeout,
        )
        if "exceptionDetails" in result:
            text = result["exceptionDetails"].get("exception", {}).get("description") \
                or result["exceptionDetails"].get("text", "probe raised")
            raise RuntimeError(text.splitlines()[0][:300])
        return result.get("result", {}).get("value")

    async def wait_ready(self, timeout: float) -> str:
        """Wait until the document has loaded; return the URL it settled on."""
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            try:
                state = await self.evaluate(
                    "({ready: document.readyState, url: location.href})", timeout=5,
                )
            except (CDPError, RuntimeError, asyncio.TimeoutError, TimeoutError):
                # A navigation or redirect in flight destroys the execution
                # context under the evaluation. That is the page still
                # loading, not a failure: ask again until the deadline.
                state = None
            if state and state["ready"] == "complete" and state["url"] != "about:blank":
                return state["url"]
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("page did not finish loading")
            await asyncio.sleep(0.25)

    async def close(self) -> None:
        try:
            await self.cdp.send(method="Target.closeTarget", params={"targetId": self.target_id})
        except Exception:
            pass


async def _probe_tab(tab: _Tab, url: str, probe: str, timeout: float) -> LoginResult:
    try:
        landed = await tab.wait_ready(timeout=timeout)
        # Off the site's host means a sign-in redirect took over. The probe is
        # written for the site's own pages, so do not run it somewhere else.
        if not same_site(landed, url):
            return LoginResult(NEEDS_LOGIN, url=landed, detail={"reason": "redirected off site"})
        status, detail = classify(await tab.evaluate(probe, timeout=timeout))
        return LoginResult(status, url=landed, detail=detail)
    except (TimeoutError, asyncio.TimeoutError):
        return LoginResult(ERROR, url=url, detail={"reason": "timed out"})
    except Exception as exc:
        return LoginResult(ERROR, url=url, detail={"reason": str(exc)[:300]})


async def login_check(port: int, url: str, probe: str, timeout: float = 30.0) -> LoginResult:
    """Check sign-in state in a fresh background tab, then close that tab.

    Never touches a tab it did not open.
    """
    try:
        ws_url = get_ws_url(port=port, target_type="browser")
    except ConnectionError as exc:
        return LoginResult(ERROR, url=url, detail={"reason": str(exc)})
    async with CDPClient(ws_url=ws_url) as cdp:
        tab = await _Tab.open(cdp, url=url, background=True)
        try:
            return await _probe_tab(tab, url=url, probe=probe, timeout=timeout)
        finally:
            await tab.close()


async def wait_for_login(
    port: int, url: str, probe: str, timeout: float = 600.0,
    poll_interval: float = 3.0, probe_timeout: float = 20.0,
) -> LoginResult:
    """Open the site in a visible tab and wait for a person to sign in.

    Polls only that tab, and only while it is on the site's own host: while
    the person is on an identity provider's pages nothing is evaluated or
    navigated, so the sign-in ceremony is never disturbed. The tab is left
    open on success (the person is looking at it) and on timeout.
    """
    try:
        ws_url = get_ws_url(port=port, target_type="browser")
    except ConnectionError as exc:
        return LoginResult(ERROR, url=url, detail={"reason": str(exc)})
    async with CDPClient(ws_url=ws_url) as cdp:
        tab = await _Tab.open(cdp, url=url, background=False)
        deadline = asyncio.get_event_loop().time() + timeout
        last = LoginResult(NEEDS_LOGIN, url=url)
        while asyncio.get_event_loop().time() < deadline:
            last = await _probe_tab(tab, url=url, probe=probe, timeout=probe_timeout)
            if last.status == OK:
                return last
            await asyncio.sleep(poll_interval)
        if last.status == ERROR:
            return last
        return LoginResult(NEEDS_LOGIN, url=last.url, detail={"reason": "timed out waiting for sign-in"})
