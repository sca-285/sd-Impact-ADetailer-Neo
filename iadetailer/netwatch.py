"""Name whatever is opening a network connection inside the WebUI process.

An antivirus that inspects TLS (Kaspersky, ESET, Bitdefender) pops a warning
when something opens an HTTPS connection it cannot verify. The warning names
the host but never the code, and nothing in this extension makes an outbound
request, so the caller is somewhere in the libraries it pulls in.

This hooks name resolution, which every normal HTTP client goes through right
before it connects, and prints the Python stack that asked for it. It only
watches and reports: the lookup itself is passed through untouched.
"""

from __future__ import annotations

import os
import socket
import threading
import traceback

# Hosts the model stack is known to reach for. Anything here gets reported
# even in "watched" mode.
WATCHED = (
    "huggingface.co",
    "hf.co",
    "hf-mirror.com",
    "cdn-lfs",
    "github.com",
    "githubusercontent.com",
    "ultralytics.com",
    "pypi.org",
    "pythonhosted.org",
    "google-analytics.com",
    "sentry.io",
)

_LOCK = threading.Lock()
_ORIGINAL = None
_REPORTED: set = set()
_MODE = "off"


_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _is_ours(filename: str) -> bool:
    """True only for files inside this extension's package. Matching on a name
    like "netwatch.py" would also hide frames from anyone else's file that
    happens to end the same way, which is exactly the caller we are hunting."""
    try:
        path = os.path.abspath(filename)
    except Exception:
        return False
    return path == _PACKAGE_DIR or path.startswith(_PACKAGE_DIR + os.sep)


def _report(host: str, port) -> None:
    frames = [f for f in traceback.extract_stack()[:-2] if not _is_ours(f.filename)]
    if not frames:
        return
    tail = frames[-14:]
    # One report per host per call site, otherwise a retry loop floods the log.
    key = (host, tuple((f.filename, f.lineno) for f in tail[-4:]))
    with _LOCK:
        if key in _REPORTED:
            return
        _REPORTED.add(key)

    where = f"{host}:{port}" if port else host
    print("")
    print(f"[Impact ADetailer] network: something is looking up {where}")
    for frame in tail:
        print(f"    {frame.filename}:{frame.lineno} in {frame.name}")
        if frame.line:
            print(f"        {frame.line.strip()}")
    print("[Impact ADetailer] ^ the last few lines are the code your antivirus")
    print("[Impact ADetailer]   is warning about. Nothing above is this extension.")
    print("")


def enable(mode: str = "watched") -> None:
    """mode: 'watched' for the model hosts above, 'all' for every lookup."""
    global _ORIGINAL, _MODE
    _MODE = mode
    if _ORIGINAL is not None:
        return
    # A reload can drop this module while its hook is still installed. Unwrap
    # any previous one instead of stacking a second layer on top of it.
    current = socket.getaddrinfo
    _ORIGINAL = getattr(current, "__iad_original__", current)

    def traced(host, port, *args, **kwargs):
        try:
            name = str(host or "")
            if _MODE == "all" or any(w in name for w in WATCHED):
                _report(name, port)
        except Exception:
            pass
        return _ORIGINAL(host, port, *args, **kwargs)

    traced.__iad_netwatch__ = True
    traced.__iad_original__ = _ORIGINAL
    socket.getaddrinfo = traced
    print(
        "[Impact ADetailer] network trace is on. The console will name whatever "
        "opens a connection."
    )


def disable() -> None:
    global _ORIGINAL, _MODE
    _MODE = "off"
    current = socket.getaddrinfo
    if getattr(current, "__iad_netwatch__", False):
        socket.getaddrinfo = getattr(current, "__iad_original__", _ORIGINAL or current)
    _ORIGINAL = None


def sync(mode: str) -> None:
    """Turn the trace on or off to match the setting, without a restart."""
    if mode and mode != "Off":
        enable("all" if str(mode).lower().startswith("every") else "watched")
    else:
        disable()
