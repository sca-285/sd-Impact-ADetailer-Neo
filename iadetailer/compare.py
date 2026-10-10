from __future__ import annotations

import atexit
import shutil
import threading
import time
from pathlib import Path

from PIL import Image

from .paths import session_temp_root

_SESSION = time.strftime("%Y%m%d-%H%M%S")
_LOCK = threading.Lock()
_RUN = 0
_INDEX = 0
_STORE: list[dict] = []
_REV = 0
_DIR: Path | None = None

ORIGINAL = "original"
DETAILED = "detailed"

# The comparer is never wider than ~1100 css px, so there is nothing to gain
# from keeping full-resolution PNGs around for it. Encoding the preview once at
# save time also means a browser request is a plain file read.
PREVIEW_MAX = 2048
PREVIEW_QUALITY = 95


def _dir() -> Path:
    global _DIR
    if _DIR is None:
        _DIR = session_temp_root() / _SESSION
        _DIR.mkdir(parents=True, exist_ok=True)
    return _DIR


def cleanup() -> None:
    global _DIR, _STORE, _INDEX, _RUN
    root = session_temp_root()
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    with _LOCK:
        _DIR = None
        _STORE = []
        _INDEX = 0
        _RUN = 0


atexit.register(cleanup)


def start_run() -> int:
    global _RUN, _STORE, _INDEX, _REV
    with _LOCK:
        _RUN += 1
        _REV += 1
        _STORE = []
        _INDEX = 0
        return _RUN


def _write_preview(image: Image.Image, path: Path) -> None:
    w, h = image.size
    scale = min(1.0, float(PREVIEW_MAX) / max(w, h))
    if scale < 1.0:
        image = image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
        )
    image.save(path, format="JPEG", quality=PREVIEW_QUALITY, optimize=False)


def save_pair(original: Image.Image, detailed: Image.Image) -> dict:
    global _INDEX, _REV
    with _LOCK:
        _INDEX += 1
        _REV += 1
        stem = f"{_RUN:03d}-{_INDEX:04d}"
        run = _RUN
        rev = _REV
    folder = _dir()
    original_p = folder / f"{stem}_{ORIGINAL}.jpg"
    detailed_p = folder / f"{stem}_{DETAILED}.jpg"
    original = original.convert("RGB")
    detailed = detailed.convert("RGB")
    _write_preview(original, original_p)
    _write_preview(detailed, detailed_p)
    rec = {
        "run": run,
        "rev": rev,
        "index": _INDEX,
        "slot": 0,
        ORIGINAL: str(original_p),
        DETAILED: str(detailed_p),
        "w": detailed.width,
        "h": detailed.height,
    }
    with _LOCK:
        rec["slot"] = len(_STORE)
        _STORE.append(rec)
    try:
        from .settings import output_dir, save_before_images

        dest = output_dir()
        if dest and save_before_images():
            out = Path(dest)
            out.mkdir(parents=True, exist_ok=True)
            original.save(out / f"{stem}_{ORIGINAL}.png")
            detailed.save(out / f"{stem}_{DETAILED}.png")
    except Exception:
        pass
    return rec


def pairs() -> list[dict]:
    with _LOCK:
        return list(_STORE)


def last_pair() -> dict | None:
    with _LOCK:
        return _STORE[-1] if _STORE else None


def current_run() -> int:
    with _LOCK:
        return _RUN


def state() -> dict:
    """Small JSON payload the browser polls. No image data in here."""
    from .settings import show_comparer
    from .version import build_id

    items = pairs()
    return {
        "ok": bool(items),
        "build": build_id(),
        "run": current_run(),
        "rev": _REV,
        "show": bool(show_comparer()),
        "count": len(items),
        "pairs": [
            {"slot": r["slot"], "index": r["index"], "w": r["w"], "h": r["h"]}
            for r in items
        ],
    }


def _record(slot: int) -> dict | None:
    items = pairs()
    if not items:
        return None
    if slot < 0 or slot >= len(items):
        return items[-1]
    return items[slot]


def image_bytes(slot: int, side: str) -> bytes | None:
    """JPEG preview of one side of one pair. side is 'original' or 'detailed'.
    Already encoded at save time, so this is a file read."""
    rec = _record(slot)
    if not rec:
        return None
    path = rec.get(ORIGINAL if side == ORIGINAL else DETAILED)
    if not path:
        return None
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


class CompareStore:
    start_run = staticmethod(start_run)
    save_pair = staticmethod(save_pair)
    pairs = staticmethod(pairs)
    last_pair = staticmethod(last_pair)
    current_run = staticmethod(current_run)
    state = staticmethod(state)
    image_bytes = staticmethod(image_bytes)
    cleanup = staticmethod(cleanup)
    session_dir = staticmethod(_dir)
