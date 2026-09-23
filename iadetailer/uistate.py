"""Remember what the tabs were set to.

The WebUI's own ui-config.json cannot do this job here, for two reasons:

* It keys every entry on the component's LABEL, not its elem_id. Each tab of
  this extension uses the same labels ("ADetailer prompt", "ADetailer
  detector", ...), so all four tabs collapse onto one key and overwrite each
  other. Adding elem_ids does not change that - ui_loadsave never looks at them.
* It only ever writes DEFAULTS, plus whatever you edit by hand in
  Settings -> Defaults. Nothing there records the value you actually typed.

So this keeps its own small file. It lives beside the WebUI's own config rather
than inside the extension folder, so replacing the extension with a new build
does not wipe your settings.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import fields
from pathlib import Path

from .args import RunArgs, UnitArgs
from .paths import _webui_root

FILENAME = "impact-adetailer-ui.json"
_LOCK = threading.Lock()
_PATH: Path | None = None

_COERCE = {
    "bool": bool,
    "int": lambda v: int(float(v)),
    "float": float,
    "str": str,
}


def _field_types() -> dict[str, str]:
    # `from __future__ import annotations` leaves these as strings, which is
    # exactly the form the coercion table wants.
    return {f.name: str(f.type) for f in fields(UnitArgs)}


FIELD_TYPES = _field_types()


def path() -> Path:
    """Beside the WebUI's config when that is writable, else inside the
    extension. Resolved once, because a read-only probe is not free."""
    global _PATH
    if _PATH is not None:
        return _PATH
    candidates = [_webui_root() / FILENAME, Path(__file__).resolve().parents[1] / FILENAME]
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            if candidate.exists():
                _PATH = candidate
                return _PATH
            # Prove it is writable before committing to it.
            with open(candidate, "a", encoding="utf-8"):
                pass
            candidate.unlink(missing_ok=True)
            _PATH = candidate
            return _PATH
        except OSError:
            continue
    _PATH = candidates[-1]
    return _PATH


def _read_all() -> dict:
    try:
        with open(path(), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load(tab: str) -> dict:
    """{'enable': bool, 'save_before': bool, 'preview': str,
        'units': [ {field: value}, ... ]} - never raises.

    A file written by a build before the head controls existed simply has no
    'save_before'/'preview' keys, and the defaults below fill them in."""
    section = _read_all().get(tab)
    if not isinstance(section, dict):
        return {}
    units = section.get("units")
    return {
        "enable": bool(section.get("enable", False)),
        "save_before": bool(section.get("save_before", False)),
        "preview": str(section.get("preview", "Off") or "Off"),
        "units": [u if isinstance(u, dict) else {} for u in units] if isinstance(units, list) else [],
    }


def save(tab: str, run: RunArgs, units: list[UnitArgs]) -> None:
    payload = {
        "enable": bool(run.enable),
        "save_before": bool(run.save_before),
        "preview": str(run.preview or "Off"),
        "units": [
            {name: getattr(unit, name) for name in FIELD_TYPES if hasattr(unit, name)}
            for unit in units
        ],
    }
    with _LOCK:
        data = _read_all()
        if data.get(tab) == payload:
            return                      # nothing changed, skip the disk write
        data[tab] = payload
        _write(data)


def reset(tab: str) -> None:
    with _LOCK:
        data = _read_all()
        if data.pop(tab, None) is not None:
            _write(data)


def _write(data: dict) -> None:
    target = path()
    temp = target.with_suffix(".tmp")
    try:
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
        os.replace(temp, target)        # atomic, so a crash mid-write cannot
    except OSError as exc:              # leave a half-file behind
        print(f"[Impact ADetailer] could not save the tab settings: {exc}")
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def pick(saved_unit: dict, field: str, default, choices: list | None = None):
    """The remembered value for one field, or the default.

    A remembered value is refused when it no longer makes sense: the wrong
    type, or a detector/sampler that is not on the machine any more."""
    if not saved_unit or field not in saved_unit:
        return default
    raw = saved_unit[field]
    coerce = _COERCE.get(FIELD_TYPES.get(field, ""))
    if coerce is None:
        return default
    try:
        value = coerce(raw)
    except (TypeError, ValueError):
        return default
    if choices is not None and value not in choices:
        # The .pt was deleted, or the sampler list changed between versions.
        return default
    return value


def pick_choice(raw, default: str, choices: list[str]) -> str:
    """Same idea as pick(), for the head controls - they are not UnitArgs
    fields, so they have no entry in FIELD_TYPES to coerce through."""
    value = str(raw or "")
    return value if value in choices else default
