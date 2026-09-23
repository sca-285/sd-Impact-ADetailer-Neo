"""Stop the detector stack from phoning home.

Only the variables below actually do something. Earlier versions also set
ULTRALYTICS_OFFLINE and YOLO_OFFLINE, which ultralytics does not read, so they
looked like protection while changing nothing.
"""

from __future__ import annotations

import os
import sys

_KEYS = {
    # ultralytics chatter
    "YOLO_VERBOSE": "False",
    # albumentations, pulled in by ultralytics, pings pypi on import unless told not to
    "NO_ALBUMENTATIONS_UPDATE": "1",
    # honoured by a number of libraries in this dependency tree
    "DO_NOT_TRACK": "1",
}

# Set only when the user opts in: these are process-wide and would stop any
# OTHER extension from downloading a model too.
_HF_KEYS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
}


def apply() -> None:
    for key, value in _KEYS.items():
        os.environ.setdefault(key, value)
    _quiet_ultralytics_settings()


def _quiet_ultralytics_settings() -> None:
    # Only if ultralytics is already loaded. Importing it here would drag a
    # heavy dependency into WebUI startup for no reason.
    if "ultralytics.utils" not in sys.modules:
        return
    try:
        from ultralytics.utils import SETTINGS

        if hasattr(SETTINGS, "update"):
            SETTINGS.update({"sync": False})
        else:
            SETTINGS["sync"] = False
    except Exception:
        pass


def harden_ultralytics() -> None:
    """Called right after ultralytics is imported, when the module objects
    actually exist to patch."""
    _quiet_ultralytics_settings()

    # ultralytics decides at import time whether it is online and gates its
    # auto-update, its telemetry and (on older versions) its downloads on that
    # one flag. Setting ultralytics.utils.ONLINE alone is not enough: every
    # consumer does `from ultralytics.utils import ONLINE`, which copies the
    # value into its own module namespace at import time. checks.py already
    # holds True by the time we get here, so the copy is what has to be
    # overwritten - in whichever modules this version happens to have made one.
    patched = 0
    try:
        for name, module in list(sys.modules.items()):
            if not name.startswith("ultralytics") or module is None:
                continue
            try:
                if getattr(module, "ONLINE", False) is True:
                    module.ONLINE = False
                    patched += 1
            except Exception:
                continue
    except Exception:
        pass
    if patched:
        # Imported here, not at module scope: offline.apply() runs at import
        # time from detect.py, and settings reaches into modules.shared.
        from .settings import log

        log(f"ultralytics online flag cleared in {patched} module(s)")


def force_huggingface_offline() -> bool:
    """Opt-in. Returns True if it took effect.

    The env vars only bite if they are set before huggingface_hub is imported,
    and by the time an extension loads the WebUI has usually imported it
    already, so the live constants are patched too."""
    for key, value in _HF_KEYS.items():
        os.environ[key] = value
    patched = False
    for name, attr in (
        ("huggingface_hub.constants", "HF_HUB_OFFLINE"),
        ("transformers.utils.hub", "_is_offline_mode"),
    ):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, attr):
            try:
                setattr(module, attr, True)
                patched = True
            except Exception:
                pass
    return patched


apply()
