"""Let ControlNet run inside the detail pass.

Every detection is sampled by a nested img2img job, and that job is handed a
script runner with nothing in it - which is what keeps Hires-fix, upscalers and
every other alwayson script from firing once per face. The cost of that clean
room is that ControlNet cannot see the job either, so a tile or inpaint model
that would have held the region's structure never gets a say.

This module puts exactly one script back: ControlNet, and only when a tab asks
for it. Three ways to go:

  None         the empty runner, unchanged. The default, and free.
  Passthrough  the parent generation's own ControlNet units, forwarded as they
               are - the pose or depth map that shaped the picture keeps
               shaping the repaint.
  <a model>    a fresh unit built for this crop, with the crop itself as the
               control image.

Forge, ReForge and Forge Neo all ship ControlNet as a builtin whose package is
`lib_controlnet`, and all three expose the same ControlNetUnit dataclass and the
same global_state helpers (checked against the three trees, not assumed). A1111
with sd-webui-controlnet is a different API and is not covered; there the panel
stays on "None" and says why.
"""

from __future__ import annotations

from copy import copy
from typing import Any

NONE = "None"
PASSTHROUGH = "Passthrough"

# Which ControlNet families are worth offering for a face- or hand-sized crop.
SUPPORTED_TYPES = ("Inpaint", "Tile", "Depth", "Lineart", "OpenPose", "Scribble", "Canny")

_SCRIPT_TITLE = "ControlNet"


def _lib():
    try:
        from lib_controlnet import external_code, global_state
        from lib_controlnet.external_code import ControlNetUnit
    except Exception:
        return None
    return external_code, global_state, ControlNetUnit


def available() -> bool:
    return _lib() is not None


def list_models() -> list[str]:
    lib = _lib()
    if lib is None:
        return [NONE]
    _, global_state, _ = lib
    names: set[str] = set()
    for tag in SUPPORTED_TYPES:
        try:
            names.update(global_state.get_filtered_controlnet_names(tag))
        except Exception:
            continue
    names.discard(NONE)
    return [NONE, PASSTHROUGH, *sorted(names)]


def list_modules() -> list[str]:
    lib = _lib()
    if lib is None:
        return [NONE]
    _, global_state, _ = lib
    names: set[str] = set()
    for tag in SUPPORTED_TYPES:
        try:
            # get_filtered_preprocessors returns a dict on Neo and a list on the
            # others; update() takes the keys either way.
            names.update(global_state.get_filtered_preprocessors(tag))
        except Exception:
            continue
    names.discard(NONE)
    return [NONE, *sorted(names)]


def _find_script(runner, title: str):
    for script in getattr(runner, "scripts", []) or []:
        try:
            if script.title() == title:
                return script
        except Exception:
            continue
    return None


def _args_copy(args):
    """A shallow copy of every script arg.

    ControlNet mutates its units while it runs (it caches the preprocessed map
    on them). Handing it the parent's live objects would let a detail pass write
    back into the outer generation's state."""
    out = []
    for arg in list(args or []):
        try:
            out.append(copy(arg))
        except TypeError:
            out.append(arg)
    return out


def _only_controlnet(runner):
    """The same runner with every alwayson script but ControlNet removed."""
    trimmed = copy(runner)
    kept = []
    for script in getattr(runner, "alwayson_scripts", []) or []:
        title = ""
        try:
            title = script.title() or ""
        except Exception:
            pass
        if title == _SCRIPT_TITLE:
            kept.append(script)
    trimmed.alwayson_scripts = kept
    return trimmed, bool(kept)


def attach(job: Any, parent: Any, model: str, module: str, weight: float,
           start: float, end: float) -> bool:
    """Give `job` a script runner that will run ControlNet. True if it took."""
    model = str(model or NONE)
    if model == NONE:
        return False
    lib = _lib()
    if lib is None:
        return False
    external_code, _, ControlNetUnit = lib

    parent_runner = getattr(parent, "scripts", None)
    if parent_runner is None:
        return False

    if model == PASSTHROUGH:
        runner, found = _only_controlnet(parent_runner)
        if not found:
            return False
        # The scripts keep their original args_from/args_to, so the copied arg
        # list has to stay the same length and order for those to still point
        # at the right things.
        job.scripts_value = runner
        job.script_args_value = _args_copy(getattr(parent, "script_args", None))
        return True

    script = _find_script(parent_runner, _SCRIPT_TITLE)
    if script is None:
        return False

    import numpy as np

    image = np.asarray(job.init_images[0].convert("RGB"))
    unit_kwargs = {
        "enabled": True,
        # The whole crop is fair game: the detail pass already carries its own
        # mask, and a second one here would fight it.
        "image": {"image": image, "mask": np.full_like(image, 255)},
        "model": model,
        "module": str(module or NONE),
        "weight": float(weight),
        "guidance_start": float(start),
        "guidance_end": float(end),
    }
    try:
        unit_kwargs["processor_res"] = external_code.pixel_perfect_resolution(
            image,
            target_H=job.height,
            target_W=job.width,
            resize_mode=external_code.resize_mode_from_value(job.resize_mode),
        )
    except Exception:
        pass

    # Only pass what this fork's dataclass actually declares.
    try:
        import dataclasses

        allowed = {f.name for f in dataclasses.fields(ControlNetUnit) if f.init}
        unit_kwargs = {k: v for k, v in unit_kwargs.items() if k in allowed}
    except Exception:
        pass

    runner = copy(parent_runner)
    own = copy(script)
    own.args_from = 0
    own.args_to = 1
    runner.alwayson_scripts = [own]
    job.scripts_value = runner
    job.script_args_value = [ControlNetUnit(**unit_kwargs)]
    return True


def describe(model: str, module: str) -> str:
    if not model or model == NONE:
        return ""
    if model == PASSTHROUGH:
        return "ControlNet: passthrough"
    return f"ControlNet: {model}" + (f" / {module}" if module and module != NONE else "")
