"""Check, at load time, that this WebUI build actually provides what the
extension calls.

Forge, ReForge and Forge Neo expose the same 37 symbols this extension touches,
but not with identical shapes: PostprocessImageArgs takes (image, index) on
Forge and Neo and just (image) on ReForge, for one. Rather than sniffing which
fork is running - a test that breaks the day a new fork appears - every
assumption is probed directly and reported. A failure then names the missing
symbol on the terminal at startup, instead of surfacing as a traceback in the
middle of a generation.

Nothing here raises. A broken probe is reported as unknown and the extension
carries on.
"""

from __future__ import annotations

import importlib
import inspect

# module -> the names this extension actually uses from it
REQUIRED: dict[str, tuple[str, ...]] = {
    "modules.processing": (
        "StableDiffusionProcessing",
        "StableDiffusionProcessingImg2Img",
        "process_images",
    ),
    "modules.scripts": ("Script", "AlwaysVisible", "PostprocessImageArgs"),
    "modules.script_callbacks": ("on_ui_settings", "on_app_started"),
    "modules.shared": ("opts", "cmd_opts", "OptionInfo"),
    "modules.sd_samplers": ("visible_sampler_names",),
    "modules.sd_schedulers": ("schedulers",),
    "modules.devices": ("torch_gc",),
    "modules.extra_networks": ("activate",),
}

# What _copy_processing hands to StableDiffusionProcessingImg2Img.
JOB_KWARGS = (
    "sd_model", "outpath_samples", "outpath_grids", "prompt", "negative_prompt",
    "seed", "sampler_name", "scheduler", "batch_size", "n_iter", "steps",
    "cfg_scale", "width", "height", "init_images", "denoising_strength",
    "resize_mode",
)


def _get(path: str):
    module_name, _, attr = path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, attr, None)


def _accepted_kwargs(cls) -> set[str] | None:
    """Field names the class will accept, whether it is a dataclass or not."""
    try:
        import dataclasses

        if dataclasses.is_dataclass(cls):
            return {f.name for f in dataclasses.fields(cls) if f.init}
    except Exception:
        pass
    try:
        return set(inspect.signature(cls.__init__).parameters) - {"self"}
    except Exception:
        return None


def missing() -> list[str]:
    """Required symbols this build does not have. Empty means good to go."""
    gaps: list[str] = []
    for module_name, names in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            gaps.append(module_name)
            continue
        gaps.extend(f"{module_name}.{n}" for n in names if not hasattr(module, n))
    return gaps


def optional() -> dict[str, bool | None]:
    """Things the extension adapts to rather than requires.
    True = present, False = absent, None = could not tell."""
    found: dict[str, bool | None] = {}

    pp = _get("modules.scripts.PostprocessImageArgs")
    try:
        found["pp.index"] = (
            pp is not None and "index" in _accepted_kwargs(pp)  # type: ignore[operator]
        )
    except Exception:
        found["pp.index"] = None

    found["sampler autocorrect"] = bool(
        _get("modules.sd_samplers.fix_p_invalid_sampler_and_scheduler")
    )
    found["inner sampler entry"] = bool(_get("modules.processing.process_images_inner"))

    img2img = _get("modules.processing.StableDiffusionProcessingImg2Img")
    accepted = _accepted_kwargs(img2img) if img2img is not None else None
    if accepted is None:
        found["job kwargs"] = None
        found["soft noise mask"] = None
    else:
        found["job kwargs"] = set(JOB_KWARGS).issubset(accepted)
        found["soft noise mask"] = "mask_round" in accepted or hasattr(img2img, "mask_round")

    # The Forge-family builtin ControlNet. Absent on plain A1111 (and on a
    # build where it is disabled), which is the one optional feature that then
    # cannot work at all rather than degrading.
    try:
        importlib.import_module("lib_controlnet")
        found["controlnet"] = True
    except Exception:
        found["controlnet"] = False

    # Whether ui_loadsave still honours do_not_save_to_config. If it ever
    # stops, it would go back to overwriting the values uistate restored, and
    # the tabs would quietly forget themselves again. Probing the function that
    # does the work, not the module: inspect.getsource on a module needs a real
    # file on disk and is easy to fool.
    found["ui-config opt-out"] = None
    try:
        loadsave = importlib.import_module("modules.ui_loadsave")
        holder = getattr(loadsave, "UiLoadsave", loadsave)
        for attr in ("add_component", "add_block"):
            fn = getattr(holder, attr, None)
            if fn is None:
                continue
            if "do_not_save_to_config" in inspect.getsource(fn):
                found["ui-config opt-out"] = True
                break
            found["ui-config opt-out"] = False
    except Exception:
        pass

    return found


def report() -> list[str]:
    lines: list[str] = []
    gaps = missing()
    total = sum(len(v) for v in REQUIRED.values())
    if gaps:
        lines.append(
            f"[Impact ADetailer] compat: {total - len(gaps)}/{total} required WebUI "
            f"APIs present - MISSING:"
        )
        lines.extend(f"[Impact ADetailer]   {g}" for g in gaps)
        lines.append(
            "[Impact ADetailer]   this build is not supported; expect failures"
        )
    else:
        lines.append(
            f"[Impact ADetailer] compat: {total}/{total} required WebUI APIs present"
        )

    marks = {True: "yes", False: "no", None: "?"}
    detail = ", ".join(f"{k} {marks[v]}" for k, v in optional().items())
    lines.append(f"[Impact ADetailer] compat: {detail}")
    return lines


def log_report() -> None:
    try:
        for line in report():
            print(line)
    except Exception as exc:
        print(f"[Impact ADetailer] compat check failed to run: {exc}")
