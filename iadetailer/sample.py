from __future__ import annotations

import re
from copy import copy
from typing import Any

from PIL import Image


class _QuietScripts:
    alwayson_scripts = []
    selectable_scripts = []
    scripts = []
    titles = []
    callback_map = {}

    def __bool__(self):
        return True

    def setup_scrips(self, *args, **kwargs):
        return None

    def setup_scripts(self, *args, **kwargs):
        return None

    def before_process(self, *args, **kwargs):
        return None

    def process(self, *args, **kwargs):
        return None

    def process_batch(self, *args, **kwargs):
        return None

    def postprocess(self, *args, **kwargs):
        return None

    def postprocess_image(self, *args, **kwargs):
        return None

    def postprocess_batch(self, *args, **kwargs):
        return None

    def before_process_batch(self, *args, **kwargs):
        return None

    def after_extra_networks_activate(self, *args, **kwargs):
        return None

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _round8(n: int) -> int:
    n = max(64, int(n))
    return n - (n % 8)


def _attach_empty_scripts(dst) -> None:
    runner = _QuietScripts()
    for attr in ("scripts", "scripts_value", "_scripts"):
        try:
            setattr(dst, attr, runner)
        except Exception:
            pass
    dst.script_args = []
    if hasattr(dst, "script_args_value"):
        try:
            dst.script_args_value = []
        except Exception:
            pass


def _copy_processing(src, **overrides):
    from modules.processing import StableDiffusionProcessingImg2Img

    dst = StableDiffusionProcessingImg2Img(
        sd_model=getattr(src, "sd_model", None),
        outpath_samples=getattr(src, "outpath_samples", None),
        outpath_grids=getattr(src, "outpath_grids", None),
        prompt=overrides.get("prompt", src.prompt),
        negative_prompt=overrides.get("negative_prompt", src.negative_prompt),
        seed=overrides.get("seed", src.seed),
        sampler_name=overrides.get("sampler_name", src.sampler_name),
        scheduler=overrides.get("scheduler", getattr(src, "scheduler", "Automatic")),
        batch_size=1,
        n_iter=1,
        steps=overrides.get("steps", src.steps),
        cfg_scale=overrides.get("cfg_scale", src.cfg_scale),
        width=overrides["width"],
        height=overrides["height"],
        init_images=[overrides["init"]],
        denoising_strength=overrides["denoising_strength"],
        resize_mode=0,
    )
    dst.do_not_save_samples = True
    dst.do_not_save_grid = True
    # The parent's override_settings were already applied by the outer job.
    # Carrying them in here would re-apply them (and can trigger a checkpoint
    # reload) once per detection.
    dst.override_settings = {}
    dst.override_settings_restore_afterwards = False
    dst.disable_extra_networks = True
    if hasattr(dst, "is_iadetailer_job"):
        dst.is_iadetailer_job = True
    else:
        try:
            dst.is_iadetailer_job = True
        except Exception:
            pass
    for name in (
        "clip_skip",
        "sd_model_checkpoint",
        "sd_vae",
        "token_merging_ratio",
        "token_merging_ratio_hr",
        "s_min_uncond",
        "s_churn",
        "s_tmin",
        "s_tmax",
        "s_noise",
        "eta",
        "hr_second_pass_steps",
    ):
        if hasattr(src, name):
            try:
                setattr(dst, name, copy(getattr(src, name)))
            except Exception:
                try:
                    setattr(dst, name, getattr(src, name))
                except Exception:
                    pass
    # Fresh cond cache in the A1111/Forge shape: a 2-slot list, not None and
    # not []. process_images does cached_c[0] without a guard — None gave
    # "NoneType is not subscriptable", [] gave "list index out of range".
    for attr, value in (
        ("cached_c", [None, None]),
        ("cached_uc", [None, None]),
        ("all_prompts", [dst.prompt]),
        ("all_negative_prompts", [dst.negative_prompt]),
        ("all_seeds", [dst.seed]),
        ("all_subseeds", [int(dst.seed)]),
        ("iteration", 0),
        ("n_iter", 1),
        ("batch_index", 0),
    ):
        if hasattr(dst, attr):
            try:
                setattr(dst, attr, value)
            except Exception:
                pass
    _attach_empty_scripts(dst)
    # LoRAs attached from the extra-networks UI live on extra_network_data
    # even after their <lora:> tags have been stripped from the resolved
    # prompt. Copy the parent's map so the inner job can activate the same
    # set; tags in the tab prompt still work on top of that.
    parent_nets = getattr(src, "extra_network_data", None)
    if parent_nets:
        try:
            dst.extra_network_data = copy(parent_nets)
        except Exception:
            try:
                dst.extra_network_data = parent_nets
            except Exception:
                pass
    if overrides.get("mask") is not None:
        dst.image_mask = overrides["mask"]
        dst.mask_blur = 0
        dst.inpainting_fill = 1
        dst.inpaint_full_res = False
        dst.inpainting_mask_invert = 0
        if overrides.get("soft_mask") and hasattr(dst, "mask_round"):
            # A1111 rounds the latent mask to 0/1 unless this is off, which
            # would throw away a feathered noise mask before it does anything.
            dst.mask_round = False
    return dst


_NETWORK_TAG = re.compile(r"<(lora|lyco|lycoris|hypernet)\s*:", re.IGNORECASE)


def _wants_extra_networks(*prompts: str, parent: Any = None) -> bool:
    if any(_NETWORK_TAG.search(text or "") for text in prompts):
        return True
    data = getattr(parent, "extra_network_data", None) if parent is not None else None
    if not data:
        return False
    try:
        return any(bool(v) for v in (data.values() if isinstance(data, dict) else data))
    except Exception:
        return True


def _restore_parent_networks(p: Any) -> None:
    """The inner job deactivates every extra network on its way out, and that
    would strip the LoRAs the outer generation is still holding. Put the
    parent's back before the next image is sampled."""
    try:
        from modules import extra_networks

        data = getattr(p, "extra_network_data", None)
        if not data or getattr(p, "disable_extra_networks", False):
            return
        extra_networks.activate(p, data)
    except Exception as exc:
        print(f"[Impact ADetailer] could not restore the parent's extra networks: {exc}")


_CN_WARNED = False


def _arm_controlnet(job: Any, parent: Any, cfg: dict) -> bool:
    """True when ControlNet is now wired into this job."""
    global _CN_WARNED
    try:
        from . import controlnet

        if controlnet.attach(
            job, parent,
            model=cfg.get("model", "None"),
            module=cfg.get("module", "None"),
            weight=cfg.get("weight", 1.0),
            start=cfg.get("start", 0.0),
            end=cfg.get("end", 1.0),
        ):
            return True
        if not _CN_WARNED:
            _CN_WARNED = True
            reason = (
                "this build has no lib_controlnet (Forge-family ControlNet)"
                if not controlnet.available()
                else "the parent generation is not running the ControlNet script"
            )
            print(f"[Impact ADetailer] ControlNet skipped for the detail pass: {reason}")
    except Exception as exc:
        if not _CN_WARNED:
            _CN_WARNED = True
            print(f"[Impact ADetailer] ControlNet could not be attached: {exc}")
    return False


_RUNNER_WARNED = False


def _warn_if_runner_replaced(job: Any) -> None:
    """process_images on some builds rebinds p.scripts to the global runner.

    The empty runner and is_iadetailer_job are what stop Hires-fix (and this
    extension) from firing once per face. If either is gone after the call,
    say so once — a crop that looks like a full txt2img is this latch failing.
    """
    global _RUNNER_WARNED
    if _RUNNER_WARNED:
        return
    flag = bool(getattr(job, "is_iadetailer_job", False))
    runner = getattr(job, "scripts", None) or getattr(job, "scripts_value", None)
    titles = []
    try:
        for script in getattr(runner, "alwayson_scripts", []) or []:
            try:
                titles.append(script.title() or type(script).__name__)
            except Exception:
                titles.append(type(script).__name__)
    except Exception:
        pass
    quiet = isinstance(runner, _QuietScripts)
    extra = [t for t in titles if t and t != "ControlNet"]
    if flag and (quiet or not extra):
        return
    _RUNNER_WARNED = True
    why = []
    if not flag:
        why.append("is_iadetailer_job was cleared")
    if extra:
        why.append("alwayson scripts rebound: " + ", ".join(extra[:8]))
    elif not quiet and runner is not None:
        why.append("script runner is no longer the empty one")
    print(
        "[Impact ADetailer] WARNING: the inner img2img job lost its isolation "
        f"({'; '.join(why) or 'unknown'}). A later WebUI may be resetting "
        "p.scripts inside process_images."
    )


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "out of memory" in text or "outofmemory" in text or "cuda error" in text


def sample_crop(
    p: Any,
    crop: Image.Image,
    mask: Image.Image | None,
    denoise: float,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    soft_mask: bool = False,
    controlnet: dict | None = None,
) -> Image.Image:
    # With force_inpaint on, Impact Pack samples a large crop at its own size,
    # which can be bigger than anything the card has run before. Rather than
    # killing the whole generation, back off and try again smaller.
    steps_down = (1.0, 0.7, 0.5)
    for i, shrink in enumerate(steps_down):
        try:
            return _sample_once(
                p, crop, mask, denoise, steps, cfg, sampler_name, scheduler,
                prompt, negative_prompt, seed, shrink, soft_mask, controlnet,
            )
        except Exception as exc:
            if i == len(steps_down) - 1 or not _is_oom(exc):
                raise
            nxt = steps_down[i + 1]
            print(
                f"[Impact ADetailer] out of memory on a {crop.size[0]}x{crop.size[1]} "
                f"region, retrying it at {int(nxt * 100)}% size"
            )
            try:
                from modules import devices

                devices.torch_gc()
            except Exception:
                pass
    return crop


def _sample_once(
    p: Any,
    crop: Image.Image,
    mask: Image.Image | None,
    denoise: float,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    shrink: float = 1.0,
    soft_mask: bool = False,
    controlnet: dict | None = None,
) -> Image.Image:
    from modules import processing

    width, height = crop.size
    if shrink < 1.0:
        width, height = int(width * shrink), int(height * shrink)
    width, height = _round8(width), _round8(height)
    work = crop
    work_mask = mask
    if work.size != (width, height):
        work = work.resize((width, height), Image.Resampling.LANCZOS)
        if work_mask is not None:
            work_mask = work_mask.resize((width, height), Image.Resampling.BILINEAR)

    job = _copy_processing(
        p,
        init=work,
        width=width,
        height=height,
        denoising_strength=float(denoise),
        steps=int(steps),
        cfg_scale=float(cfg),
        sampler_name=sampler_name,
        scheduler=scheduler,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=int(seed),
        mask=work_mask,
        soft_mask=soft_mask,
    )

    # The inner job normally runs with an empty script runner: that is what
    # stops every other alwayson script from firing once per detection. A tab
    # that asked for ControlNet gets a runner back with ControlNet in it, and
    # nothing else.
    def arm() -> None:
        if controlnet and _arm_controlnet(job, p, controlnet):
            return
        _attach_empty_scripts(job)

    arm()

    # A <lora:...> tag in a tab's own prompt only does something if the inner
    # job is allowed to load it. Left off otherwise: activating networks per
    # detection costs a load/unload cycle each time.
    use_networks = _wants_extra_networks(prompt, negative_prompt, parent=p)
    job.disable_extra_networks = not use_networks

    processed = None
    last_err = None
    try:
        # process_images() first, on purpose. On all three forks it runs
        # fix_p_invalid_sampler_and_scheduler(p) before handing over, plus the
        # backend's own model/prompt-cache step (manage_model_and_prompt_cache
        # on Forge and Neo, reload_model_weights on ReForge). Calling
        # process_images_inner directly skips both: an unrecognised
        # sampler/scheduler goes through uncorrected, and on the Forge family
        # the nested job can end up holding a stale model handle.
        for fn in (
            processing.process_images,
            getattr(processing, "process_images_inner", None),
        ):
            if fn is None:
                continue
            try:
                arm()
                processed = fn(job)
                _warn_if_runner_replaced(job)
                break
            except Exception as exc:
                if _is_oom(exc):
                    # The other entry point would run out of memory at the same
                    # size too. Let the caller shrink instead.
                    raise
                last_err = exc
                processed = None
        if processed is None:
            if last_err is not None:
                raise last_err
            raise RuntimeError("inner img2img returned no result")
        if not getattr(processed, "images", None):
            return crop
        out = processed.images[0]
        if out.size != crop.size:
            out = out.resize(crop.size, Image.Resampling.LANCZOS)
        return out.convert("RGB")
    finally:
        # process_images_inner never closes the job, so without this every
        # detection leaves a Processing object holding a model reference and
        # its cached conditioning behind.
        try:
            job.close()
        except Exception:
            pass
        if use_networks or getattr(p, "extra_network_data", None):
            _restore_parent_networks(p)
