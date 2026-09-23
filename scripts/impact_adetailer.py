from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _stale in [
    _name for _name in list(sys.modules)
    if _name == "iadetailer" or _name.startswith("iadetailer.")
]:
    del sys.modules[_stale]

from iadetailer.offline import apply as _iad_offline

_iad_offline()

from modules import script_callbacks, scripts
from modules.processing import StableDiffusionProcessing
from modules.scripts import PostprocessImageArgs

from iadetailer.compare import CompareStore
from iadetailer.detect import clear_model_cache

def _clear_all_caches() -> None:
    clear_model_cache()
    try:
        from iadetailer.sam import clear_sam_cache

        clear_sam_cache()
    except Exception:
        pass
    try:
        from iadetailer.clip_filter import clear_clip_cache

        clear_clip_cache()
    except Exception:
        pass

from iadetailer.pipeline import run_units
from iadetailer import netwatch, preview as preview_mod
from iadetailer.offline import force_huggingface_offline
from iadetailer.settings import (
    force_hf_offline,
    register_settings,
    remember_ui,
    trace_network,
    unload_detectors,
)
from iadetailer import uistate
from iadetailer.ui import (
    build_ui,
    infotext_fields,
    infotext_params,
    parse_args,
    tab_name,
)
from iadetailer.version import banner, build_id
from iadetailer import compat


print(banner())
compat.log_report()


_HF_OFFLINE_DONE = [False]


def _apply_network_options() -> None:
    """Re-read both network options every generation, so flipping one in
    Settings takes effect without a restart."""
    try:
        netwatch.sync(trace_network())
    except Exception:
        pass
    try:
        if force_hf_offline() and not _HF_OFFLINE_DONE[0]:
            _HF_OFFLINE_DONE[0] = True
            patched = force_huggingface_offline()
            print(
                "[Impact ADetailer] huggingface/transformers forced offline"
                + (" (live modules patched too)" if patched else "")
            )
    except Exception:
        pass


def _set_status(text):
    """Show what is happening under the progress bar. Returns the old value."""
    try:
        from modules import shared

        state = getattr(shared, "state", None)
        if state is None:
            return None
        previous = getattr(state, "textinfo", None)
        state.textinfo = text
        return previous
    except Exception:
        return None


IAD_PREFIX = "Impact ADetailer"


def _clean_infotext(p) -> str:
    """The generation parameters for the image currently in postprocess_image,
    minus this extension's own keys.

    process() has already written those keys into p.extra_generation_params, so
    a naive create_infotext would stamp the BEFORE copy with the settings of a
    detail pass it never went through - and dragging it back into txt2img would
    silently re-enable the tabs. They go back afterwards, because the finished
    image is saved from the same dict a moment later and does want them.
    """
    from modules.processing import create_infotext

    extra = getattr(p, "extra_generation_params", None)
    removed = {}
    if isinstance(extra, dict):
        removed = {k: v for k, v in extra.items() if str(k).startswith(IAD_PREFIX)}
        for key in removed:
            extra.pop(key, None)
    try:
        return create_infotext(
            p,
            p.all_prompts,
            p.all_seeds,
            p.all_subseeds,
            iteration=int(getattr(p, "iteration", 0) or 0),
            position_in_batch=int(getattr(p, "batch_index", 0) or 0),
        )
    finally:
        if removed and isinstance(extra, dict):
            extra.update(removed)


def _save_before(p, image) -> None:
    """Write the pre-detailer copy beside the finished one.

    Same call the host uses for its own -before-face-restoration and
    -before-color-correction copies, so it lands in the same folder, under the
    same naming pattern, and obeys the same save_samples rules.
    """
    from modules import images, shared

    if hasattr(p, "save_samples") and not p.save_samples():
        return
    try:
        info = _clean_infotext(p)
    except Exception:
        info = None
    seeds = getattr(p, "seeds", None) or [getattr(p, "seed", -1)]
    prompts = getattr(p, "prompts", None) or [getattr(p, "prompt", "")]
    i = min(int(getattr(p, "batch_index", 0) or 0), len(seeds) - 1)
    images.save_image(
        image,
        p.outpath_samples,
        "",
        seeds[i],
        prompts[min(i, len(prompts) - 1)],
        shared.opts.samples_format,
        info=info,
        p=p,
        suffix="-before-adetailer",
    )


def _signature(image) -> bytes:
    """Cheap identity of an image, used to line up the gallery copy with the
    snapshot we detailed. Hashing instead of keeping raw pixels keeps a batch
    of large images from sitting in memory twice."""
    import hashlib

    try:
        return hashlib.blake2b(image.convert("RGB").tobytes(), digest_size=16).digest()
    except Exception:
        return b""


class Script(scripts.Script):
    def title(self):
        return "Impact ADetailer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        # Which tab this instance belongs to. The Script object is built once
        # per tab, so this is the reliable place to record it: by the time
        # process() runs there is nothing on `p` that says txt2img vs img2img.
        self.iad_tab = tab_name(is_img2img)
        widgets = build_ui(is_img2img)

        try:
            self.infotext_fields = infotext_fields(widgets)
        except Exception:
            self.infotext_fields = []
        return widgets

    def _ensure_run(self, p) -> None:
        """Open a fresh comparer run, once per generation, whichever hook gets
        here first. Stamped on `p` rather than tracked on self, because the
        Script instance is reused across generations while p is not.

        This runs even when the extension is switched off. Otherwise a plain
        generation leaves the previous run's pair sitting under a gallery it has
        nothing to do with, which is how the comparer ends up showing someone
        from three images ago."""
        if getattr(p, "iad_run", None) is not None:
            return
        try:
            p.iad_run = CompareStore.start_run()
        except Exception as exc:
            p.iad_run = 0
            print(f"[Impact ADetailer] could not start a comparer run: {exc}")

    def _remember(self, run, units) -> None:
        """Store what this generation was launched with, so the tabs come back
        the same way next session. Saved even when the extension is switched
        off, otherwise turning it off would never stick."""
        if not remember_ui():
            return
        try:
            uistate.save(getattr(self, "iad_tab", "txt2img"), run, units)
        except Exception as exc:
            print(f"[Impact ADetailer] could not remember the tab settings: {exc}")

    def process(self, p: StableDiffusionProcessing, *args):
        if getattr(p, "is_iadetailer_job", False):
            return
        self._results = {}
        self._ensure_run(p)
        _apply_network_options()
        run, units = parse_args(args)
        self._remember(run, units)
        if not (run.enable and any(u.enabled and u.model not in ("", "None") for u in units)):
            return
        # Written into the PNG's generation parameters, so a run can be read
        # back off the image later.
        try:
            p.extra_generation_params.update(infotext_params(units))
        except Exception:
            pass

    def postprocess_image(self, p: StableDiffusionProcessing, pp: PostprocessImageArgs, *args):
        if getattr(p, "is_iadetailer_job", False):
            return
        self._ensure_run(p)
        run, units = parse_args(args)
        if not run.enable:
            return
        if not any(u.enabled and u.model not in ("", "None") for u in units):
            return

        import numpy as np
        from PIL import Image

        # Snapshot of the image exactly as it arrives here, before we touch it.
        original = Image.fromarray(np.array(pp.image.convert("RGB")))
        work = Image.fromarray(np.array(original))
        previous_info = _set_status("Impact ADetailer")
        shot = preview_mod.attach(p, run.preview)
        try:
            detailed = run_units(work, units, p)
        except Exception as exc:
            import traceback

            print(f"[Impact ADetailer] failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return
        finally:
            _set_status(previous_info)
            preview_mod.detach(p)
            # The sheets go to the gallery even when the pass raised: a run that
            # fell over halfway is exactly when you want to see what it had
            # found. Every fork in this family concatenates extra_result_images
            # onto the result, after the real images.
            if shot.images:
                try:
                    if not isinstance(getattr(p, "extra_result_images", None), list):
                        p.extra_result_images = []
                    p.extra_result_images.extend(shot.images)
                except Exception as exc:
                    print(f"[Impact ADetailer] could not show the process images: {exc}")
        detailed = Image.fromarray(np.array(detailed.convert("RGB")))

        original_sig = _signature(original)
        pp.image = detailed

        if original_sig == _signature(detailed):
            # Nothing was detected, or the loop changed nothing. A comparer of
            # an image against itself is just noise - and so is a -before copy
            # that is byte-identical to the image saved next to it.
            return

        if run.save_before:
            try:
                _save_before(p, original)
            except Exception as exc:
                print(f"[Impact ADetailer] could not save the before image: {exc}")

        if getattr(self, "_results", None) is None:
            self._results = {}
        self._results[original_sig] = detailed
        try:
            CompareStore.save_pair(original=original, detailed=detailed)
        except Exception as exc:
            print(f"[Impact ADetailer] compare save failed: {exc}")

    def postprocess(self, p, processed, *args):
        # Some ReForge/Forge builds drop the postprocess_image result for the
        # copy that lands in the gallery. Patch only the images that are still
        # byte-identical to a snapshot we detailed during THIS run, so nothing
        # another extension produced gets clobbered.
        results = getattr(self, "_results", None)
        if results and unload_detectors():
            _clear_all_caches()
        if not results or processed is None or not getattr(processed, "images", None):
            return
        for i, img in enumerate(processed.images):
            try:
                hit = results.get(_signature(img))
            except Exception:
                continue
            if hit is not None:
                processed.images[i] = hit
        self._results = {}
        try:
            delattr(p, "iad_run")
        except Exception:
            pass


def _on_app_started(*args):
    app = None
    for item in args:
        if item is not None and (hasattr(item, "add_api_route") or hasattr(item, "router")):
            app = item
    if app is None:
        return
    # Catch anything that reaches out while the UI is still coming up, not just
    # during a generation.
    _apply_network_options()
    try:
        from fastapi import Response
        from fastapi.responses import JSONResponse
    except Exception:
        return

    def state():
        try:
            payload = CompareStore.state()
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)})
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    def image(slot: int = -1, side: str = "original", r: int = 0):
        side = "detailed" if str(side).lower().startswith("d") else "original"
        try:
            # A request left over from a previous generation must not be
            # answered with this generation's pixels.
            if int(r) != CompareStore.current_run():
                return Response(status_code=404)
            data = CompareStore.image_bytes(int(slot), side)
        except Exception:
            data = None
        if not data:
            return Response(status_code=404)
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    for path, fn in (
        ("/impact_adetailer/state", state),
        ("/impact_adetailer/image", image),
    ):
        try:
            app.add_api_route(path, fn, methods=["GET"])
        except Exception:
            pass


def _on_reload():
    CompareStore.cleanup()
    _clear_all_caches()


script_callbacks.on_ui_settings(register_settings)
script_callbacks.on_app_started(_on_app_started)
if hasattr(script_callbacks, "on_before_reload"):
    script_callbacks.on_before_reload(_on_reload)
