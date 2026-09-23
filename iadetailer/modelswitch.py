"""Swap the loaded checkpoint / VAE for one detailer tab, then put the
parent generation's weights back.

The inner img2img job copies `p.sd_model`. Reloading only `shared.sd_model`
and leaving `p.sd_model` pointing at the previous engine would sample the
crop with the model the base image was made from. Both have to move, and
both have to move back — including when a tab raises halfway through the
faces.

Switching is done once per tab, not once per detection. A 4 GB reload on
every face is not a feature.
"""

from __future__ import annotations

from typing import Any

SAME_CHECKPOINT = "Use same checkpoint"
SAME_VAE = "Use same VAE"


def _opts():
    try:
        from modules import shared

        return getattr(shared, "opts", None)
    except Exception:
        return None


def list_checkpoints() -> list[str]:
    names: list[str] = [SAME_CHECKPOINT]
    try:
        from modules import sd_models

        tiles = []
        if hasattr(sd_models, "checkpoint_tiles"):
            try:
                tiles = list(sd_models.checkpoint_tiles())
            except TypeError:
                tiles = list(sd_models.checkpoint_tiles(use_shorts=False))
        elif hasattr(sd_models, "checkpoints_list"):
            tiles = [getattr(info, "title", name) for name, info in sd_models.checkpoints_list.items()]
        for tile in tiles:
            text = str(tile or "").strip()
            if text and text not in names:
                names.append(text)
    except Exception:
        pass
    return names


def list_vaes() -> list[str]:
    names: list[str] = [SAME_VAE, "None", "Automatic"]
    try:
        from modules import sd_vae

        try:
            sd_vae.refresh_vae_list()
        except Exception:
            pass
        for name in getattr(sd_vae, "vae_dict", {}) or {}:
            text = str(name or "").strip()
            if text and text not in names:
                names.append(text)
    except Exception:
        pass
    return names


def _current_checkpoint() -> str:
    opts = _opts()
    if opts is not None:
        for attr in ("sd_model_checkpoint", "sd_checkpoint"):
            value = getattr(opts, attr, None)
            if value:
                return str(value)
    try:
        from modules import shared

        model = getattr(shared, "sd_model", None)
        info = getattr(model, "sd_checkpoint_info", None)
        title = getattr(info, "title", None) or getattr(info, "model_name", None)
        if title:
            return str(title)
    except Exception:
        pass
    return ""


def _current_vae() -> str:
    opts = _opts()
    if opts is not None:
        value = getattr(opts, "sd_vae", None)
        if value:
            return str(value)
    try:
        from modules import sd_vae

        loaded = getattr(sd_vae, "loaded_vae_file", None)
        if loaded:
            from pathlib import Path

            return Path(str(loaded)).name
    except Exception:
        pass
    return "Automatic"


def _resolve_checkpoint(name: str):
    from modules import sd_models

    match = None
    for fn in (
        "get_closet_checkpoint_match",
        "get_closest_checkpoint_match",
        "get_checkpoint",
    ):
        getter = getattr(sd_models, fn, None)
        if callable(getter):
            try:
                match = getter(name)
            except Exception:
                match = None
            if match is not None:
                return match
    listing = getattr(sd_models, "checkpoints_list", None) or {}
    if name in listing:
        return listing[name]
    lowered = name.lower()
    for title, info in listing.items():
        if str(title).lower() == lowered or str(title).lower().startswith(lowered):
            return info
    return None


def _reload_checkpoint(info) -> None:
    from modules import sd_models

    try:
        sd_models.reload_model_weights(info=info)
        return
    except TypeError:
        pass
    try:
        sd_models.reload_model_weights(sd_models.model_data.sd_model, info)
        return
    except Exception:
        pass
    sd_models.reload_model_weights()


def _vae_path(name: str):
    if name in ("", SAME_VAE, "Automatic", "None", "Baked VAE"):
        return None
    try:
        from modules import sd_vae

        table = getattr(sd_vae, "vae_dict", None) or {}
        if name in table:
            return table[name]
        lowered = name.lower()
        for key, path in table.items():
            if str(key).lower() == lowered or str(path).lower().endswith(lowered):
                return path
    except Exception:
        pass
    return name


def _reload_vae(name: str) -> None:
    from modules import sd_vae, shared

    path = _vae_path(name)
    # Forge Neo: reload_vae_weights(vae: str) -> bool
    try:
        if sd_vae.reload_vae_weights(name if path is None else str(path)):
            return
    except TypeError:
        pass
    except Exception:
        pass
    # A1111 / ReForge: reload_vae_weights(sd_model=None, vae_file=path)
    try:
        sd_vae.reload_vae_weights(shared.sd_model, vae_file=path)
        return
    except TypeError:
        pass
    sd_vae.reload_vae_weights(vae_file=path)


def _swap_forge_vae_module(name: str, previous: list | None) -> list | None:
    """Neo keeps the extra VAE on opts.forge_additional_modules.

    Replacing the VAE entry (not the text encoder) is what process_images
    itself does when a job carries an sd_vae override. We do the same so a
    tab that asked for a different VAE does not keep sampling with the
    parent's one after reload_vae_weights has already run.
    """
    opts = _opts()
    if opts is None or not hasattr(opts, "forge_additional_modules"):
        return previous
    try:
        from modules import sd_vae
        import os

        vae_names = {str(k) for k in (getattr(sd_vae, "vae_dict", None) or {})}
        modules = list(opts.forge_additional_modules or [])
        if previous is None:
            previous = list(modules)
        kept = []
        for item in modules:
            base = os.path.basename(str(item))
            if base in vae_names or str(item) in vae_names:
                continue
            kept.append(item)
        if name not in ("", SAME_VAE, "Automatic", "None", "Baked VAE"):
            path = _vae_path(name)
            if path:
                kept.append(str(path))
        opts.forge_additional_modules = kept
        return previous
    except Exception:
        return previous


def architecture_of(model: Any) -> str:
    """Coarse family of the loaded engine.

    Used only to refuse ControlNet when a tab swaps SDXL ↔ Anima / Flux / SD1.
    Unknown stays unknown — that is not treated as a mismatch.
    """
    if model is None:
        return "unknown"
    flags = (
        ("is_anima", "anima"),
        ("is_flux", "flux"),
        ("is_sd3", "sd3"),
        ("is_sdxl", "sdxl"),
        ("is_sd2", "sd2"),
        ("is_sd1", "sd1"),
        ("is_sd15", "sd1"),
    )
    for attr, key in flags:
        try:
            if getattr(model, attr, False):
                return key
        except Exception:
            pass
    try:
        unet = getattr(getattr(model, "forge_objects", None), "unet", None)
        name = type(getattr(unet, "model", unet) or model).__name__.lower()
    except Exception:
        name = type(model).__name__.lower()
    for needle, key in (
        ("anima", "anima"),
        ("flux", "flux"),
        ("qwen", "qwen"),
        ("wan", "wan"),
        ("sdxl", "sdxl"),
        ("sd3", "sd3"),
        ("sd1", "sd1"),
    ):
        if needle in name:
            return key
    return "unknown"


def _bind_model(p: Any) -> None:
    """Point the parent processing object at whatever is now loaded."""
    try:
        from modules import shared

        model = getattr(shared, "sd_model", None)
        if model is None:
            return
        p.sd_model = model
    except Exception:
        pass


class ModelSwap:
    def __init__(self, p: Any):
        self.p = p
        self._ckpt_prev: str | None = None
        self._vae_prev: str | None = None
        self._modules_prev: list | None = None
        self._did_ckpt = False
        self._did_vae = False
        self.arch_before = "unknown"
        self.arch_after = "unknown"
        self.arch_mismatch = False

    def apply(self, unit: Any) -> str:
        """Load what the tab asked for. Returns a short label for the sheet."""
        try:
            from modules import shared

            self.arch_before = architecture_of(getattr(shared, "sd_model", None))
        except Exception:
            self.arch_before = "unknown"
        notes: list[str] = []
        if getattr(unit, "use_separate_checkpoint", False):
            wanted = str(getattr(unit, "checkpoint", "") or SAME_CHECKPOINT)
            if wanted not in ("", SAME_CHECKPOINT):
                self._switch_checkpoint(wanted)
                notes.append(f"checkpoint {wanted}")
        if getattr(unit, "use_separate_vae", False):
            wanted = str(getattr(unit, "vae", "") or SAME_VAE)
            if wanted not in ("", SAME_VAE):
                self._switch_vae(wanted)
                notes.append(f"VAE {wanted}")
        try:
            from modules import shared

            self.arch_after = architecture_of(getattr(shared, "sd_model", None))
        except Exception:
            self.arch_after = "unknown"
        self.arch_mismatch = (
            self.arch_before != "unknown"
            and self.arch_after != "unknown"
            and self.arch_before != self.arch_after
        )
        try:
            self.p.iad_arch_mismatch = self.arch_mismatch
            self.p.iad_arch_before = self.arch_before
            self.p.iad_arch_after = self.arch_after
        except Exception:
            pass
        if self.arch_mismatch:
            print(
                f"[Impact ADetailer] checkpoint family changed "
                f"{self.arch_before} -> {self.arch_after}; "
                "ControlNet on this tab is skipped"
            )
        return ", ".join(notes)

    def restore(self) -> None:
        failed: list[str] = []
        try:
            if self._did_vae:
                self._restore_vae()
        except Exception as exc:
            failed.append(f"VAE ({exc})")
            print(f"[Impact ADetailer] could not restore the parent VAE: {exc}")
        try:
            if self._did_ckpt:
                self._restore_checkpoint()
        except Exception as exc:
            failed.append(f"checkpoint ({exc})")
            print(f"[Impact ADetailer] could not restore the parent checkpoint: {exc}")
        _bind_model(self.p)
        try:
            self.p.iad_arch_mismatch = False
        except Exception:
            pass
        if failed:
            try:
                from .settings import set_status

                set_status(
                    "Impact ADetailer: could not restore "
                    + " and ".join(failed)
                    + " — re-select the checkpoint in the WebUI"
                )
            except Exception:
                pass

    def _switch_checkpoint(self, name: str) -> None:
        current = _current_checkpoint()
        info = _resolve_checkpoint(name)
        if info is None:
            print(f"[Impact ADetailer] checkpoint not found: {name}")
            return
        title = getattr(info, "title", None) or name
        if current and (current == title or current.startswith(str(title).split(" ")[0])):
            return
        self._ckpt_prev = current
        opts = _opts()
        if opts is not None and hasattr(opts, "sd_model_checkpoint"):
            opts.sd_model_checkpoint = title
        _reload_checkpoint(info)
        self._did_ckpt = True
        _bind_model(self.p)

    def _restore_checkpoint(self) -> None:
        if not self._ckpt_prev:
            return
        info = _resolve_checkpoint(self._ckpt_prev)
        opts = _opts()
        if opts is not None and hasattr(opts, "sd_model_checkpoint"):
            opts.sd_model_checkpoint = self._ckpt_prev
        if info is not None:
            _reload_checkpoint(info)
        self._did_ckpt = False

    def _switch_vae(self, name: str) -> None:
        self._vae_prev = _current_vae()
        self._modules_prev = _swap_forge_vae_module(name, None)
        opts = _opts()
        if opts is not None and hasattr(opts, "sd_vae"):
            opts.sd_vae = name
        _reload_vae(name)
        self._did_vae = True
        _bind_model(self.p)

    def _restore_vae(self) -> None:
        opts = _opts()
        if self._modules_prev is not None and opts is not None and hasattr(opts, "forge_additional_modules"):
            opts.forge_additional_modules = list(self._modules_prev)
        if self._vae_prev and opts is not None and hasattr(opts, "sd_vae"):
            opts.sd_vae = self._vae_prev
        if self._vae_prev:
            _reload_vae(self._vae_prev)
        else:
            try:
                from modules import sd_vae

                if hasattr(sd_vae, "restore_vae_weights"):
                    sd_vae.restore_vae_weights()
            except Exception:
                pass
        self._did_vae = False
