from __future__ import annotations

ORDINALS = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th")
SECTION = ("impact_adetailer", "Impact ADetailer")


def _opts():
    try:
        from modules import shared

        return getattr(shared, "opts", None)
    except Exception:
        return None


def get_opt(name: str, default):
    opts = _opts()
    if opts is None:
        return default
    return getattr(opts, name, default)


def max_tabs() -> int:
    n = int(get_opt("iad_max_tabs", 4) or 4)
    return max(1, min(8, n))


def extra_model_dirs() -> list[str]:
    raw = str(get_opt("iad_extra_models_dir", "") or "")
    out = []
    for chunk in raw.replace(";", "|").split("|"):
        chunk = chunk.strip()
        if chunk:
            out.append(chunk)
    return out


def output_dir() -> str:
    return str(get_opt("iad_output_dir", "") or "")


def save_before_images() -> bool:
    return bool(get_opt("iad_save_before_images", False))


def sort_bboxes() -> str:
    return str(get_opt("iad_sort_bboxes", "Area (large to small)") or "Area (large to small)")


def same_seed() -> bool:
    return bool(get_opt("iad_same_seed", False))


def offline_detectors() -> bool:
    return bool(get_opt("iad_offline_detectors", True))


def trace_network() -> str:
    return str(get_opt("iad_trace_network", "Off") or "Off")


def force_hf_offline() -> bool:
    return bool(get_opt("iad_force_hf_offline", False))


def detector_device() -> str:
    return str(get_opt("iad_detector_device", "Automatic") or "Automatic")


def unload_detectors() -> bool:
    return bool(get_opt("iad_unload_detectors", False))


def quiet_logs() -> bool:
    return bool(get_opt("iad_quiet_logs", True))


def log(message: str) -> None:
    """Informational line, silenced by the Quiet terminal logs option.
    Warnings and errors always print."""
    if not quiet_logs():
        print(f"[Impact ADetailer] {message}")


def remember_ui() -> bool:
    return bool(get_opt("iad_remember_ui", True))


def show_comparer() -> bool:
    return bool(get_opt("iad_show_comparer", True))


def max_preview_sheets() -> int:
    n = int(get_opt("iad_max_preview_sheets", 16) or 16)
    return max(0, min(64, n))


def set_status(text: str | None) -> None:
    """Text under the progress bar. None clears nothing; pass a string."""
    try:
        from modules import shared

        state = getattr(shared, "state", None)
        if state is None:
            return
        state.textinfo = text
    except Exception:
        pass


def register_settings() -> None:
    import gradio as gr
    from modules import shared

    OptionInfo = shared.OptionInfo
    section = SECTION

    def add(key, info):
        shared.opts.add_option(key, info)

    max_info = OptionInfo(
        4,
        "Max tabs (requires Reload UI)",
        gr.Slider,
        {"minimum": 1, "maximum": 8, "step": 1},
        section=section,
    )
    if hasattr(max_info, "needs_reload_ui"):
        max_info = max_info.needs_reload_ui()
    add("iad_max_tabs", max_info)

    extra = OptionInfo(
        "",
        "Extra paths to scan for detector models, separated by vertical bars | "
        "(e.g. D:\\models\\adetailer|E:\\ultralytics)",
        gr.Textbox,
        {"lines": 2},
        section=section,
    )
    if hasattr(extra, "needs_reload_ui"):
        extra = extra.needs_reload_ui()
    add("iad_extra_models_dir", extra)

    add(
        "iad_output_dir",
        OptionInfo(
            "",
            "Optional output directory for before/after copies (leave empty to use only temp)",
            gr.Textbox,
            {"lines": 1},
            section=section,
        ),
    )
    add(
        "iad_save_before_images",
        OptionInfo(
            False,
            "Also write the comparer's before/after pair, as PNG, into the output "
            "directory above. This is not the 'Save the image before detailing' "
            "checkbox in the panel - that one saves a single copy into your normal "
            "outputs folder and needs no directory set here",
            section=section,
        ),
    )
    add(
        "iad_sort_bboxes",
        OptionInfo(
            "Area (large to small)",
            "Sort bounding boxes by",
            gr.Radio,
            {"choices": ["None", "Position (left to right)", "Position (center to edge)", "Area (large to small)"]},
            section=section,
        ),
    )
    add("iad_same_seed", OptionInfo(False, "Use the same seed for every tab", section=section))
    add(
        "iad_offline_detectors",
        OptionInfo(
            True,
            "Keep YOLO/Ultralytics offline (do not contact huggingface.co / GitHub when loading .pt files)",
            section=section,
        ),
    )
    add(
        "iad_detector_device",
        OptionInfo(
            "Automatic",
            "Device for the YOLO detector",
            gr.Radio,
            {"choices": ["Automatic", "cuda", "cpu"]},
            section=section,
        ),
    )
    add(
        "iad_unload_detectors",
        OptionInfo(
            False,
            "Unload detector models after each generation (frees VRAM, costs a reload next run)",
            section=section,
        ),
    )
    add(
        "iad_trace_network",
        OptionInfo(
            "Off",
            "Trace outgoing connections (prints to the console which code opens "
            "them — use this when an antivirus warns about huggingface.co and you "
            "want to know what is actually calling out)",
            gr.Radio,
            {"choices": ["Off", "Model hosts only", "Everything"]},
            section=section,
        ),
    )
    add(
        "iad_force_hf_offline",
        OptionInfo(
            False,
            "Force huggingface/transformers into offline mode. Stops the antivirus "
            "warning at the source, but it is PROCESS-WIDE: any other extension "
            "that downloads a model will fail while this is on",
            section=section,
        ),
    )
    add("iad_quiet_logs", OptionInfo(True, "Quiet terminal logs", section=section))
    add(
        "iad_remember_ui",
        OptionInfo(
            True,
            "Remember the tab settings you last pressed Generate with, and restore "
            "them next time (the WebUI's own ui-config cannot do this: it keys on "
            "labels, which every tab shares)",
            section=section,
        ),
    )
    add("iad_show_comparer", OptionInfo(True, "Show the before/after splitter under the result gallery", section=section))
    add(
        "iad_max_preview_sheets",
        OptionInfo(
            16,
            "Max process-image sheets appended to the gallery (0 = no cap). "
            "Detection + crops on a multi-face batch can add dozens of images "
            "that are not saved",
            gr.Slider,
            {"minimum": 0, "maximum": 64, "step": 1},
            section=section,
        ),
    )
